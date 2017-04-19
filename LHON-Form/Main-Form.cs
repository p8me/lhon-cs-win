﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using AviFile;
using System.Drawing.Drawing2D;
using System.Xml.Serialization;
using System.IO;

using Cudafy;
using Cudafy.Host;
using Cudafy.Atomics;
using Cudafy.Translator;
using System.Runtime.InteropServices;
using MathNet.Numerics.Distributions;

// Speed inside bundle: Fastest, outside bundles slower, boundaries slowest, 

namespace LHON_Form
{
    [System.ComponentModel.DesignerCategory("Code")]
    public partial class Main_Form : Form
    {
        int gui_iteration_period;

        private void Main_Form_Load(object sender, EventArgs e)
        {
            alg_worker.DoWork += (s, ev) => Run_Alg_GPU(); alg_worker.WorkerSupportsCancellation = true;
            new_model_worker.DoWork += (s, ev) => new_model();

            init_settings_gui();

            if (init_gpu())
            {
                MessageBox.Show("No Nvidia GPU detected! This program requires an Nvidia GPU.", "Fatal Error");
                this.Close();
                return;
            }

            string[] fileEntries = Directory.GetFiles(ProjectOutputDir + @"Models\");
            if (fileEntries.Length > 0) load_model(fileEntries[fileEntries.Length - 1]);

            fileEntries = Directory.GetFiles(ProjectOutputDir + @"Settings\");
            if (fileEntries.Length > 0) load_settings(fileEntries[fileEntries.Length - 1]);

            if (mdl.n_axons > 0 && mdl.n_axons < 100000 && setts.resolution > 0)
                preprocess_model();
        }

        // =============================== MAIN LOOP

        bool en_prof = false;

        unsafe private void Run_Alg_GPU()
        {
            if (iteration == 0)
            {
                load_gpu_from_cpu();

                progress_step = 1F / (float)progress_num_frames;

                next_areal_progress_snapshot = progress_step;
                next_chron_progress_snapshot = progress_step;

                areal_progression_image_stack_cnt = 0;
                chron_progression_image_stack_cnt = 0;

                tt_sim.restart();

                last_itr = (uint)(mdl_nerve_r * Math.Pow(setts.resolution, 4) * 1.6F);
                last_areal_prog = 1F - ((axon_min_r_mean + axon_max_r_mean) / mdl_nerve_r / 2) * ((axon_min_r_mean + axon_max_r_mean) / mdl_nerve_r / 2);
                tic();
            }

            gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId); // should be reloaded for reliability
            
            alg_prof.time(0);

            gui_iteration_period = 10; // Max(10, (int)pow2(im_size));

            tt_sim.start();

            while (true)
            {
                iteration++;

                bool update_gui = iteration % gui_iteration_period == 0;

                alg_prof.time(-1); 

                gpu.Launch(blocks_per_grid_1D_axons, threads_per_block_1D).cuda_update_live(mdl.n_axons, tox_dev, rate_dev, detox_dev, tox_prod_dev, k_rate_dead_axon, k_detox_extra, death_tox_lim,
                    axons_cent_pix_dev, axons_inside_pix_dev, axons_inside_pix_idx_dev, axon_surr_rate_dev, axon_surr_rate_idx_dev,
                    axon_is_alive_dev, axon_mask_dev, num_alive_axons_dev, death_itr_dev, iteration);
                if (en_prof) { gpu.Synchronize(); alg_prof.time(1); }
                
                gpu.Launch(blocks_per_grid_2D_pix, threads_per_block_1D).cuda_diffusion(pix_idx_dev, pix_idx_num, im_size,
                    tox_dev, rate_dev, detox_dev, tox_prod_dev);
                
                if (en_prof) { gpu.Synchronize(); alg_prof.time(2); }

                if (update_gui)
                {
                    gpu.CopyFromDevice(axon_is_alive_dev, axon_is_alive);

                    // Calc tox_sum for sanity check
                    gpu.Set(sum_tox_dev);
                    gpu.Launch(blocks_per_grid_2D_pix, threads_per_block_1D).cuda_tox_sum(pix_idx_dev, pix_idx_num, tox_dev, sum_tox_dev);
                    gpu.CopyFromDevice(sum_tox_dev, out sum_tox);

                    update_gui_labels();

                    if (en_prof) { gpu.Synchronize(); alg_prof.time(3); }

                    update_bmp_image();

                    if (en_prof) alg_prof.time(4);
                }

                if (sim_stat != sim_stat_enum.Running) break;
                if (iteration == stop_at_iteration) stop_sim(sim_stat_enum.Paused);
            }

            tt_sim.pause();

            if (en_prof) alg_prof.report();
            else Debug.WriteLine("Sim took " + (toc() / 1000).ToString("0.000") + " secs\n");
        }

        
        // ==================== Reset State  =======================

        private void reset_state()
        {
            if (InvokeRequired)
                Invoke(new Action(() => reset_state()));
            else
            {
                // Identify first dying axon
                //int min_dis = 1000000000;
                //int iicx = (int)((init_insult[0] + mdl_nerve_r) * setts.resolution + 1);
                //int iicy = (int)((init_insult[1] + mdl_nerve_r) * setts.resolution + 1);
                //float min_first_r = float.Parse(txt_min_first_r.Text) * setts.resolution;
                //for (int i = 0; i < mdl.n_axons; i++)
                //{
                //    int dx = (int)axons_coor[i, 0] - iicx;
                //    int dy = (int)axons_coor[i, 1] - iicy;
                //    int dis = (dx * dx + dy * dy);
                //    if (min_dis > dis && axons_coor[i, 2] > min_first_r)
                //    {
                //        min_dis = dis;
                //        first_axon_idx = i;
                //    }
                //    axon_is_alive[i] = true;
                //    death_itr[i] = 0;
                //}

                tox = (float[,])tox_init.Clone();
                rate = (float[,,])rate_init.Clone();
                detox = (float[,])detox_init.Clone();

                /* NO_GUI
                sum_tox = 0;
                for (int y = 0; y < im_size; y++)
                    for (int x = 0; x < im_size; x++)
                        sum_tox += tox_init[x, y];
                */

                iteration = 0;

                update_gui_labels();

                for (int i = 0; i < mdl.n_axons; i++) axon_lbl[i].lbl = "";
                //axon_lbl[first_axon_idx].lbl = "X";

                prog_im_siz = prog_im_siz_default;
                resolution_reduction_ratio = (double)prog_im_siz / (double)im_size;
                if (resolution_reduction_ratio > 1)
                {
                    resolution_reduction_ratio = 1;
                    prog_im_siz = (ushort)im_size;
                }
                areal_progression_image_stack = new byte[progress_num_frames, prog_im_siz, prog_im_siz];
                chron_progression_image_stack = new byte[progress_num_frames, prog_im_siz, prog_im_siz];

                areal_progress_chron_val = new float[progress_num_frames];
                chron_progress_areal_val = new float[progress_num_frames];

                num_alive_axons[0] = mdl.n_axons - 1;

                load_gpu_from_cpu();

                update_show_opts();

                gpu.CopyToDevice(tox, tox_dev);
                update_bmp_image();
                picB_Resize(null, null);

                sim_stat = sim_stat_enum.None;

            }
        }
    }
}
