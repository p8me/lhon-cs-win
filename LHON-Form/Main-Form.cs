using System;
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
            init_settings();

            mdl.min_r_abs = 0.19F / 2;
            mdl.max_r_abs = 6.87F / 2;

            if (init_gpu())
            {
                MessageBox.Show("No Nvidia GPU detected! This program requires an Nvidia GPU.", "Fatal Error");
                this.Close();
                return;
            }

            alg_worker.DoWork += (s, ev) => Run_Alg_GPU(); alg_worker.WorkerSupportsCancellation = true;
            new_model_worker.DoWork += (s, ev) => new_model();

            if (mdl.n_axons > 0 && mdl.n_axons < 100000 && setts.resolution > 0)
                preprocess_model();

            read_model_cdf_file();
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

                last_itr = (uint)(mdl.nerve_r * Math.Pow(setts.resolution, 4) * 1.6F);
                last_areal_prog = 1F - ((mdl.min_r + mdl.max_r) / mdl.nerve_r / 2) * ((mdl.min_r + mdl.max_r) / mdl.nerve_r / 2);
                tic();
            }

            gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId); // should be reloaded for reliability

            float[,] diff = new float[im_size, im_size];

            alg_prof.time(0);

            gui_iteration_period = (int)(50 * setts.resolution * mdl.min_r);

            tt_sim.start();

            while (true)
            {
                iteration++;

                bool update_gui = iteration % gui_iteration_period == 0;

                alg_prof.time(-1);

                //if ((iteration % (int)area_res_factor) == 0)

                gpu.Launch(mdl.n_axons / 16, 16).cuda_update_live(mdl.n_axons, tox_dev, rate_dev, detox_dev, tox_prod_dev, k_rate_dead_axon, k_detox_extra, death_tox_lim,
                    axons_cent_pix_dev, axons_inside_pix_dev, axons_inside_pix_idx_dev, axon_is_alive_dev, num_alive_axons_dev, death_itr_dev, iteration);
                if (en_prof) { gpu.Synchronize(); alg_prof.time(1); }


                gpu.Launch(blocks_per_grid, threads_per_block).cuda_diffusion(im_size, tox_dev, rate_dev, detox_dev, tox_prod_dev);
                if (en_prof) { gpu.Synchronize(); alg_prof.time(2); }

                if (update_gui)
                {
                    gpu.CopyFromDevice(axon_is_alive_dev, axon_is_alive);

                    // Calc tox_sum
                    //gpu.Set(sum_tox_dev);
                    //gpu.Launch(blocks_per_grid, threads_per_block).gpu_sum_tox(tox_dev, sum_tox_dev);
                    //gpu.CopyFromDevice(sum_tox_dev, out sum_tox);

                    update_gui_labels();

                    if (en_prof) { gpu.Synchronize(); alg_prof.time(3); }

                    gpu.Launch(blocks_per_grid, threads_per_block).gpu_fill_bmp(tox_dev, bmp_dev, show_opts_dev);
                    gpu.CopyFromDevice(bmp_dev, bmp_bytes);

                    gpu.CopyFromDevice(tox_dev, tox);

                    update_bmp_from_bmp_bytes_and_rec();
                    if (en_prof) alg_prof.time(4);

                }

                if (sim_stat != sim_stat_enum.Running) break;
                if (iteration == stop_iteration) stop_sim(sim_stat_enum.Paused);
            }

            tt_sim.pause();

            if (en_prof) alg_prof.report();
            else Debug.WriteLine("Sim took " + (toc() / 1000).ToString("0.000") + " secs\n");

        }

        // ============= Copy from GPU to CPU and vice-versa ==================

        private void load_gpu_from_cpu()
        {
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

            gpu.FreeAll();

            gpu.Synchronize();

            tox_dev = gpu.Allocate(tox); gpu.CopyToDevice(tox, tox_dev);
            rate_dev = gpu.Allocate(rate); gpu.CopyToDevice(rate, rate_dev);
            detox_dev = gpu.Allocate(detox); gpu.CopyToDevice(detox, detox_dev);
            tox_prod_dev = gpu.Allocate(tox_prod); gpu.CopyToDevice(tox_prod, tox_prod_dev);

            axons_cent_pix_dev = gpu.Allocate(axons_cent_pix); gpu.CopyToDevice(axons_cent_pix, axons_cent_pix_dev);
            axon_is_alive_dev = gpu.Allocate(axon_is_alive); gpu.CopyToDevice(axon_is_alive, axon_is_alive_dev);

            pix_idx_dev = gpu.Allocate(pix_idx); gpu.CopyToDevice(pix_idx, pix_idx_dev);

            num_alive_axons_dev = gpu.Allocate<int>(1); gpu.CopyToDevice(num_alive_axons, num_alive_axons_dev);
            death_itr_dev = gpu.Allocate(death_itr); gpu.CopyToDevice(death_itr, death_itr_dev);
            bmp_dev = gpu.Allocate(bmp_bytes); gpu.CopyToDevice(bmp_bytes, bmp_dev);

            sum_tox_dev = gpu.Allocate<float>(1);
            progress_dev = gpu.Allocate<float>(3);

            progression_image_sum_float_dev = gpu.Allocate<float>(prog_im_siz, prog_im_siz);
            progress_image_num_averaged_pix_dev = gpu.Allocate<uint>(prog_im_siz, prog_im_siz);
            progression_image_dev = gpu.Allocate<byte>(prog_im_siz, prog_im_siz);

            // ==================== Constants

            blocks_per_grid = new dim3((im_size) / threads_per_block_1D, (im_size) / threads_per_block_1D);
            threads_per_block = new dim3(threads_per_block_1D, threads_per_block_1D);
            show_opts_dev = gpu.Allocate(show_opts); gpu.CopyToDevice(show_opts, show_opts_dev);


            axons_inside_pix_dev = gpu.Allocate(axons_inside_pix); gpu.CopyToDevice(axons_inside_pix, axons_inside_pix_dev);
            axons_inside_pix_idx_dev = gpu.Allocate(axons_inside_pix_idx); gpu.CopyToDevice(axons_inside_pix_idx, axons_inside_pix_idx_dev);

            gpu.Synchronize();

            Debug.WriteLine("GPU used memory: " + (100.0 * (1 - (double)gpu.FreeMemory / (double)gpu.TotalMemory)).ToString("0.0") + " %\n");
        }

        private void load_cpu_from_gpu()
        {
            //GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

            gpu.CopyFromDevice(tox_dev, tox);
            gpu.CopyFromDevice(rate_dev, rate);
            gpu.CopyFromDevice(detox_dev, detox);
            gpu.CopyFromDevice(tox_prod_dev, tox_prod);

            gpu.CopyFromDevice(axon_is_alive_dev, axon_is_alive);
            gpu.CopyFromDevice(death_itr_dev, death_itr);
        }

        // ================== Helpers  =======================

        void init_bmp_write()
        {
            bmp = new Bitmap(im_size, im_size);
            Rectangle rect = new Rectangle(0, 0, im_size, im_size);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, bmp.PixelFormat);
            bmp_scan0 = bmpData.Scan0;
            bmp.UnlockBits(bmpData);
            bmp_bytes = new byte[im_size, im_size, 4];

            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                    bmp_bytes[y, x, 3] = 255;
            tox = new float[im_size, im_size];
        }

        void mouse_click(int x, int y)
        {
            if (sim_stat == sim_stat_enum.Running) return;

            // Sets the initial insult location
            init_insult[0] = (float)x / setts.resolution - (mdl.nerve_r + nerve_clear);
            init_insult[1] = (float)y / setts.resolution - (mdl.nerve_r + nerve_clear);

            reset_state();
        }

        // ==================== Reset State  =======================

        private void reset_state()
        {
            if (InvokeRequired)
                Invoke(new Action(() => reset_state()));
            else
            {
                // Identify first dying axon
                int min_dis = 1000000000;
                int iicx = (int)((init_insult[0] + mdl.nerve_r + nerve_clear) * setts.resolution);
                int iicy = (int)((init_insult[1] + mdl.nerve_r + nerve_clear) * setts.resolution);
                float min_first_r = float.Parse(txt_min_first_r.Text) * setts.resolution;
                for (int i = 0; i < mdl.n_axons; i++)
                {
                    int dx = (int)axons_coor[i, 0] - iicx;
                    int dy = (int)axons_coor[i, 1] - iicy;
                    int dis = (dx * dx + dy * dy);
                    if (min_dis > dis && axons_coor[i, 2] > min_first_r)
                    {
                        min_dis = dis;
                        first_axon_idx = i;
                    }

                    axon_is_alive[i] = true;
                    death_itr[i] = 0;
                }

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
                update_bmp_from_tox(true);
                picB_Resize(null, null);

                sim_stat = sim_stat_enum.None;

            }
        }
    }
}
