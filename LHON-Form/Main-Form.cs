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

            //if (mdl.n_neurs > 0) preprocess_model();
        }

        
        // =============================== MAIN LOOP

        unsafe private void Run_Alg_GPU()
        {
            if (iteration == 0)
            {
                kill_neur(first_neur_idx);
                load_gpu_from_cpu();

                progress_step = 1F / (float)progress_num_frames;

                next_areal_progress_snapshot = progress_step;
                next_chron_progress_snapshot = progress_step;

                areal_progression_image_stack_cnt = 0;
                chron_progression_image_stack_cnt = 0;

                tt_sim.restart();

                last_itr = (uint)(mdl.nerve_r * Math.Pow(setts.resolution, 4) * 1.6F);
                last_areal_prog = 1F - ((mdl.min_r + mdl.max_r) / mdl.nerve_r / 2) * ((mdl.min_r + mdl.max_r) / mdl.nerve_r / 2);
            }

            gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId); // should be reloaded for reliability

            float[,] diff = new float[im_size, im_size];

            alg_prof.time(0);

            gui_iteration_period = (int)(10 * setts.resolution * mdl.min_r / 2);

            tt_sim.start();

            while (true)
            {
                iteration++;

                bool update_gui = iteration % gui_iteration_period == 0;

                alg_prof.time(-1);

                if ((iteration % (int)area_res_factor) == 0)
                {
                    gpu.Launch(1, 32).gpu_update_live_neurs(mdl.n_neurs, tox_dev, rate_dev, live_neur_dev, num_live_neur_dev, tox_touch_neur_dev, neur_tol_dev, neurs_bound_touch_pix_dev, neurs_bound_touch_npix_dev,
                       neurs_inside_pix_dev, neurs_inside_pix_idx_dev, locked_pix_dev, death_itr_dev, iteration);

                    gpu.Set(progress_dev);
                    gpu.Launch(block_s_r, thread_s_r).gpu_areal_progress(tox_dev, locked_pix_dev, progress_dev, areal_progress_lim);
                    gpu.CopyFromDevice(progress_dev, progress_dat);
                    gpu.CopyFromDevice(num_live_neur_dev, num_live_neur);

                    areal_progress = progress_dat[2] / progress_dat[0];
                    chron_progress = (float)iteration / last_itr;

                    if (progress_dat[1] == 0)
                    {
                        stop_sim(sim_stat_enum.Failed);
                        append_stat_ln(string.Format("Simulation failed after {0:0.0} secs.", tt_sim.read() / 1000F));
                        tt_sim.pause();
                    }

                    if (areal_progress > next_areal_progress_snapshot || areal_progress >= last_areal_prog)
                    {
                        areal_progress_chron_val[areal_progression_image_stack_cnt] = chron_progress;
                        if (areal_progress >= last_areal_prog) last_areal_prog = Single.PositiveInfinity;
                        next_areal_progress_snapshot += progress_step;
                        Take_Progress_Snapshot(areal_progression_image_stack, areal_progression_image_stack_cnt++);
                    }

                    if (chron_progress > next_chron_progress_snapshot)
                    {
                        chron_progress_areal_val[chron_progression_image_stack_cnt] = areal_progress;
                        next_chron_progress_snapshot += progress_step;
                        Take_Progress_Snapshot(chron_progression_image_stack, chron_progression_image_stack_cnt++);
                    }

                    // Automatic End of Simulation
                    if (iteration > last_itr + 5)
                    {
                        stop_sim(sim_stat_enum.Successful);
                        append_stat_ln(string.Format("Simulation Successful after {0:0.0} secs.", tt_sim.read() / 1000F));
                        tt_sim.pause();

                        if (!sweep_is_running && chk_save_prog.Checked)
                            Save_Progress(@"Progression\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + ".prgim");
                    }

                    gpu.Synchronize();
                    alg_prof.time(1);
                }
                gpu.Set(diff_dev); gpu.Synchronize();
                alg_prof.time(2);
                gpu.Launch(block_s_r, thread_s_r).gpu_calc_diff(tox_dev, rate_dev, locked_pix_dev, diff_dev); gpu.Synchronize();
                alg_prof.time(3);
                gpu.Launch(block_s_r, thread_s_r).gpu_calc_tox(tox_dev, rate_dev, locked_pix_dev, diff_dev); gpu.Synchronize();
                alg_prof.time(4);

                if (update_gui)
                {
                    tic();

                    gpu.CopyFromDevice(tox_touch_neur_dev, tox_touch_neur);
                    gpu.CopyFromDevice(live_neur_dev, live_neur);
                    gpu.Launch(block_s_r, thread_s_r).gpu_fill_bmp(tox_dev, bmp_dev, show_opts_dev, touch_pix_dev);
                    gpu.CopyFromDevice(bmp_dev, bmp_bytes);
                    gpu.Set(sum_tox_dev);
                    gpu.Launch(block_s_r, thread_s_r).gpu_sum_tox(tox_dev, sum_tox_dev);
                    gpu.CopyFromDevice(sum_tox_dev, out sum_tox); gpu.Synchronize();

                    alg_prof.time(5);

                    update_gui_labels();

                    update_bmp_from_bmp_bytes_and_rec();

                    alg_prof.time(6);
                }

                if (sim_stat != sim_stat_enum.Running) break;
                //if (iteration == 500) stop_sim(sim_stat_enum.Failed);
            }

            tt_sim.pause();

            alg_prof.report();
        }

        // =============================== Copy from GPU to CPU and vice-versa

        private void load_gpu_from_cpu()
        {
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

            gpu.FreeAll();

            gpu.Synchronize();

            tox_dev = gpu.Allocate(tox); gpu.CopyToDevice(tox, tox_dev);
            rate_dev = gpu.Allocate(rate); gpu.CopyToDevice(rate, rate_dev);
            locked_pix_dev = gpu.Allocate(locked_pix); gpu.CopyToDevice(locked_pix, locked_pix_dev);
            tox_touch_neur_dev = gpu.Allocate(tox_touch_neur); gpu.CopyToDevice(tox_touch_neur, tox_touch_neur_dev);
            live_neur_dev = gpu.Allocate(live_neur); gpu.CopyToDevice(live_neur, live_neur_dev);
            num_live_neur_dev = gpu.Allocate<int>(1); gpu.CopyToDevice(num_live_neur, num_live_neur_dev);
            death_itr_dev = gpu.Allocate(death_itr); gpu.CopyToDevice(death_itr, death_itr_dev);
            bmp_dev = gpu.Allocate(bmp_bytes); gpu.CopyToDevice(bmp_bytes, bmp_dev);

            diff_dev = gpu.Allocate<float>(im_size, im_size);
            sum_tox_dev = gpu.Allocate<float>(1);
            progress_dev = gpu.Allocate<float>(3);

            progression_image_sum_float_dev = gpu.Allocate<float>(prog_im_siz, prog_im_siz);
            progress_image_num_averaged_pix_dev = gpu.Allocate<uint>(prog_im_siz, prog_im_siz);
            progression_image_dev = gpu.Allocate<byte>(prog_im_siz, prog_im_siz);

            // ==================== Constants

            block_s_r = new dim3((im_size) / Block_Size, (im_size) / Block_Size);
            thread_s_r = new dim3(Block_Size, Block_Size);
            show_opts_dev = gpu.Allocate(show_opts); gpu.CopyToDevice(show_opts, show_opts_dev);
            touch_pix_dev = gpu.Allocate(touch_pix); gpu.CopyToDevice(touch_pix, touch_pix_dev);

            neurs_inside_pix_dev = gpu.Allocate(neurs_inside_pix); gpu.CopyToDevice(neurs_inside_pix, neurs_inside_pix_dev);
            neurs_inside_pix_idx_dev = gpu.Allocate(neurs_inside_pix_idx); gpu.CopyToDevice(neurs_inside_pix_idx, neurs_inside_pix_idx_dev);

            neurs_inside_npix_dev = gpu.Allocate(neurs_inside_npix); gpu.CopyToDevice(neurs_inside_npix, neurs_inside_npix_dev);
            neurs_bound_touch_pix_dev = gpu.Allocate(neurs_bound_touch_pix); gpu.CopyToDevice(neurs_bound_touch_pix, neurs_bound_touch_pix_dev);
            neurs_bound_touch_npix_dev = gpu.Allocate(neurs_bound_touch_npix); gpu.CopyToDevice(neurs_bound_touch_npix, neurs_bound_touch_npix_dev);
            neur_tol_dev = gpu.Allocate(neur_tol); gpu.CopyToDevice(neur_tol, neur_tol_dev);

            gpu.Synchronize();

            //Debug.WriteLine("GPU Free memory: " + 100.0 * (double)gpu.FreeMemory / (double)gpu.TotalMemory + " %\n");
        }

        private void load_cpu_from_gpu()
        {
            //GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

            gpu.CopyFromDevice(tox_dev, tox);
            gpu.CopyFromDevice(rate_dev, rate);
            gpu.CopyFromDevice(locked_pix_dev, locked_pix);
            gpu.CopyFromDevice(tox_touch_neur_dev, tox_touch_neur);
            gpu.CopyFromDevice(live_neur_dev, live_neur);
            gpu.CopyFromDevice(death_itr_dev, death_itr);
        }

        // =============================== Helpers

        private void kill_neur(int idx)
        {
            if (mdl.n_neurs <= idx) return;

            for (int i = neurs_inside_pix_idx[idx]; i < neurs_inside_pix_idx[idx + 1]; i++)
                locked_pix[neurs_inside_pix[i, 0], neurs_inside_pix[i, 1]]--;


            if (idx != first_neur_idx) neur_lbl[idx].lbl = "";

            death_itr[idx] = iteration;
            live_neur[idx] = false;
        }

        int round_block_siz(int siz)
        {
            return siz / Block_Size * Block_Size;
        }

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
            touch_pix = new UInt16[im_size, im_size];
        }

        void mouse_click(int x, int y)
        {
            if (sim_stat == sim_stat_enum.Running) return; //iteration != 0 || 
                                                           // set the initial cell which dies

            init_insult[0] = (float)x / setts.resolution - (mdl.nerve_r + nerve_clear);
            init_insult[1] = (float)y / setts.resolution - (mdl.nerve_r + nerve_clear);

            reset_state();
        }

        // =============================== Reset State

        private void reset_state()
        {
            if (InvokeRequired)
                Invoke(new Action(() => reset_state()));
            else
            {
                // find first neur
                int min_dis = 1000000000;
                int iicx = (int)((init_insult[0] + mdl.nerve_r + nerve_clear) * setts.resolution);
                int iicy = (int)((init_insult[1] + mdl.nerve_r + nerve_clear) * setts.resolution);

                float min_first_r = float.Parse(txt_min_first_r.Text) * setts.resolution;

                for (int i = 0; i < mdl.n_neurs; i++)
                {
                    int dx = (int)neurs_coor[i, 0] - iicx;
                    int dy = (int)neurs_coor[i, 1] - iicy;
                    int dis = (dx * dx + dy * dy);
                    if (min_dis > dis && neurs_coor[i, 2] > min_first_r)
                    {
                        min_dis = dis;
                        first_neur_idx = i;
                    }
                }

                sum_tox = 0;
                for (int y = 0; y < im_size; y++)
                    for (int x = 0; x < im_size; x++)
                    {
                        tox[x, y] = tox_init[x, y];
                        sum_tox += tox_init[x, y];
                        rate[x, y] = rate_init[x, y];
                        locked_pix[x, y] = locked_pix_init[x, y];
                    }
                iteration = 0;

                //btn_start.Text = "&Start";

                for (int i = 0; i < mdl.n_neurs; i++)
                {
                    live_neur[i] = true;
                    death_itr[i] = 0;
                    tox_touch_neur[i] = 0;
                }

                update_gui_labels();

                for (int i = 0; i < mdl.n_neurs; i++) neur_lbl[i].lbl = "";
                neur_lbl[first_neur_idx].lbl = "X";

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

                num_live_neur[0] = mdl.n_neurs - 1;

                load_gpu_from_cpu();

                update_show_opts();
                update_bmp_from_tox(true);
                picB_Resize(null, null);

                sim_stat = sim_stat_enum.None;

            }
        }

        // ============= Save Progress Image

        private void Take_Progress_Snapshot(byte[,,] dest, uint frame)
        {
            if (InvokeRequired)
                Invoke(new Action(() => Take_Progress_Snapshot(dest, frame)));
            else
            {
                gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

                gpu.Set(progression_image_sum_float_dev);
                gpu.Set(progress_image_num_averaged_pix_dev);
                gpu.Launch(block_s_r, thread_s_r).gpu_progress_image_1(tox_dev, locked_pix_dev, progression_image_sum_float_dev, progress_image_num_averaged_pix_dev, resolution_reduction_ratio);
                gpu.Launch(new dim3(prog_im_siz, prog_im_siz), 1).gpu_progress_image_2(tox_dev, locked_pix_dev, progression_image_sum_float_dev, progress_image_num_averaged_pix_dev, progression_image_dev, prog_im_siz);

                byte[,] progression_image = new byte[prog_im_siz, prog_im_siz];
                gpu.CopyFromDevice(progression_image_dev, progression_image);
                gpu.Synchronize();

                for (int i = 0; i < prog_im_siz; i++)
                    for (int j = 0; j < prog_im_siz; j++)
                        dest[frame, i, j] = progression_image[i, j];
                
                //Rectangle bounds = picB.Bounds;
                //var org = picB.PointToScreen(new Point(0, 0));

                //using (Bitmap bitmap = new Bitmap(bounds.Width, bounds.Height))
                //{
                //    using (Graphics g = Graphics.FromImage(bitmap))
                //    {
                //        g.CopyFromScreen(org, Point.Empty, bounds.Size);
                //    }
                //    string pth = @"Recordings\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + ".jpg";
                //    bitmap.Save(pth, ImageFormat.Jpeg);
                //}
            }
        }

        private void Save_Progress(string progression_fil_name)
        {
            using (FileStream fileStream = new FileStream(progression_fil_name, FileMode.Append, FileAccess.Write, FileShare.None))
            {
                using (BinaryWriter writer = new BinaryWriter(fileStream))
                {
                    writer.Write(mdl.nerve_r);
                    writer.Write(mdl.max_r);
                    writer.Write(mdl.min_r);
                    writer.Write(mdl.clearance);
                    writer.Write(mdl.n_neurs);
                    writer.Write(setts.resolution);
                    writer.Write(setts.neur_tol_coeff);
                    writer.Write(setts.neur_rate);
                    writer.Write(im_size);
                    writer.Write(prog_im_siz);

                    writer.Write(init_insult[0]);
                    writer.Write(init_insult[1]);
                    writer.Write(progress_num_frames);
                    writer.Write(tt_sim.read());
                    writer.Write(last_itr);

                    for (int m = 0; m < progress_num_frames; m++)
                        writer.Write(areal_progress_chron_val[m]);

                    for (int m = 0; m < progress_num_frames; m++)
                        writer.Write(chron_progress_areal_val[m]);

                    gpu.CopyFromDevice(death_itr_dev, death_itr);

                    for (int m = 0; m < mdl.n_neurs; m++)
                    {
                        float x = (mdl.neur_cor[m][0] / mdl.nerve_r / 2 + 0.5F) * 256F;
                        writer.Write((byte)(x));
                        float y = (mdl.neur_cor[m][1] / mdl.nerve_r / 2 + 0.5F) * 256F;
                        writer.Write((byte)(y));
                        writer.Write((byte)(mdl.neur_cor[m][2] * 40));
                        float r = (float)death_itr[m] / (float)last_itr * 256F;
                        writer.Write((byte)(r));
                    }

                    for (int m = 0; m < progress_num_frames; m++)
                        for (int i = 0; i < prog_im_siz; i++)
                            for (int j = 0; j < prog_im_siz; j++)
                                writer.Write(areal_progression_image_stack[m, i, j]);

                    for (int m = 0; m < progress_num_frames; m++)
                        for (int i = 0; i < prog_im_siz; i++)
                            for (int j = 0; j < prog_im_siz; j++)
                                writer.Write(chron_progression_image_stack[m, i, j]);

                    writer.Flush();

                    //append_stat_ln("Sim Progress saved to " + progression_fil_name);

                }
            }
        }

        // ================ Mouse Drawing =======================

        bool _mousePressed = false;
        private void picB_MouseDown(object sender, MouseEventArgs e) { _mousePressed = true; }
        private void picB_MouseUp(object sender, MouseEventArgs e) { _mousePressed = false; }
        private void picB_MouseMove(object sender, MouseEventArgs e)
        {
            if (_mousePressed) { } // Debug.WriteLine("dragging");
        }

        // =======================================

        private void btn_sweep_Click(object sender, EventArgs e)
        {

            if (sweep_is_running)
            {
                stop_sweep_req = true;
                stop_sim(sim_stat_enum.Paused);
                btn_sweep.Text = "S&weep";
                append_stat_ln("Sweeping Terminated by User!");
            }
            else sweep();
        }

        enum param_select
        {
            Repeat,
            Nerve_Rad,
            Min_Rad,
            Max_Rad,
            Clearance,
            Resolution,
            Tolerance,
            Neur_Rate,
            Insult_Rad,
            Insult_Peri
        };

        async void sweep()
        {
            if (sim_stat == sim_stat_enum.Running) return;

            try
            {
                int delay_ms = 2000;

                int sweep_repetitions1, sweep_repetitions2 = 0;
                float start1 = 0, end1 = 0;
                float start2 = 0, end2 = 0;
                int selection1 = cmb_sw_sel1.SelectedIndex;
                int selection2 = cmb_sw_sel2.SelectedIndex;

                // Get first dimension of sweep
                try
                {
                    string[] values = txt_sw_range1.Text.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    sweep_repetitions1 = int.Parse(values[0]);
                    if (selection1 > 0)
                    {
                        start1 = float.Parse(values[1]);
                        if (sweep_repetitions1 > 1) end1 = float.Parse(values[2]);
                    }
                }
                catch
                {
                    MessageBox.Show("Not enough or bad input for Sweep command!");
                    return;
                }
                // Get Second dimension
                try
                {
                    string[] values = txt_sw_range2.Text.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    sweep_repetitions2 = int.Parse(values[0]);
                    if (selection2 > 0)
                    {
                        start2 = float.Parse(values[1]);
                        if (sweep_repetitions2 > 1) end2 = float.Parse(values[2]);
                    }
                }
                catch { }

                btn_sweep.Text = "S&top";
                sweep_is_running = true;

                string parameter_name1 = cmb_sw_sel1.Items[selection1].ToString();
                string parameter_name2 = cmb_sw_sel2.Items[selection2].ToString();

                int failures = 0;
                string dir_name = null;

                for (int i1 = 0; i1 < sweep_repetitions1; i1++)
                {
                    float val1 = sweep_upd_param((param_select)selection1, start1, end1, i1, sweep_repetitions1);
                    float val2 = 0;
                    append_stat_ln(parameter_name1 + " : " + val1.ToString());

                    bool regenerate_model = selection1 < (int)param_select.Resolution;
                    int i2 = 0;
                    do
                    {
                        if (sweep_repetitions2 > 0)
                        {
                            val2 = sweep_upd_param((param_select)selection2, start2, end2, i2, sweep_repetitions2);
                            append_stat_ln(parameter_name2 + " : " + val2.ToString());
                        }
                        update_mdl_and_setts_ui();

                        if (regenerate_model || (sweep_repetitions2 > 0 && selection2 < (int)param_select.Resolution))
                        {
                            regenerate_model = false;
                            // Regenerate Model
                            new_model_worker.RunWorkerAsync();
                            await Task.Delay(delay_ms);
                            while (sim_stat == sim_stat_enum.Running)
                            {
                                await Task.Delay(delay_ms);
                                if (stop_sweep_req) { stop_sweep_req = false; sweep_is_running = false; return; }
                            }
                        }
                        else
                            preprocess_model();

                        start_sim();
                        //Debug.WriteLine("simulation should've been started" + sim_stat);
                        while (sim_stat != sim_stat_enum.Successful && sim_stat != sim_stat_enum.Failed)
                        {
                            await Task.Delay(delay_ms);
                            if (stop_sweep_req) { stop_sweep_req = false; sweep_is_running = false; return; }
                        }
                        await Task.Delay(delay_ms / 5);

                        if (sim_stat == sim_stat_enum.Failed) failures++;
                        else // Successful
                        if (chk_save_sw_prog.Checked)
                        {
                            if (dir_name == null)
                            {
                                string par_nam = "(" + parameter_name1 + ")";
                                if (sweep_repetitions2 > 0)
                                    par_nam += "(" + parameter_name2 + ")";
                                dir_name = string.Format("Progression\\{0} {1}", DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss"), par_nam);
                                Directory.CreateDirectory(dir_name);
                            }
                            string par_val;
                            if ((param_select)selection1 == param_select.Repeat)
                                par_val = val1.ToString("(00)");
                            else
                                par_val = val1.ToString("(0.00)");
                            if (sweep_repetitions2 > 0)
                                if ((param_select)selection2 == param_select.Repeat)
                                    par_val += val2.ToString("(00)");
                                else
                                    par_val += val2.ToString("(0.00)");
                            Save_Progress(string.Format("{0}\\{1}.prgim", dir_name, par_val));
                        }
                        i2++;
                    }
                    while (i2 < sweep_repetitions2);
                }

                append_stat_ln(string.Format("Sweeping finished after {0} repetitions and {1} failure(s).", (sweep_repetitions2 > 0 ? sweep_repetitions2 * sweep_repetitions1 : sweep_repetitions1).ToString(), failures > 0 ? failures.ToString() : "no"));
                sweep_is_running = false;
                btn_sweep.Text = "S&weep";
            }
            catch (Exception e)
            {
                MessageBox.Show(e.ToString());
            }
        }

        float sweep_upd_param(param_select selection, float start, float end, int i, int sweep_repetitions)
        {
            float sig = end < start ? -1 : 1;
            float step_siz;
            if (selection == 0 || sweep_repetitions == 1) step_siz = 1;
            else step_siz = Math.Abs(end - start) / (float)(sweep_repetitions - 1);
            float val = start + sig * step_siz * i;

            switch (selection)
            {
                case param_select.Nerve_Rad:
                    mdl.nerve_r = val;
                    break;
                case param_select.Min_Rad:
                    mdl.min_r = val;
                    break;
                case param_select.Max_Rad:
                    mdl.max_r = val;
                    break;
                case param_select.Clearance:
                    mdl.clearance = val;
                    break;
                case param_select.Resolution:
                    setts.resolution = val;
                    break;
                case param_select.Tolerance:
                    setts.neur_tol_coeff = val;
                    break;
                case param_select.Neur_Rate:
                    setts.neur_rate = val;
                    break;
                case param_select.Insult_Rad:
                    init_insult[0] = mdl.nerve_r * (val * 2 - 1);
                    init_insult[1] = 0;
                    break;
                case param_select.Insult_Peri:
                    init_insult[0] = -mdl.nerve_r * (float)Math.Cos(val * Math.PI / 180);
                    init_insult[1] = -mdl.nerve_r * (float)Math.Sin(val * Math.PI / 180);
                    break;
            }
            return val;
        }

        void Export_model() // no death info, text file
        {
            string path = @"Exported\" + DateTime.Now.ToString("yyyy - MM - dd @HH - mm - ss") + ".txt";
            using (StreamWriter file = new StreamWriter(path, true))
            {
                file.WriteLine("{0}, {1}, {2}", mdl.nerve_r, mdl.vein_rat, mdl.clearance);
                for (int i = 0; i < mdl.n_neurs; i++)
                    file.WriteLine("{0}, {1}, {2}", mdl.neur_cor[i][0], mdl.neur_cor[i][1], mdl.neur_cor[i][2]);
            }
            append_stat_ln("Model exported to " + path);
        }

    }
}

