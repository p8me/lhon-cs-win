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
using System.Windows.Media.Imaging;

namespace LHON_Form
{
    public partial class Main_Form : Form
    {
        GifBitmapEncoder gifEnc = new GifBitmapEncoder();

        public Main_Form()
        {
            InitializeComponent();
            this.CenterToScreen();
            DoubleBuffered = true;

            chk_show_bound.CheckedChanged += (o, e) => update_show_opts();
            chk_show_tox.CheckedChanged += (o, e) => update_show_opts();
            btn_reset.Click += (s, e) =>
            {
                if (rate == null)
                {
                    append_stat_ln("You must preprocess the model before resetting the state.\n");
                    return;
                }
                if (sim_stat == sim_stat_enum.Running)
                {
                    append_stat_ln("You must stop the simulation before resetting the states.\n");
                    return;
                }
                reset_state(); set_btn_start_txt("&Start");
            };

            btn_start.Click += (s, e) =>
            {
                if (sim_stat == sim_stat_enum.None || sim_stat == sim_stat_enum.Paused)
                    start_sim();
                else if (sim_stat == sim_stat_enum.Running)
                    stop_sim(sim_stat_enum.Paused);
            };

            btn_redraw.Click += (s, e) => { if (sim_stat != sim_stat_enum.Running && !new_model_worker.IsBusy) new_model_worker.RunWorkerAsync(); };

            btn_preprocess.Click += (s, e) => preprocess_model();

            btn_clr.Click += (s, e) => txt_status.Text = "";

            cmb_sw_sel1.SelectedIndex = 0;
            cmb_sw_sel2.SelectedIndex = 0;

            chk_rec_avi.CheckedChanged += (s, e) =>
            {
                if (chk_rec_avi.Checked)
                {
                    chk_rec_avi.Text = "Recoding AVI";
                    //avi_file = @"Recordings\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + '(' + (im_size * im_size / 1e6).ToString("0.0") + "Mpix).avi";
                    //aviManager = new AviManager(avi_file, false);
                    //Avi.AVICOMPRESSOPTIONS options = new Avi.AVICOMPRESSOPTIONS();
                    //options.fccType = (uint)Avi.mmioStringToFOURCC("vids", 5);
                    //options.fccHandler = (uint)Avi.mmioStringToFOURCC("CVID", 5);
                    ////options.dwQuality = 1;
                    //aviStream = aviManager.AddVideoStream(options, 10, bmp);
                }
                else
                {
                    chk_rec_avi.Text = "Record AVI";
                    //aviManager.Close();
                    //Process.Start(avi_file);
                    var path = @"Recordings\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + '(' + (im_size * im_size / 1e6).ToString("0.0") + "Mpix).gif";
                    using (FileStream fs = new FileStream(path, FileMode.Create))
                        gifEnc.Save(fs);
                }
            };

            chk_save_sw_prog.CheckedChanged += (s, e) =>
            {
                if (chk_save_sw_prog.Checked) chk_save_sw_prog.Text = "Saving Sweep";
                else chk_save_sw_prog.Text = "Save Sweep";
            };

            chk_save_prog.CheckedChanged += (s, e) =>
            {
                if (chk_save_prog.Checked) chk_save_prog.Text = "Saving Progress";
                else chk_save_prog.Text = "Save Progress";
            };

            btn_export.Click += (s, e) => Export_model();

            append_stat_ln("Welcome to LHON-2D Simulation software!\n");

            btn_snapshot.Click += (s, e) =>
            {
                string adr = @"Snapshots\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + ".jpg";
                append_stat_ln("Snapshot saved to: " + adr);
                bmp.Save(adr);
            };
        }

        void update_show_opts()
        {
            show_opts[0] = chk_show_bound.Checked;
            show_opts[1] = chk_show_tox.Checked;

            gpu.CopyToDevice(show_opts, show_opts_dev);
            update_bmp_from_tox(true);
        }

        // ========================================= Start / Stop

        void start_sim()
        {
            if (rate == null)
            {
                append_stat_ln("You must preprocess the model before running simulation.\n");
                return;
            }
            if (sim_stat == sim_stat_enum.None || sim_stat == sim_stat_enum.Paused)
            {
                sim_stat = sim_stat_enum.Running;
                alg_worker.RunWorkerAsync();
                set_btn_start_txt("&Pause");
                update_bottom_stat("Simulation is Running");
            }
        }

        void stop_sim(sim_stat_enum stat)
        {
            if (sim_stat == sim_stat_enum.Running)
            {
                sim_stat = stat;
                if (sim_stat == sim_stat_enum.Paused) set_btn_start_txt("&Continue");
                else set_btn_start_txt("&Start");

                update_bottom_stat("Simulation is " + sim_stat.ToString());
            }
        }

        // =========================== Low level GUI

        private void Main_Form_FormClosing(object sender, FormClosingEventArgs e)
        {
            sim_stat = sim_stat_enum.Paused;
            Thread.Sleep(10);
        }

        void set_btn_start_txt(string s)
        {
            if (InvokeRequired) Invoke(new Action(() => btn_start.Text = s));
            else btn_start.Text = s;
        }

        void update_n_neur_lbl()
        {
            if (InvokeRequired)
                Invoke(new Action(() => update_n_neur_lbl()));
            else
                lbl_n_Neurs.Text = mdl.n_neurs.ToString() + " (" +
                    (Math.Pow((mdl.nerve_r / real_model_nerve_r), 2) * real_model_num_neurs).ToString("0") + ")";
        }

        void update_mdl_prog(float prog)
        {
            update_bottom_stat("Generating Model ... " + (prog * 100).ToString("0.0") + " %");
        }

        void update_image_siz_lbl()
        {
            string s = string.Format("{0} x {0}", im_size);
            if (InvokeRequired) Invoke(new Action(() => lbl_image_siz.Text = s));
            else lbl_image_siz.Text = s;
        }

        void update_bottom_stat(string s)
        {
            statlbl.Text = s;
            if (InvokeRequired) Invoke(new Action(() => statlbl.Text = s));
            else statlbl.Text = s;
        }

        void update_stat_sw_lbl(string s)
        {
            if (InvokeRequired) Invoke(new Action(() => statlbl_sweep.Text = s));
            else statlbl_sweep.Text = s;
        }

        void append_stat(string s)
        {
            if (InvokeRequired) Invoke(new Action(() => txt_status.AppendText(s.Replace("\n", Environment.NewLine))));
            else txt_status.AppendText(s);
        }

        void append_stat_ln(string s) { append_stat(s + Environment.NewLine); }

        void update_gui_labels()
        {
            Invoke(new Action(() =>
            {
                lbl_itr.Text = iteration.ToString("0");
                lbl_tox.Text = (sum_tox / area_res_factor).ToString("0");
                lbl_areal_progress.Text = (areal_progress * 100).ToString("0.0") + "%";
                lbl_live_neur_perc.Text = ((float)num_live_neur[0] * 100 / mdl.n_neurs).ToString("0.0") + "%";
                var span = TimeSpan.FromSeconds(tt_sim.read() / 1000);
                lbl_el_time.Text = string.Format("{0}:{1:00}:{2:00}", (int)span.TotalHours, span.Minutes, span.Seconds);


                float itr_p_s = 0;
                if (sim_stat == sim_stat_enum.Running)
                    itr_p_s = iteration / tt_sim.read() * 1000;

                string s = itr_p_s.ToString("0.0");
                lbl_itr_s.Text = s;
                lbl_chron_progress.Text = (chron_progress * 100).ToString("0.0") + "%";

                float x = (float)iteration / last_itr;
                float rat = 0.3F;
                float m = itr_p_s / (1 - x * rat);
                float estimated_total_itr_s = (m + (m * rat) * (1F - x)) / 2;

                if (!float.IsInfinity(estimated_total_itr_s) && !float.IsNaN(estimated_total_itr_s) && estimated_total_itr_s > 0)
                {
                    span = TimeSpan.FromSeconds((last_itr - iteration) / estimated_total_itr_s);
                    lbl_rem_time.Text = string.Format("{0}:{1:00}:{2:00}", (int)span.TotalHours, span.Minutes, span.Seconds);
                }
            }));

            if (chk_neur_lvl.Checked)
            {
                for (int i = 0; i < mdl.n_neurs; i++)
                    if (live_neur[i] && tox_touch_neur[i] > 1)
                    {
                        if (tox_touch_neur_last[i] == tox_touch_neur[i]) continue;
                        neur_lbl[i].lbl = (tox_touch_neur[i] / neur_tol[i] * 100).ToString("0");
                        tox_touch_neur_last[i] = tox_touch_neur[i];
                    }
                    else if (i != first_neur_idx) neur_lbl[i].lbl = "";
            }

        }

        // =========================== Settings

        void init_settings()
        {
            txt_nerve_rad.TextChanged += (s, e) => mdl.nerve_r = read_float(s);
            txt_vein_rad.TextChanged += (s, e) => mdl.vein_rat = read_float(s);
            txt_max_rad.TextChanged += (s, e) => mdl.max_r = read_float(s);
            txt_min_rad.TextChanged += (s, e) => mdl.min_r = read_float(s);
            txt_clearance.TextChanged += (s, e) => mdl.clearance = read_float(s);
            txt_num_tries.TextChanged += (s, e) => mdl.num_tries = read_float(s);

            txt_resolution.TextChanged += (s, e) =>
            {
                setts.resolution = read_float(s);
                float temp = setts.resolution * (mdl.max_r + mdl.min_r) / 2;
                area_res_factor = Maxf(temp * temp * 2, 1F);
            };
            txt_Tol.TextChanged += (s, e) => setts.neur_tol_coeff = read_float(s);

            txt_neur_rate.TextChanged += (s, e) => setts.neur_rate = read_float(s);

            btn_save_model.Click += (s, e) =>
            {
                var fil_name = @"Models\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + ".mdat";
                Debug.WriteLine(fil_name);

                FileStream outFile = File.Create(fil_name);
                XmlSerializer formatter = XmlSerializer.FromTypes(new[] { mdl.GetType() })[0];
                formatter.Serialize(outFile, mdl);
                append_stat_ln("Model saved to " + fil_name);
            };

            btn_load_model.Click += (s, e) =>
            {
                var FD = new OpenFileDialog()
                {
                    InitialDirectory = @"Models\",
                    Title = "Load Model",
                    Filter = "Model Data files (*.mdat) | *.mdat",
                    RestoreDirectory = true,
                    AutoUpgradeEnabled = false
                };
                if (FD.ShowDialog() != DialogResult.OK) return;
                load_model(FD.FileName);
            };

            btn_save_setts.Click += (s, e) =>
            {
                var fil_name = @"Settings\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + ".sdat";
                Debug.WriteLine(fil_name);

                FileStream outFile = File.Create(fil_name);
                XmlSerializer formatter = XmlSerializer.FromTypes(new[] { setts.GetType() })[0];
                formatter.Serialize(outFile, setts);
                append_stat_ln("Settings saved to " + fil_name);
            };

            btn_load_setts.Click += (s, e) =>
            {
                var FD = new OpenFileDialog()
                {
                    InitialDirectory = @"Settings\",
                    Title = "Load Settings",
                    Filter = "Setting files (*.sdat) | *.sdat",
                    RestoreDirectory = true,
                    AutoUpgradeEnabled = false
                };
                if (FD.ShowDialog() != DialogResult.OK) return;
                load_settings(FD.FileName);
            };

            // =============== Init Value

            string[] fileEntries = Directory.GetFiles(@"Models\");
            if (fileEntries.Length > 0) load_model(fileEntries[fileEntries.Length - 1]);

            fileEntries = Directory.GetFiles(@"Settings\");
            if (fileEntries.Length > 0) load_settings(fileEntries[fileEntries.Length - 1]);

        }

        void load_model(string path)
        {
            if (!File.Exists(path)) return;
            //XmlSerializer formatter = new XmlSerializer(mdl.GetType()); // throws an exception during deep debuging
            // workaround:
            XmlSerializer formatter = XmlSerializer.FromTypes(new[] { mdl.GetType() })[0];
            FileStream aFile = new FileStream(path, FileMode.Open);
            byte[] buffer = new byte[aFile.Length];
            aFile.Read(buffer, 0, (int)aFile.Length);
            MemoryStream stream = new MemoryStream(buffer);
            mdl = (Model)formatter.Deserialize(stream);
            aFile.Close();

            update_mdl_and_setts_ui();

            //preprocess_model();
        }

        void load_settings(string path)
        {
            if (!File.Exists(path)) return;

            XmlSerializer formatter = XmlSerializer.FromTypes(new[] { setts.GetType() })[0];
            FileStream aFile = new FileStream(path, FileMode.Open);
            byte[] buffer = new byte[aFile.Length];
            aFile.Read(buffer, 0, (int)aFile.Length);
            MemoryStream stream = new MemoryStream(buffer);
            setts = (Setts)formatter.Deserialize(stream);
            aFile.Close();

            update_mdl_and_setts_ui();
        }

        void update_mdl_and_setts_ui()
        {
            txt_nerve_rad.Text = mdl.nerve_r.ToString();
            txt_vein_rad.Text = mdl.vein_rat.ToString();
            txt_max_rad.Text = mdl.max_r.ToString();
            txt_min_rad.Text = mdl.min_r.ToString();
            txt_clearance.Text = mdl.clearance.ToString();
            txt_num_tries.Text = mdl.num_tries.ToString();

            txt_resolution.Text = setts.resolution.ToString();
            txt_Tol.Text = setts.neur_tol_coeff.ToString();
            txt_neur_rate.Text = setts.neur_rate.ToString();
        }

        float read_float(object o)
        {
            TextBox txtB = (TextBox)o;
            float num;
            if (!float.TryParse(txtB.Text, out num))
            {
                //txtB.Text = "0";
                //txtB.SelectionStart = 0;
                //txtB.SelectionLength = txtB.Text.Length;
                return 0;
            }
            return num;
        }

        int read_int(object o)
        {
            TextBox txtB = (TextBox)o;
            int num;
            if (!int.TryParse(txtB.Text, out num))
            {
                //txtB.Text = "0";
                //txtB.SelectionStart = 0;
                //txtB.SelectionLength = txtB.Text.Length;
                return 0;
            }
            return num;
        }

        // =========================== BMP Management
        unsafe private void update_bmp_from_tox(bool reload_tox_dev)
        {
            if (reload_tox_dev)
            {
                gpu.CopyToDevice(tox, tox_dev);
                gpu.Launch(blocks_per_grid, threads_per_block).gpu_fill_bmp(tox_dev, bmp_dev, show_opts_dev, touch_pix_dev);
                gpu.CopyFromDevice(bmp_dev, bmp_bytes);
            }

            else
                for (int y = 0; y < im_size; y++)
                    for (int x = 0; x < im_size; x++)
                        fixed (byte* pix_addr = &bmp_bytes[y, x, 0])
                            update_bmp_pix(tox[x, y], pix_addr, show_opts, touch_pix[x, y]);

            update_bmp_from_bmp_bytes_and_rec();
        }

        [Cudafy]
        unsafe public static bool update_bmp_pix(float tx, byte* pix_addr, bool[] opts, UInt16 touch)
        {
            int r = 0, g = 0, b = 0;
            int v = (int)(tx * 255);

            if (touch == 1 && opts[0])
                b = 255;
            else
            {
                if (opts[1])
                {
                    if (v < 64) { r = 0; g = 4 * v; b = 255; }
                    else if (v < 128) { r = 0; b = 255 + 4 * (64 - v); g = 255; }
                    else if (v < 192) { r = 4 * (v - 128); b = 0; g = 255; }
                    else { g = 255 + 4 * (192 - v); b = 0; r = 255; }
                }
                else
                    r = 255 - v;
            }

            pix_addr[0] = (byte)b;
            pix_addr[1] = (byte)g;
            pix_addr[2] = (byte)r;
            return false; // must return something!
        }

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

        unsafe void update_bmp_from_bmp_bytes_and_rec()
        {
            if (InvokeRequired)
                Invoke(new Action(() => update_bmp_from_bmp_bytes_and_rec()));
            else
            {
                fixed (byte* dat = &bmp_bytes[0, 0, 0])
                    CopyMemory(bmp_scan0, (IntPtr)dat, (uint)bmp_bytes.Length);
                picB.Image = bmp;

                if (sim_stat == sim_stat_enum.Running && chk_rec_avi.Checked)
                {
                    //aviStream.AddFrame((Bitmap)bmp.Clone());

                    var bmh = bmp.GetHbitmap();
                    var opts = BitmapSizeOptions.FromWidthAndHeight(200, 200);
                    var src = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(bmh, IntPtr.Zero, System.Windows.Int32Rect.Empty, opts);
                    gifEnc.Frames.Add(BitmapFrame.Create(src));
                }
            }
        }

        int picB_offx, picB_offy;
        float picB_ratio;

        private void picB_Resize(object sender, EventArgs e)
        {
            float picW = picB.Size.Width;
            float picH = picB.Size.Height;

            float asp_im = (float)im_size / (float)im_size,
                asp_box = picW / picH;

            if (asp_im > asp_box)
            {
                picB_ratio = picW / im_size;
                picB_offx = 0;
                picB_offy = (int)((picH - picB_ratio * (float)im_size) / 2f);
            }
            else
            {
                picB_ratio = picH / im_size;
                picB_offx = (int)((picW - picB_ratio * (float)im_size) / 2f);
                picB_offy = 0;
            }

        }

        private void picB_Paint(object sender, PaintEventArgs e)
        {

            if (neur_lbl != null)
            {
                // the X on the first neuron
                var nlbl0 = neur_lbl[first_neur_idx];
                SizeF textSize0 = e.Graphics.MeasureString(nlbl0.lbl, this.Font);
                e.Graphics.DrawString(nlbl0.lbl, this.Font, Brushes.Beige, nlbl0.x * picB_ratio + picB_offx - (textSize0.Width / 2), nlbl0.y * picB_ratio + picB_offy - (textSize0.Height / 2));

                if (chk_neur_lvl.Checked)
                    for (int i = 0; i < mdl.n_neurs; i++)
                    {
                        var nlbl = neur_lbl[i];
                        if (show_neur_lvl[i] && i != first_neur_idx && nlbl.lbl.Length > 0)
                        {
                            SizeF textSize = e.Graphics.MeasureString(nlbl.lbl, this.Font);
                            e.Graphics.DrawString(nlbl.lbl, this.Font, Brushes.White, nlbl.x * picB_ratio + picB_offx - (textSize.Width / 2), nlbl.y * picB_ratio + picB_offy - (textSize.Height / 2));
                        }
                    }
            }

            //if (mdl_neur_lbl != null && mdl_neur_lbl.Length > 0)
            //    for (int i = 0; i < mdl_n_neurs; i++)
            //    {
            //        var lbli = mdl_neur_lbl[i];
            //        if (lbli != null)
            //        {
            //            SizeF textSize = e.Graphics.MeasureString(lbli.lbl, this.Font);
            //            float x = lbli.x * picB_ratio + picB_offx - (textSize.Width / 2);
            //            float y = lbli.y * picB_ratio + picB_offy - (textSize.Height / 2);
            //            e.Graphics.DrawString(lbli.lbl, this.Font, Brushes.White, x, y);
            //        }
            //    }
        }

        private void picB_Click(object sender, EventArgs e)
        {
            var mouseEventArgs = e as MouseEventArgs;
            if (mouseEventArgs != null)
            {
                int x = (int)((mouseEventArgs.X - picB_offx) / picB_ratio);
                int y = (int)((mouseEventArgs.Y - picB_offy) / picB_ratio);
                if (x >= 0 && x < im_size && y >= 0 && y < im_size)
                    mouse_click(x, y); //Debug.WriteLine("X= " + x + " Y= " + y);
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

    }
}

