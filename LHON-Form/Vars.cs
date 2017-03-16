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

namespace LHON_Form
{
    public partial class Main_Form : Form
    {
        private BackgroundWorker alg_worker = new BackgroundWorker(), new_model_worker = new BackgroundWorker();
        AviManager aviManager;
        string avi_file;
        VideoStream aviStream;

        // Used as measure of comparison
        const float real_model_nerve_r = 750; // um
        const int real_model_num_neurs = 1200000;
        
        int first_neur_idx = 0;

        int max_set_size_bound, max_set_size_bound_touch;

        Bitmap bmp;
        int im_size;
        IntPtr bmp_scan0;
        byte[,,] bmp_bytes;

        bool[] show_opts = new bool[2],
            show_opts_dev = new bool[2];

        float[,] tox, tox_init, tox_dev;
        uint[,] locked_pix, locked_pix_dev, locked_pix_init;

        float sum_tox, areal_progress, chron_progress;
        float[] progress_dat = new float[3];

        const int progress_num_frames = 20;
        double resolution_reduction_ratio;
        ushort prog_im_siz, prog_im_siz_default = 100;
        byte[,] progression_image_dev;

        byte[,,] areal_progression_image_stack, chron_progression_image_stack;
        float[] areal_progress_chron_val, chron_progress_areal_val;
        uint areal_progression_image_stack_cnt, chron_progression_image_stack_cnt;

        float[,] progression_image_sum_float_dev;
        uint[,] progress_image_num_averaged_pix_dev;
        
        float areal_progress_lim;

        bool stop_sweep_req = false, sweep_is_running = false;

        float[,] diff_dev;
        float[] sum_tox_dev, progress_dev;

        float[,] rate, rate_init,
            rate_dev;

        UInt16[,] touch_pix,
            touch_pix_dev;

        uint iteration = 0;

        float[,] neurs_coor;

        ushort[,] neurs_inside_pix, neurs_inside_pix_dev;
        int[] neurs_inside_pix_idx, neurs_inside_pix_idx_dev;

        int[,,] neurs_bound_touch_pix, neurs_bound_touch_pix_dev;

        uint[] neurs_inside_npix, neurs_bound_touch_npix, death_itr,
            neurs_inside_npix_dev, neurs_bound_touch_npix_dev, death_itr_dev;

        bool[] live_neur,
            live_neur_dev;

        int[] num_live_neur_dev, num_live_neur = new int[1];

        bool[] show_neur_lvl;

        float[] neur_tol, tox_touch_neur, tox_touch_neur_last,
            neur_tol_dev, tox_touch_neur_dev;

        int nerve_clear = 4; // clearance of nerve from image borders in length units

        float area_res_factor = 1;

        float[] init_insult = new float[2] { 0, 0 };

        const int Block_Size = 12;
        dim3 block_s_r, thread_s_r;
        byte[,,] bmp_dev;
        
        enum sim_stat_enum { None, Running, Paused, Successful, Failed };
        sim_stat_enum sim_stat = sim_stat_enum.None;
        
        private class neur_lbl_class
        {
            public string lbl;
            public float x;
            public float y;
        }

        neur_lbl_class[] neur_lbl;

        public class Model
        {
            public float nerve_r, vein_rat, min_r, max_r,
                min_r_abs,
                max_r_abs,
                clearance;
            public int n_neurs;
            public float num_tries;
            public List<float[]> neur_cor;
        }

        public class Setts
        {
            public float resolution;
            public float neur_tol_coeff;
            public float neur_rate;
        }

        Model mdl = new Model();

        Setts setts = new Setts();

        // ============= Main loop stuff

        float progress_step, next_areal_progress_snapshot, next_chron_progress_snapshot;
        tic_toc tt_sim = new tic_toc();
        float sim_time = 0;
        uint last_itr;
        float last_areal_prog;

        // =============== Profiling Class =============

        class profile_class
        {
            double[] T = new double[100];
            Stopwatch sw = Stopwatch.StartNew();
            Stopwatch sw_tot = Stopwatch.StartNew();
            public void time(int idx) // Pass 0 start of main program to start tot_time
            {
                if (idx > 0)
                    T[idx] += sw.Elapsed.TotalMilliseconds;
                else if (idx == 0)
                {
                    Array.Clear(T, 0, T.Length);
                    sw_tot = Stopwatch.StartNew();
                }
                sw = Stopwatch.StartNew(); // Pass negative to reset only sw
            }
            public void report() // This will stop and conclude tot_time
            {
                sw_tot.Stop();
                sw.Stop();
                double tot_time = sw_tot.Elapsed.TotalMilliseconds;
                Debug.WriteLine("Total: " + (tot_time / 1000).ToString("0.0") + "s");
                for (int k = 0; k < T.Length; k++)
                    if (T[k] > 0)
                        Debug.WriteLine("{0}: {1} % ({2}ms)", k, (T[k] / tot_time * 100).ToString("0.0"), T[k].ToString("0"));
            }
        }
        profile_class gpu_prof = new profile_class(), alg_prof = new profile_class();



        // ======= Basic Math Functions =========

        private float Maxf(float v1, float v2)
        {
            return (v1 > v2) ? v1 : v2;
        }
        private float Minf(float v1, float v2)
        {
            return (v1 < v2) ? v1 : v2;
        }


        public int Max(int v1, int v2)
        {
            return (v1 > v2) ? v1 : v2;
        }

        private int Min(int v1, int v2)
        {
            return (v1 < v2) ? v1 : v2;
        }

        private float within_circle2(int x, int y, float xc, float yc, float rc)
        {
            float dx = x - xc;
            float dy = y - yc;
            return rc * rc - (dx * dx + dy * dy);
        }

        // ========== tic toc ==========

        Stopwatch sw = new Stopwatch();
        void tic()
        {
            sw = Stopwatch.StartNew();
        }

        float toc()
        {
            float t = sw.ElapsedMilliseconds;
            tic();
            return t;
        }

        class tic_toc
        {
            Stopwatch sw = new Stopwatch();
            public void restart()
            {
                sw = Stopwatch.StartNew();
            }
            public float read()
            {
                return sw.ElapsedMilliseconds;
            }
            public void pause()
            {
                sw.Stop();
            }
            public void start()
            {
                sw.Start();
            }
        }

    }
}
