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
        const string ProjectOutputDir = @"..\..\Project_Output\";
        
        // Used as measure of comparison
        const float real_model_nerve_r = 750; // um
        const int real_model_num_axons = 1200000;
        
        int threads_per_block_1D = 8;
        
        int nerve_clear = 4; // clearance of nerve from image borders in unit length

        int stop_iteration = 0;

        // ======================================================

        private BackgroundWorker alg_worker = new BackgroundWorker(), new_model_worker = new BackgroundWorker();
        AviManager aviManager;
        string avi_file;
        VideoStream aviStream;

        int first_axon_idx = 0;
        
        Bitmap bmp;
        ushort im_size;
        IntPtr bmp_scan0;
        byte[,,] bmp_bytes;

        bool[] show_opts = new bool[2],
            show_opts_dev = new bool[2];
        
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

        float[] sum_tox_dev, progress_dev;

        uint iteration = 0;
        
        float area_res_factor = 1;

        float[] init_insult = new float[2] { 0, 0 };
        
        dim3 blocks_per_grid, threads_per_block;
        byte[,,] bmp_dev;
        
        enum sim_stat_enum { None, Running, Paused, Successful, Failed };
        sim_stat_enum sim_stat = sim_stat_enum.None;
        
        private class axon_lbl_class
        {
            public string lbl;
            public float x;
            public float y;
        }

        axon_lbl_class[] axon_lbl;

        public class Model
        {
            public float nerve_r, vessel_rat, min_r, max_r,
                min_r_abs,
                max_r_abs,
                clearance;
            public float circ_gen_ratio;

            public int n_axons;
            public List<float[]> axon_coor;
        }

        public class Setts
        {
            public float resolution;

            public float rate_live;
            public float rate_dead;
            public float rate_bound;
            public float rate_extra;

            public float tox_prod;
            public float detox_intra;
            public float detox_extra;

            public float death_tox_lim;
        }

        Model mdl = new Model();

        Setts setts = new Setts();

        // ============= Main loop vars =================

        float progress_step, next_areal_progress_snapshot, next_chron_progress_snapshot;
        tic_toc tt_sim = new tic_toc();
        float sim_time = 0;
        uint last_itr;
        float last_areal_prog;

        // =============== Profiling Class =============

        class profile_class
        {
            const int max = 100;
            double[] T = new double[max];
            int[] num_occur = new int[max];
            Stopwatch sw = Stopwatch.StartNew();
            Stopwatch sw_tot = Stopwatch.StartNew();
            public void time(int idx) // Pass 0 start of main program to start tot_time
            {
                if (idx > 0)
                {
                    T[idx] += sw.Elapsed.TotalMilliseconds;
                    num_occur[idx]++;
                }
                else if (idx == 0)
                {
                    Array.Clear(T, 0, max);
                    Array.Clear(num_occur, 0, max);
                    sw_tot = Stopwatch.StartNew();
                }
                sw = Stopwatch.StartNew(); // Pass negative to reset only sw
            }
            public void report() // This will stop and conclude tot_time
            {
                sw_tot.Stop();
                sw.Stop();
                double tot_time = sw_tot.Elapsed.TotalMilliseconds;
                Debug.WriteLine("Total: " + (tot_time / 1000).ToString("0.000") + "s");
                for (int k = 0; k < T.Length; k++)
                    if (T[k] > 0)
                        Debug.WriteLine("{0}:\t{1}%\t{2}ms\t{3}K >> {4}ms", k, (T[k] / tot_time * 100).ToString("00.0"), T[k].ToString("000000"), (num_occur[k]/1000).ToString("0000"), (T[k]/num_occur[k]).ToString("000.000"));
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
            float dx = (float)x - xc;
            float dy = (float)y - yc;
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
