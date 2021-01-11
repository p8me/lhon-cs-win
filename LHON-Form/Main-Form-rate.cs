//
//  Main-Form-rate.cs
//  LHON-Form
//
//  Created by Pooya Merat in 2016.
//

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

// Speed inside bundle: Fastest, outside bundles slower, boundaries slowest, 

namespace LHON_Form
{
    public partial class Main_Form : Form
    {
        int gui_update_itr = 20;

        string file_base_addr = @"Models\";

        #region Variables
        private BackgroundWorker alg_worker = new BackgroundWorker(),
            gui_worker = new BackgroundWorker();
        AviManager aviManager;

        int max_set_size_bound, max_set_size_bound_touch, max_set_size_inside;

        bool running, gpu_available;

        Bitmap bmp;
        int width, height;

        float[,] tox, init_tox, tox_dev;
        uint[,] barr, barr_init, trapped_pix, barr_dev, trapped_pix_dev;
        int[] rad_range;

        float inside_rate, outside_rate, boundary_rate;

        float[,] rate, rate_dev;

        int iteration;

        int n_neurs;
        List<int[]> neurs_coor;

        int[,,] neurs_inside_pix, neurs_bound_pix, neurs_bound_touch_pix,
            neurs_inside_pix_dev, neurs_bound_pix_dev, neurs_bound_touch_pix_dev;
        int[] neurs_inside_npix, neurs_bound_npix, neurs_bound_touch_npix,
            neurs_inside_npix_dev, neurs_bound_npix_dev, neurs_bound_touch_npix_dev;
        bool[] live_neur,
            live_neur_dev;
        float[] neur_tol, tox_touch_neur, tox_touch_neur_last,
            neur_tol_dev, tox_touch_neur_dev;
        neur_lbl_class[] neur_lbl;

        List<Bitmap> bmp_list;

        base_inf_class bas = new base_inf_class(); // for loading and saving

        private class neur_lbl_class
        {
            public string lbl;
            public int x;
            public int y;
        }

        public class base_inf_class
        {
            public int im_w, im_h;
            public List<int[]> n_cor;
        }

        #endregion

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

        public Main_Form()
        {
            width = 1024; // width of total image in pixelss
            height = width;
            inside_rate = 0.1f; // the higher the faster the diffusion
            outside_rate = 0.5f;
            boundary_rate = 0.5f;
            rad_range = new int[2] { 4, 25 }; // range of neuron radii in pixels

            #region Basic Init

            float max_r = rad_range.Max();

            max_set_size_inside = (int)(max_r * max_r * 3.14);
            max_set_size_bound = (int)(2 * max_r * 3.14);
            max_set_size_bound_touch = (int)(3 * max_r * 3.14);

            InitializeComponent();
            this.CenterToScreen();

            DoubleBuffered = true;

            running = false;
            iteration = 0;

            bmp_list = new List<Bitmap>();
            bmp = new Bitmap(width, height);

            btn_save_rec.Visible = false;

            #endregion
        }

        public bool init_gpu()
        {
            gpu_prof.time(0);
            lbl_gpu_stat.Text = "No GPU found.";

            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu_prof.time(1);
            int deviceCount = CudafyHost.GetDeviceCount(CudafyModes.Target);
            gpu_prof.time(2);

            if (deviceCount == 0)
                return true;

            string gpu_name = gpu.GetDeviceProperties(false).Name;

            gpu_prof.time(3);

            if (gpu is CudaGPU && gpu.GetDeviceProperties().Capability < new Version(1, 2))
            {
                Debug.WriteLine("Compute capability 1.2 or higher required for atomics.");
                lbl_gpu_stat.Text = gpu_name + " not supported.";
                return true;
            }
            gpu_prof.time(4);

            CudafyModule km = CudafyModule.TryDeserialize();
            gpu_prof.time(5);

            if (km == null || !km.TryVerifyChecksums())
            {
                km = CudafyTranslator.Cudafy();
                km.TrySerialize();
            }
            gpu.LoadModule(km);

            gpu_prof.time(6);
            gpu_prof.report();

            lbl_gpu_stat.Text = gpu_name;
            return false;
        }

        private bool running_on_gpu()
        {
            return gpu_available && !chk_force_cpu.Checked;
        }

        private void Main_Form_Load(object sender, EventArgs e)
        {
            gpu_available = false;
            try { gpu_available = !init_gpu(); }
            catch { }// no GPU or not supported
            if (!gpu_available) chk_force_cpu.Visible = false;

            alg_worker.DoWork += new DoWorkEventHandler(run_alg_worker);
            alg_worker.WorkerSupportsCancellation = true;

            if (reload_model_file(Properties.Settings.Default.last_path))
                new_model();
        }

        private void new_model()
        {
            neurs_coor = new List<int[]>();

            // Generate random coordinates and raduis
            Random random = new Random();
            int neur_place_tries = width * width / 5;

            for (int i = 0; i < neur_place_tries; i++)
            {
                int rc = random.Next(rad_range[0], rad_range[1]);
                int xc = random.Next(rc, width - rc);
                int yc = random.Next(rc, height - rc);

                bool ignored = false;

                if (neurs_coor.Count > 0)
                {
                    for (int j = 0; j < neurs_coor.Count; j++)
                    {
                        int dx = neurs_coor[j][0] - xc;
                        int dy = neurs_coor[j][1] - yc;
                        int rsum = neurs_coor[j][2] + rc;
                        if ((dx * dx + dy * dy) - rsum * rsum < 25)
                        {
                            ignored = true;
                            break; // ignore the new coordinates
                        }
                    }
                }
                if (ignored) continue;
                neurs_coor.Add(new int[3] { xc, yc, rc });
            }
            preprocess_model();
        }

        // Assigns tox, barr, etc according to width, height and neurs_coor
        private void preprocess_model()
        {
            Stopwatch s = Stopwatch.StartNew();
            // inits
            tox = new float[width, height];
            rate = new float[width, height];
            init_tox = new float[width, height];
            barr = new uint[width, height];
            barr_init = new uint[width, height];
            trapped_pix = new uint[width, height];

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    rate[x, y] = outside_rate;

            // Enclosing Barrier
            for (int x = 0; x < width; x++)
            {
                barr[x, height - 1] = 1;
                barr[x, 0] = 1;
            }

            for (int y = 0; y < height; y++)
            {
                barr[0, y] = 1;
                barr[width - 1, y] = 1;
            }

            n_neurs = neurs_coor.Count;

            neurs_bound_pix = new int[n_neurs, max_set_size_bound, 2];
            neurs_bound_npix = new int[n_neurs];
            neurs_inside_pix = new int[n_neurs, 10 * max_set_size_inside, 2];
            neurs_inside_npix = new int[n_neurs];
            neurs_bound_touch_pix = new int[n_neurs, 10 * max_set_size_bound_touch, 2];
            neurs_bound_touch_npix = new int[n_neurs];

            live_neur = new bool[n_neurs];
            neur_tol = new float[n_neurs];
            tox_touch_neur = new float[n_neurs];
            tox_touch_neur_last = new float[n_neurs];
            neur_lbl = new neur_lbl_class[n_neurs];

            for (int i = 0; i < n_neurs; i++)
            {
                int xc = neurs_coor[i][0];
                int yc = neurs_coor[i][1];
                int rc = neurs_coor[i][2];

                live_neur[i] = true;

                neur_tol[i] = rc * rc / 10;

                neur_lbl[i] = new neur_lbl_class { lbl = "0", x = xc, y = yc };

                for (int y = Max(yc - rc, 0); y <= yc + rc && y < height; y++)
                {
                    for (int x = Max(xc - rc, 0); x <= xc + rc && x < width; x++)
                    {
                        int wc = within_circle2(x, y, xc, yc, rc);
                        bool inside = wc > 0;
                        bool on_bound = Math.Abs(wc) <= rc && barr[x, y] == 0;

                        if (inside && !on_bound)
                        {
                            neurs_inside_pix[i, neurs_inside_npix[i], 0] = x;
                            neurs_inside_pix[i, neurs_inside_npix[i]++, 1] = y;
                            trapped_pix[x, y] = 1;
                            tox[x, y] = 1;
                            rate[x, y] = inside_rate;
                        }

                        if (on_bound) // if near cell boundary and not already a barrier
                        {
                            neurs_bound_pix[i, neurs_bound_npix[i], 0] = x;
                            neurs_bound_pix[i, neurs_bound_npix[i]++, 1] = y;
                            barr[x, y] = 1;
                            rate[x, y] = boundary_rate;
                        }
                    }
                }
                for (int m = 0; m < neurs_bound_npix[i]; m++)
                {
                    int x = neurs_bound_pix[i, m, 0];
                    int y = neurs_bound_pix[i, m, 1];

                    int[,] arr = new int[4, 2] { { x + 1, y }, { x - 1, y }, { x, y - 1 }, { x, y - 1 } };

                    for (int k = 0; k < 4; k++)
                    {
                        if (within_circle2(arr[k, 0], arr[k, 1], xc, yc, rc) < 0)
                        {
                            bool on_bound_again = false;
                            for (int kk = 0; kk < neurs_bound_npix[i]; kk++)
                            {
                                if (neurs_bound_pix[i, kk, 0] == arr[k, 0] && neurs_bound_pix[i, kk, 1] == arr[k, 1])
                                {
                                    on_bound_again = true;
                                    break;
                                }
                            }
                            if (!on_bound_again)
                            {
                                neurs_bound_touch_pix[i, neurs_bound_touch_npix[i], 0] = arr[k, 0];
                                neurs_bound_touch_pix[i, neurs_bound_touch_npix[i]++, 1] = arr[k, 1];
                            }
                        }
                    }
                }
            }

            // Keep back up of inital state
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    init_tox[x, y] = tox[x, y];
                    barr_init[x, y] = barr[x, y];
                }
            init_state();

            Debug.WriteLine("Number of Neurons: {0}", n_neurs);
            Debug.WriteLine("Preprocess: " + s.ElapsedMilliseconds.ToString("0") + " ms");
            Debug.WriteLine(neurs_inside_npix.Max());
            Debug.WriteLine(neurs_bound_npix.Max());
            Debug.WriteLine(neurs_bound_touch_npix.Max());
        }

        #region cuda functions

        [Cudafy]
        // Updates: tox_touch_neur, barr, trapped_pix and live_neur
        public static void gpu_update_live_neurs(GThread thread, int n_neurs, float[,] tox, bool[] live_neur, float[] tox_touch_neur, float[] neur_tol, int[,,] neurs_bound_touch_pix, int[] neurs_bound_touch_npix,
            uint[,] barr, int[,,] neurs_bound_pix, int[] neurs_bound_npix, int[,,] neurs_inside_pix, int[] neurs_inside_npix, uint[,] trapped_pix)
        {
            int t = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int stride = thread.blockDim.x * thread.gridDim.x;

            int task_id = t;
            while (task_id < n_neurs)
            {
                tox_touch_neur[task_id] = 0;
                task_id += stride;
            }

            thread.SyncThreads();

            for (int n = 0; n < n_neurs; n++)
                if (live_neur[n])
                {
                    task_id = t;
                    while (task_id < neurs_bound_touch_npix[n])
                    {
                        thread.atomicAdd(ref tox_touch_neur[n], tox[neurs_bound_touch_pix[n, task_id, 0], neurs_bound_touch_pix[n, task_id, 1]]);
                        task_id += stride;
                    }
                }
            thread.SyncThreads();

            for (int n = 0; n < n_neurs; n++)
            {
                if (tox_touch_neur[n] > neur_tol[n])
                {
                    task_id = t;
                    while (task_id < neurs_bound_npix[n])
                    {
                        thread.atomicDec(ref barr[neurs_bound_pix[n, task_id, 0], neurs_bound_pix[n, task_id, 1]], 0);
                        task_id += stride;
                    }
                    task_id = t;
                    while (task_id < neurs_inside_npix[n])
                    {
                        thread.atomicDec(ref trapped_pix[neurs_inside_pix[n, task_id, 0], neurs_inside_pix[n, task_id, 1]], 0);
                        task_id += stride;
                    }
                    if (t == 0) live_neur[n] = false;
                }
            }
        }

        [Cudafy]
        // Updates: diff
        public static void gpu_calc_diff(GThread thread, float[,] tox, uint[,] barr, uint[,] trapped_pix, float[,] diff)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y;

            if (tox[x, y] > 0 && trapped_pix[x, y] == 0 && barr[x, y] == 0)
            {
                uint num = 4 - (barr[x + 1, y] + barr[x - 1, y] + barr[x, y + 1] + barr[x, y - 1]);

                if (num > 0)
                {
                    float temp = tox[x, y] / num;
                    if (barr[x + 1, y] == 0) thread.atomicAdd(ref diff[x + 1, y], temp);
                    if (barr[x - 1, y] == 0) thread.atomicAdd(ref diff[x - 1, y], temp);
                    if (barr[x, y + 1] == 0) thread.atomicAdd(ref diff[x, y + 1], temp);
                    if (barr[x, y - 1] == 0) thread.atomicAdd(ref diff[x, y - 1], temp);
                }
            }
        }

        [Cudafy]
        // Updates: tox
        public static void gpu_calc_tox(GThread thread, float[,] tox, uint[,] barr, uint[,] trapped_pix, float[,] diff, float[,] rate)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y;

            if (trapped_pix[x, y] == 0 && barr[x, y] == 0)
                tox[x, y] = (1 - rate[x, y]) * tox[x, y] + rate[x, y] * diff[x, y];
            //thread.atomicAdd(ref tox[x, y], -rate * tox[x, y] + rate * diff[x, y]);
        }

        [Cudafy]
        public static void gpu_fill_bmp(GThread thread, float[,] tox, byte[,,] bmp, uint[,] barr)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y;

            int r = 255, g = 255, b = 255, v = (int)(tox[x, y] * 255);

            if (barr[x, y] == 1)
            {
                r = 255; g = 0; b = 0;
            }
            else
            {
                if (v < 64) { r = 0; g = 4 * v; }
                else if (v < 128) { r = 0; b = 255 + 4 * (64 - v); }
                else if (v < 192) { r = 4 * (v - 128); b = 0; }
                else { g = 255 + 4 * (192 - v); b = 0; }
            }

            bmp[y, x, 0] = (byte)b;
            bmp[y, x, 1] = (byte)g;
            bmp[y, x, 2] = (byte)r;
            bmp[y, x, 3] = 255;
        }

        [Cudafy]
        public static void gpu_sum_tox(GThread thread, float[,] tox, float[] sum)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y;

            thread.atomicAdd(ref sum[0], tox[x, y]);
        }

        #endregion

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);

        private void run_alg_worker(object sender, DoWorkEventArgs e)
        {
            if (running_on_gpu())
                Run_Alg_GPU();
            else
                Run_Alg_CPU();
        }

        private void load_gpu_vars()
        {
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.FreeAll();

            tox_dev = gpu.Allocate(tox); gpu.CopyToDevice(tox, tox_dev);
            rate_dev = gpu.Allocate(rate); gpu.CopyToDevice(rate, rate_dev);
            barr_dev = gpu.Allocate(barr); gpu.CopyToDevice(barr, barr_dev);
            trapped_pix_dev = gpu.Allocate(trapped_pix); gpu.CopyToDevice(trapped_pix, trapped_pix_dev);
            tox_touch_neur_dev = gpu.Allocate(tox_touch_neur); gpu.CopyToDevice(tox_touch_neur, tox_touch_neur_dev);
            live_neur_dev = gpu.Allocate(live_neur); gpu.CopyToDevice(live_neur, live_neur_dev);

            neurs_inside_pix_dev = gpu.Allocate(neurs_inside_pix); gpu.CopyToDevice(neurs_inside_pix, neurs_inside_pix_dev); // constant
            neurs_inside_npix_dev = gpu.Allocate(neurs_inside_npix); gpu.CopyToDevice(neurs_inside_npix, neurs_inside_npix_dev); // constant
            neurs_bound_pix_dev = gpu.Allocate(neurs_bound_pix); gpu.CopyToDevice(neurs_bound_pix, neurs_bound_pix_dev); // constant
            neurs_bound_npix_dev = gpu.Allocate(neurs_bound_npix); gpu.CopyToDevice(neurs_bound_npix, neurs_bound_npix_dev); // constant
            neurs_bound_touch_pix_dev = gpu.Allocate(neurs_bound_touch_pix); gpu.CopyToDevice(neurs_bound_touch_pix, neurs_bound_touch_pix_dev); // constant
            neurs_bound_touch_npix_dev = gpu.Allocate(neurs_bound_touch_npix); gpu.CopyToDevice(neurs_bound_touch_npix, neurs_bound_touch_npix_dev); // constant
            neur_tol_dev = gpu.Allocate(neur_tol); gpu.CopyToDevice(neur_tol, neur_tol_dev); // constant
        }

        private void load_cpu_vars_from_gpu()
        {
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

            gpu.CopyFromDevice(tox_dev, tox);
            gpu.CopyFromDevice(barr_dev, barr);
            gpu.CopyFromDevice(trapped_pix_dev, trapped_pix);
            gpu.CopyFromDevice(tox_touch_neur_dev, tox_touch_neur);
            gpu.CopyFromDevice(live_neur_dev, live_neur);
        }

        private void chk_force_cpu_CheckedChanged(object sender, EventArgs e)
        {
            if (running) btn_start_Click(null, null);

            if (chk_force_cpu.Checked)
                load_cpu_vars_from_gpu();

            else
                load_gpu_vars();
        }

        unsafe private void Run_Alg_GPU()
        {
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

            Debug.WriteLine(gpu.GetDeviceProperties(false).Name);

            Rectangle rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, bmp.PixelFormat);
            IntPtr scan0 = bmpData.Scan0;
            bmp.UnlockBits(bmpData);
            byte[,,] bmp_temp = new byte[width, height, 4];

            float sum_tox;

            byte[,,] bmp_dev = gpu.Allocate<byte>(width, height, 4);
            float[,] diff_dev = gpu.Allocate<float>(width, height);
            float[] sum_tox_dev = gpu.Allocate<float>(1);

            const int bl_ = 8;

            dim3 block_s_r = new dim3((width) / bl_, (height) / bl_);
            dim3 thread_s_r = new dim3(bl_, bl_);

            float[,] diff = new float[width, height];

            alg_prof.time(0);

            while (true)
            {
                iteration++;

                bool update_gui = iteration % gui_update_itr == 0;

                alg_prof.time(-1);

                if ((iteration % 20) == 0)
                {
                    gpu.Launch(1, 32).gpu_update_live_neurs(n_neurs, tox_dev, live_neur_dev, tox_touch_neur_dev, neur_tol_dev, neurs_bound_touch_pix_dev, neurs_bound_touch_npix_dev,
                       barr_dev, neurs_bound_pix_dev, neurs_bound_npix_dev, neurs_inside_pix_dev, neurs_inside_npix_dev, trapped_pix_dev);
                    gpu.Synchronize();
                    alg_prof.time(1);
                }
                gpu.Set(diff_dev);
                gpu.Synchronize();
                alg_prof.time(2);
                gpu.Launch(block_s_r, thread_s_r).gpu_calc_diff(tox_dev, barr_dev, trapped_pix_dev, diff_dev);
                gpu.Synchronize();
                alg_prof.time(3);
                gpu.Launch(block_s_r, thread_s_r).gpu_calc_tox(tox_dev, barr_dev, trapped_pix_dev, diff_dev, rate_dev);
                gpu.Synchronize();
                alg_prof.time(4);

                if (update_gui)
                {
                    gpu.CopyFromDevice(tox_touch_neur_dev, tox_touch_neur);
                    gpu.CopyFromDevice(live_neur_dev, live_neur);
                    gpu.Launch(block_s_r, thread_s_r).gpu_fill_bmp(tox_dev, bmp_dev, barr_dev);
                    gpu.CopyFromDevice(bmp_dev, bmp_temp);
                    gpu.Set(sum_tox_dev);
                    gpu.Launch(block_s_r, thread_s_r).gpu_sum_tox(tox_dev, sum_tox_dev);
                    gpu.CopyFromDevice(sum_tox_dev, out sum_tox);
                    gpu.Synchronize();

                    alg_prof.time(5);

                    fixed (byte* dat = &bmp_temp[0, 0, 0])
                        CopyMemory(scan0, (IntPtr)dat, (uint)bmp_temp.Length);

                    picB.Image = bmp;
                    rec_frame();

                    //Bitmap bmp_temp2 = new Bitmap(picB.ClientSize.Width, picB.ClientSize.Height);
                    //picB.DrawToBitmap(bmp_temp2, picB.ClientRectangle);
                    //if (chkb_rec.Checked) aviStream.AddFrame((Bitmap)bmp_temp2.Clone());

                    // Labels
                    update_itr_lbl(iteration.ToString());
                    update_tox_lbl(sum_tox.ToString("000.0E-0"));
                    update_neurs_lbls();

                    update_itr_s_lbl((gui_update_itr / toc() * 1000).ToString("0.0"));

                    alg_prof.time(6);
                }

                if (!running) break;
                //if (iteration > 2000) btn_start_Click(null, null);
            }

            alg_prof.report();
        }

        void rec_frame()
        {
            if (InvokeRequired)
                Invoke(new Action(() => rec_frame()));
            else
                if (running && chkb_rec.Checked) aviStream.AddFrame((Bitmap)bmp.Clone());
            //if (running && chkb_rec.Checked)
            //{
            //    Bitmap bmp_temp = new Bitmap(picB.ClientSize.Width, picB.ClientSize.Height);
            //    picB.DrawToBitmap(bmp_temp, picB.ClientRectangle);
            //    bmp_list.Add((Bitmap)bmp_temp.Clone());
            //}
        }

        unsafe private void Run_Alg_CPU()
        {
            gui_worker.DoWork += new DoWorkEventHandler(update_GUI_CPU);
            gui_worker.WorkerSupportsCancellation = true;

            float[,] diff = new float[width, height];

            alg_prof.time(0);

            while (true)
            {
                iteration++;

                bool update_gui = iteration % gui_update_itr == 0;

                alg_prof.time(-1);
                calc_tox_neur();

                for (int i = 0; i < n_neurs; i++)
                    if (tox_touch_neur[i] > neur_tol[i])
                        kill_neur(i);
                alg_prof.time(1);

                // Calculate new toxic concentration
                Array.Clear(diff, 0, diff.Length);

                for (UInt16 y = 1; y < height - 1; y++)
                {
                    for (UInt16 x = 1; x < width - 1; x++)
                    {
                        if (tox[x, y] == 0 || trapped_pix[x, y] == 1 || barr[x, y] == 1) continue;
                        uint num = 4 - (barr[x + 1, y] + barr[x - 1, y] + barr[x, y + 1] + barr[x, y - 1]);

                        if (num == 0) continue;
                        float temp = tox[x, y] / num;
                        if (barr[x + 1, y] == 0) diff[x + 1, y] += temp;
                        if (barr[x - 1, y] == 0) diff[x - 1, y] += temp;
                        if (barr[x, y + 1] == 0) diff[x, y + 1] += temp;
                        if (barr[x, y - 1] == 0) diff[x, y - 1] += temp;
                    }
                }
                alg_prof.time(2);
                
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                    {
                        if (trapped_pix[x, y] == 1 || barr[x, y] == 1) continue;
                        tox[x, y] = (1 - rate[x,y]) * tox[x, y] + rate[x,y] * diff[x, y];
                    }
                alg_prof.time(3);

                if (update_gui && !gui_worker.IsBusy)
                {
                    update_itr_lbl(iteration.ToString());
                    double tox_sum = 0;
                    for (int y = 0; y < height; y++)
                        for (int x = 0; x < width; x++)
                            tox_sum += tox[x, y];
                    update_tox_lbl(tox_sum.ToString("000.0E-0"));
                    update_neurs_lbls();
                    update_itr_s_lbl((gui_update_itr / toc() * 1000).ToString("0.0"));
                    gui_worker.RunWorkerAsync();
                }

                alg_prof.time(4);

                if (!running) break;
                //if (iteration > 2000) btn_start_Click(null, null);
            }
            alg_prof.report();
        }

        //==================== Alg Helpers

        private void init_state()
        {
            running = false;
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    tox[x, y] = init_tox[x, y];
                    barr[x, y] = barr_init[x, y];
                }
            bmp_list.Clear();
            iteration = 0;

            for (int i = 0; i < n_neurs; i++)
            {
                live_neur[i] = true;
                tox_touch_neur[i] = 0;
            }

            int first_neur_idx = neurs_coor.Count / 3;
            kill_neur(first_neur_idx);
            update_GUI_CPU(null, null);
            neur_lbl[first_neur_idx].lbl = "+";

            if (running_on_gpu()) load_gpu_vars();
        }

        // calculates the toxic around a neuron
        private void calc_tox_neur()
        {
            for (int n = 0; n < n_neurs; n++)
            {
                if (live_neur[n])
                {
                    tox_touch_neur[n] = 0;
                    for (int i = 0; i < neurs_bound_touch_npix[n]; i++)
                        tox_touch_neur[n] += tox[neurs_bound_touch_pix[n, i, 0], neurs_bound_touch_pix[n, i, 1]];
                }
            }
        }

        private void kill_neur(int idx)
        {
            if (n_neurs <= idx) return;
            for (int i = 0; i < neurs_bound_npix[idx]; i++)
                barr[neurs_bound_pix[idx, i, 0], neurs_bound_pix[idx, i, 1]] = 0;
            for (int i = 0; i < neurs_inside_npix[idx]; i++)
                trapped_pix[neurs_inside_pix[idx, i, 0], neurs_inside_pix[idx, i, 1]] = 0;

            neur_lbl[idx].lbl = "";
            live_neur[idx] = false;
        }

        // Start / Stop Button
        private void btn_start_Click(object sender, EventArgs e)
        {
            if (InvokeRequired)
                Invoke(new Action(() => this.btn_start_Click(null, null)));
            else
            {
                if (!running)
                {
                    running = true;
                    alg_worker.RunWorkerAsync();
                    btn_start.Text = "&Stop (S)";
                }
                else
                {
                    running = false;
                    btn_start.Text = "&Start (S)";
                }
            }
        }

        private void btn_redraw_Click(object sender, EventArgs e)
        {
            if (running) return;
            new_model();
            iteration = 0;
        }

        string avi_file;
        VideoStream aviStream;

        private void chkb_rec_CheckedChanged(object sender, EventArgs e)
        {
            if (chkb_rec.Checked)
            {
                avi_file = file_base_addr + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + '(' + (width * height / 1e6).ToString("0.0") + "Mpix).avi";
                aviManager = new AviManager(avi_file, false);
                Avi.AVICOMPRESSOPTIONS options = new Avi.AVICOMPRESSOPTIONS();
                options.fccType = (uint)Avi.mmioStringToFOURCC("vids", 5);
                options.fccHandler = (uint)Avi.mmioStringToFOURCC("CVID", 5);
                //options.dwQuality = 1;
                aviStream = aviManager.AddVideoStream(options, 10, bmp);
            }
            else
            {
                aviManager.Close();
                Process.Start(avi_file);
            }
        }

        private void btn_save_model_Click(object sender, EventArgs e)
        {
            // Save all informaion to replicate the initial condition
            bas.n_cor = neurs_coor;
            bas.im_h = width;
            bas.im_w = height;
            var fil_name = file_base_addr + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + '(' + (width * height / 1e6).ToString("0.0") + "Mpix).mdat";
            Debug.WriteLine(fil_name);

            FileStream outFile = File.Create(fil_name);
            XmlSerializer formatter = new XmlSerializer(bas.GetType());
            formatter.Serialize(outFile, bas);
            Properties.Settings.Default.last_path = fil_name; // Save path in properties
            Properties.Settings.Default.Save();
        }

        private void btn_load_model_Click(object sender, EventArgs e)
        {
            var FD = new OpenFileDialog()
            {
                InitialDirectory = file_base_addr,
                Title = "Load Model",
                Filter = "Model Data files (*.mdat) | *.mdat",
                RestoreDirectory = true,
                AutoUpgradeEnabled = false
            };
            if (FD.ShowDialog() != DialogResult.OK) return;
            Properties.Settings.Default.last_path = FD.FileName; // Save path in properties
            Properties.Settings.Default.Save();
            reload_model_file(FD.FileName);
        }

        private bool reload_model_file(string path)
        {
            if (!File.Exists(path)) return true;

            XmlSerializer formatter = new XmlSerializer(bas.GetType());
            FileStream aFile = new FileStream(path, FileMode.Open);
            byte[] buffer = new byte[aFile.Length];
            aFile.Read(buffer, 0, (int)aFile.Length);
            MemoryStream stream = new MemoryStream(buffer);
            bas = (base_inf_class)formatter.Deserialize(stream);
            aFile.Close();

            neurs_coor = bas.n_cor;
            width = bas.im_h;
            height = bas.im_w;
            bmp = new Bitmap(width, height);

            preprocess_model();
            return false;
        }

        private void btn_save_rec_Click(object sender, EventArgs e)
        {
            if (bmp_list.Count == 0) return;
            aviManager = new AviManager(@"..\..\Test.avi", false);
            Avi.AVICOMPRESSOPTIONS options = new Avi.AVICOMPRESSOPTIONS();
            options.fccType = (uint)Avi.mmioStringToFOURCC("vids", 5);
            options.fccHandler = (uint)Avi.mmioStringToFOURCC("CVID", 5);
            //options.dwQuality = 1;

            VideoStream aviStream = aviManager.AddVideoStream(options, 10, bmp_list[0]);

            for (int i = 1; i < bmp_list.Count; i++)
            {
                aviStream.AddFrame(bmp_list[i]);
            }
            aviManager.Close();
            Process.Start(@"..\..\Test.avi");
        }

        private void btn_reset_Click(object sender, EventArgs e)
        {
            init_state();
        }

        private void update_GUI_CPU(object sender, DoWorkEventArgs e)
        {
            if (InvokeRequired)
                Invoke(new Action(() => update_GUI_CPU(null, null)));
            else
            {
                BitmapData dat = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
                int stride = dat.Stride;
                IntPtr Scan0 = dat.Scan0;

                unsafe
                {
                    byte* p = (byte*)(void*)Scan0;

                    int offset = stride - bmp.Width * 3;
                    for (int y = 0; y < bmp.Height; y++)
                    {
                        for (int x = 0; x < bmp.Width; x++)
                        {
                            GetColour((int)(tox[x, y] * 255), p);
                            p += 3;
                        }
                        p += offset;
                    }
                }

                bmp.UnlockBits(dat);
                picB.Image = bmp;
                rec_frame();
            }
        }

        //====================

        string GetRelativePath(string filespec, string folder)
        {
            Uri pathUri = new Uri(filespec);
            // Folders must end in a slash
            if (!folder.EndsWith(Path.DirectorySeparatorChar.ToString()))
            {
                folder += Path.DirectorySeparatorChar;
            }
            Uri folderUri = new Uri(folder);
            return Uri.UnescapeDataString(folderUri.MakeRelativeUri(pathUri).ToString().Replace('/', Path.DirectorySeparatorChar));
        }

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

        #region basic gui
        unsafe private void GetColour(int v, byte* p)
        {
            int r = 255, g = 255, b = 255;

            if (v < 64)
            {
                r = 0;
                g = 4 * v;
            }
            else if (v < 128)
            {
                r = 0;
                b = 255 + 4 * (64 - v);
            }
            else if (v < 192)
            {
                r = 4 * (v - 128);
                b = 0;
            }
            else
            {
                g = 255 + 4 * (192 - v);
                b = 0;
            }
            p[0] = (byte)b;
            p[1] = (byte)g;
            p[2] = (byte)r;
        }

        private void picB_Paint(object sender, PaintEventArgs e)
        {
            e.Graphics.InterpolationMode = InterpolationMode.NearestNeighbor;
            if (chk_neur_lvl.Checked)
            {
                int siz = Min(picB.Size.Width, picB.Size.Height);
                int off = (Max(picB.Size.Width, picB.Size.Height) - siz) / 2;
                int offx = 0, offy = 0;
                if (picB.Size.Width > picB.Size.Height)
                    offx = off;
                else
                    offy = off;

                float ratx = (float)siz / (float)width;
                float raty = (float)siz / (float)height;

                foreach (var nlbl in neur_lbl)
                {
                    SizeF textSize = e.Graphics.MeasureString(nlbl.lbl, this.Font);
                    e.Graphics.DrawString(nlbl.lbl, this.Font, Brushes.White, nlbl.x * ratx + offx - (textSize.Width / 2), nlbl.y * raty + offy - (textSize.Height / 2));
                }
            }
        }

        private void update_neurs_lbls()
        {
            if (chk_neur_lvl.Checked)
            {
                for (int i = 0; i < neurs_coor.Count; i++)
                    if (live_neur[i] && tox_touch_neur[i] > 1)
                    {
                        if (tox_touch_neur_last[i] == tox_touch_neur[i]) continue;
                        neur_lbl[i].lbl = (tox_touch_neur[i] / neur_tol[i] * 100).ToString("0");
                        tox_touch_neur_last[i] = tox_touch_neur[i];
                    }
                    else
                        neur_lbl[i].lbl = "";
            }
        }

        private void update_tox_lbl(string s)
        {
            if (InvokeRequired)
                Invoke(new Action(() => this.update_tox_lbl(s)));
            else
                lbl_tox.Text = s;
        }

        private void update_itr_lbl(string s)
        {
            if (!running) return;
            if (InvokeRequired)
                Invoke(new Action(() => this.update_itr_lbl(s)));
            else
                lbl_itr.Text = s;
        }

        private void update_itr_s_lbl(string s)
        {
            if (!running) return;
            if (InvokeRequired)
                Invoke(new Action(() => this.update_itr_s_lbl(s)));
            else
                lbl_itr_s.Text = s;
        }

        private void Main_Form_FormClosing(object sender, FormClosingEventArgs e)
        {
            running = false;
            alg_worker.CancelAsync();
        }

        #endregion

        #region basic math
        private int Max(int v1, int v2)
        {
            return (v1 > v2) ? v1 : v2;
        }

        private int Min(int v1, int v2)
        {
            return (v1 < v2) ? v1 : v2;
        }

        private int within_circle2(int x, int y, int xc, int yc, int rc)
        {
            int dx = x - xc;
            int dy = y - yc;
            return rc * rc - (dx * dx + dy * dy);
        }
        #endregion

    }
}
