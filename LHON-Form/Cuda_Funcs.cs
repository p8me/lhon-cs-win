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
        volatile GPGPU gpu;
        bool recompile_cuda = true;

        public bool init_gpu()
        {

            try { gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId); }
            catch { return true; }

            Debug.WriteLine(gpu);

            int deviceCount = CudafyHost.GetDeviceCount(CudafyModes.Target);
            if (deviceCount == 0) return true;

            string gpu_name = gpu.GetDeviceProperties(false).Name;

            //if (gpu is CudaGPU && gpu.GetDeviceProperties().Capability < new Version(1, 2))
            //{
            //    Debug.WriteLine("Compute capability 1.2 or higher required for atomics.");
            //    lbl_gpu_stat.Text = gpu_name + " not supported.";
            //    return true;
            //}

            CudafyModule km = CudafyModule.TryDeserialize();
            if (recompile_cuda && (km == null || !km.TryVerifyChecksums()))
            {
                km = CudafyTranslator.Cudafy();
                km.TrySerialize();
            }
            gpu.LoadModule(km);

            

            return false;
        }

        // ==================================================================
        //                          Preprocess
        // ==================================================================

        [Cudafy]
        public static void gpu_proprocess_1(GThread thread)
        {

        }
        [CudafyDummy]
        public static void cuda_update_live_neurs(int im_size,
            int n_neurs, float[,] tox, float[,] rate, float[,] detox, bool[] live_neur, int[] num_live_neur,
            float[] tox_touch_neur, float[] neur_tol, int[,,] axons_bound_touch_pix, int[] axons_bound_touch_npix,
            ushort[,] axons_inside_pix, int[] axons_inside_pix_idx, uint[,] locked_pix, int[] death_itr, int itr){ }
        
        [CudafyDummy]
        public static void cuda_calc_diff(int im_size,
            float[,] tox, float[,] rate, uint[,] locked_pix, float[,] diff)
        { }
        [CudafyDummy]
        public static void cuda_calc_tox(int im_size,
            float[,] tox, float[,] rate, float[,] detox, uint[,] locked_pix, float[,] diff){ }



        [Cudafy]
        // ======================================================================
        // Updates: tox_touch_neur, rate, detox, locked_pix and live_neur
        public static void gpu_update_live_neurs(GThread thread,
            int n_neurs, float[,] tox, float[,] rate, float[,] detox, bool[] live_neur, int[] num_live_neur, float[] tox_touch_neur, float[] neur_tol, int[,,] axons_bound_touch_pix, int[] axons_bound_touch_npix,
            ushort[,] axons_inside_pix, int[] axons_inside_pix_idx, uint[,] locked_pix, int[] death_itr, int itr)
        {
            int t_id = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            int stride = thread.blockDim.x * thread.gridDim.x;

            int t;

            for (int n = 0; n < n_neurs; n++)
                if (live_neur[n])
                {
                    t = t_id;
                    while (t < axons_bound_touch_npix[n])
                    {
                        int x = axons_bound_touch_pix[n, t, 0];
                        int y = axons_bound_touch_pix[n, t, 1];
                        if (locked_pix[x, y] == 0)
                            thread.atomicAdd(ref tox_touch_neur[n], tox[axons_bound_touch_pix[n, t, 0], axons_bound_touch_pix[n, t, 1]]);
                        t += stride;
                    }
                }
            thread.SyncThreads();

            for (int n = 0; n < n_neurs; n++)
            {
                if (tox_touch_neur[n] > neur_tol[n])
                {
                    // kill neuron/axon
                    t = t_id + axons_inside_pix_idx[n];
                    while (t < axons_inside_pix_idx[n + 1])
                    {
                        locked_pix[axons_inside_pix[t, 0], axons_inside_pix[t, 1]] -= 1;
                        detox[axons_inside_pix[t, 0], axons_inside_pix[t, 1]] = 0;
                        t += stride;
                    }
                    if (t_id == 0)
                    {
                        live_neur[n] = false;
                        num_live_neur[0]--;
                        death_itr[n] = itr;
                    }
                }
            }
        }
        
        [Cudafy]
        // ======================================================================
        public static void gpu_calc_neur_state(GThread thread)
        {

        }


        // ======================================================================
        //    Secondary Functions (for recording information or verification)
        // ======================================================================

        [Cudafy]
        public static void gpu_areal_progress(GThread thread, float[,] tox, uint[,] locked_pix, float[] progress, float lim)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y;

            if (locked_pix[x, y] == 0)
            {
                thread.atomicAdd(ref progress[0], 1F);
                if (tox[x, y] > lim / 2) thread.atomicAdd(ref progress[1], 1F);
                if (tox[x, y] > lim) thread.atomicAdd(ref progress[2], 1F);
            }

        }

        [Cudafy]
        public static void gpu_progress_image_1(GThread thread, float[,] tox, uint[,] locked_pix, float[,] progression_image_sum_float, uint[,] progress_image_num_averaged_pix, double ratio)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y;

            int X_ = (int)Math.Floor((double)x * ratio);
            int Y_ = (int)Math.Floor((double)y * ratio);

            if (locked_pix[x, y] == 0)
            {
                thread.atomicAdd(ref progression_image_sum_float[X_, Y_], tox[x, y]);
                thread.atomicAdd(ref progress_image_num_averaged_pix[X_, Y_], 1);
            }
        }

        [Cudafy]
        public static void gpu_progress_image_2(GThread thread, float[,] tox, uint[,] locked_pix, float[,] progression_image_sum_float, uint[,] progress_image_num_averaged_pix, byte[,] progression_image, ushort prog_im_siz)
        {
            int x = thread.blockIdx.x * thread.blockDim.x;
            int y = thread.blockIdx.y * thread.blockDim.y;

            if (progression_image_sum_float[x, y] > 0)
                progression_image[x, y] = (byte)(progression_image_sum_float[x, y] / (float)progress_image_num_averaged_pix[x, y] * 255);
        }

        [Cudafy]
        unsafe public static void gpu_fill_bmp(GThread thread, float[,] tox, byte[,,] bmp, bool[] show_opts, UInt16[,] touch)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y;

            fixed (byte* pix_addr = &bmp[y, x, 0])
                update_bmp_pix(tox[x, y], pix_addr, show_opts, touch[x, y]);
        }

        [Cudafy]
        public static void gpu_sum_tox(GThread thread, float[,] tox, float[] sum)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y;

            thread.atomicAdd(ref sum[0], tox[x, y]);
        }
    }
}


/*
 
    [Cudafy]
        public static void gpu_check_neur_overlap(GThread thread, float[,] tox, int x0, int y0, int xc, int yc, int rc_clear2, int[] overlaps)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x + x0;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y + y0;

            float dx = x - xc;
            float dy = y - yc;
            if (rc_clear2 - (dx * dx + dy * dy) > 0)
                if (tox[x, y] > 0)
                    thread.atomicAdd(ref overlaps[0], 1);
        }


    [Cudafy]
        public static void gpu_check_neur_overlap_2(GThread thread, float[,] tox, int x0, int y0, int xc, int yc, int rc_clear2, int[] overlaps)
        {
            int x = thread.blockIdx.x * thread.blockDim.x + thread.threadIdx.x + x0;
            int y = thread.blockIdx.y * thread.blockDim.y + thread.threadIdx.y + y0;

            if (overlaps[0] == 0)
            {
                float dx = x - xc;
                float dy = y - yc;
                if (rc_clear2 - (dx * dx + dy * dy) > 0)
                    tox[x, y] = 1;
            }
        }


 */
