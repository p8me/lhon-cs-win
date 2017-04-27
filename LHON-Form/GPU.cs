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

        const int max_resident_blocks = 16;
        const int max_resident_threads = 2048;
        const int warp_size = 32;

        public bool init_gpu()
        {
            try { gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId); }
            catch { return true; }

            int deviceCount = CudafyHost.GetDeviceCount(CudafyModes.Target);
            if (deviceCount == 0) return true;

            string gpu_name = gpu.GetDeviceProperties(false).Name;

            //if (gpu is CudaGPU && gpu.GetDeviceProperties().Capability < new Version(1, 2))
            //{
            //    Debug.WriteLine("Compute capability 1.2 or higher required for atomics.");
            //    append_stat_ln(gpu_name + " not supported.");
            //    return true;
            //}

            append_stat_ln("Running on " + gpu_name);

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
        //                 Copy from GPU to CPU and vice-versa 
        // ==================================================================

        dim3 blocks_per_grid_2D_pix;
        int blocks_per_grid_1D_axons;

        private void load_gpu_from_cpu()
        {
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

            gpu.FreeAll(); gpu.Synchronize();

            tox_dev = gpu.Allocate(tox); gpu.CopyToDevice(tox, tox_dev);
            rate_dev = gpu.Allocate(rate); gpu.CopyToDevice(rate, rate_dev);
            detox_dev = gpu.Allocate(detox); gpu.CopyToDevice(detox, detox_dev);
            tox_prod_dev = gpu.Allocate(tox_prod); gpu.CopyToDevice(tox_prod, tox_prod_dev);

            axons_cent_pix_dev = gpu.Allocate(axons_cent_pix); gpu.CopyToDevice(axons_cent_pix, axons_cent_pix_dev);
            axon_is_alive_dev = gpu.Allocate(axon_is_alive); gpu.CopyToDevice(axon_is_alive, axon_is_alive_dev);

            pix_idx_dev = gpu.Allocate(pix_idx); gpu.CopyToDevice(pix_idx, pix_idx_dev);

            num_alive_axons_dev = gpu.Allocate<int>(1); gpu.CopyToDevice(num_alive_axons, num_alive_axons_dev);
            death_itr_dev = gpu.Allocate(death_itr); gpu.CopyToDevice(death_itr, death_itr_dev);
            bmp_bytes_dev = gpu.Allocate(bmp_bytes); gpu.CopyToDevice(bmp_bytes, bmp_bytes_dev);
            init_insult_mask_dev = gpu.Allocate<byte>(bmp_im_size, bmp_im_size);

            sum_tox_dev = gpu.Allocate<float>(1);
            progress_dev = gpu.Allocate<float>(3);

            progression_image_sum_float_dev = gpu.Allocate<float>(prog_im_siz, prog_im_siz);
            progress_image_num_averaged_pix_dev = gpu.Allocate<uint>(prog_im_siz, prog_im_siz);
            progression_image_dev = gpu.Allocate<byte>(prog_im_siz, prog_im_siz);

            // ==================== Constants

            int tmp = (int)Math.Ceiling(Math.Sqrt((double)pix_idx_num / (double)threads_per_block_1D));
            blocks_per_grid_2D_pix = new dim3(tmp, tmp);
            blocks_per_grid_1D_axons = mdl.n_axons / threads_per_block_1D + 1;

            show_opts_dev = gpu.Allocate(show_opts); gpu.CopyToDevice(show_opts, show_opts_dev);
            
            axons_inside_pix_dev = gpu.Allocate(axons_inside_pix); gpu.CopyToDevice(axons_inside_pix, axons_inside_pix_dev);
            axons_inside_pix_idx_dev = gpu.Allocate(axons_inside_pix_idx); gpu.CopyToDevice(axons_inside_pix_idx, axons_inside_pix_idx_dev);

            axon_surr_rate_dev = gpu.Allocate(axons_surr_rate); gpu.CopyToDevice(axons_surr_rate, axon_surr_rate_dev);
            axon_surr_rate_idx_dev = gpu.Allocate(axons_surr_rate_idx); gpu.CopyToDevice(axons_surr_rate_idx, axon_surr_rate_idx_dev);

            axon_mask_dev = gpu.Allocate(axon_mask); gpu.CopyToDevice(axon_mask, axon_mask_dev);

            gpu.Synchronize();

            Debug.WriteLine("GPU used memory: " + (100.0 * (1 - (double)gpu.FreeMemory / (double)gpu.TotalMemory)).ToString("0.0") + " %\n");
        }
        /*
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
        */

        // ==================================================================
        //         Dummy functions (defined in native cuda @ cuda/...)
        // ==================================================================

        [CudafyDummy]
        public static void cuda_update_live() { }
        [CudafyDummy]
        public static void cuda_diffusion() { }
        [CudafyDummy]
        public static void cuda_update_image() { }
        [CudafyDummy]
        public static void cuda_tox_sum() { }
        [CudafyDummy]
        public static void cuda_prep0() { }
        [CudafyDummy]
        public static void cuda_prep1() { }
        [CudafyDummy]
        public static void cuda_update_init_insult() { }
    }
}

