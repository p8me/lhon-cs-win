﻿//
//  preprocess.cs
//  LHON-Form
//
//  Created by Pooya Merat in 2016.
//

using System;
using System.Diagnostics;
using System.Linq;
using Cudafy;
using Cudafy.Host;

namespace LHON_Form
{
    public partial class Main_Form
    {

        // ========================
        //      Constants
        // ========================

        float min_res = 10F;

        // ========================
        //      Parameters
        // ========================

        /* Units:
        Constant                        UI Unit                  Algorithm Unit          Conversion Factor
        k_detox, k_rate                  1 / sec                   1 / itr               K / resolution ^ 2
        k_tox_prod                       micromol / um^2 / sec     tox / pix / itr       K / resolution ^ 4
        death_tox_thres, insult_tox      micromol / um^2           tox / pix             K / resolution ^ 2
        
            sec = (CONSTANT / resolution ^ 2) * itr
            micromol / um^2 = (CONSTANT * resolution ^ 2) * tox / pix
        */

        float k_detox_intra, k_detox_extra, k_tox_prod, death_tox_thres, insult_tox, on_death_tox,
            k_rate_live_axon, k_rate_boundary, k_rate_dead_axon, k_rate_extra;


        // ====================================
        //              Variables
        // ====================================

        float[,] tox, tox_init, tox_dev, tox_new_dev; // Tox
        float[,,] rate, rate_init, rate_dev; // Rate
        float[,] detox, detox_init, detox_dev; // Detox
        float[,] tox_prod, tox_prod_init, tox_prod_dev; // Tox_prod
        uint[] axons_cent_pix, axons_cent_pix_dev; // Center pixel of each axon
        uint[] axons_inside_pix, axons_inside_pix_dev; // 1D array for all axons
        uint[] axons_inside_pix_idx, axons_inside_pix_idx_dev; // indices for the above 1D array
        uint[] axons_surr_rate, axon_surr_rate_dev; // 1D indices of rate array that have the boundary rate and are outside axons
        uint[] axons_surr_rate_idx, axon_surr_rate_idx_dev; // indices for above array
        uint[] death_itr, death_itr_dev; // death iteration of each axon
        byte[,] axon_mask, axon_mask_init, axon_mask_dev;

        // Index of pixels inside the nerve (used for cuda functions)
        int[] pix_idx, pix_idx_dev;
        int pix_idx_num; // number of pixels inside the nerve

        bool[] axon_is_alive, axon_is_alive_init, axon_is_alive_dev;
        int[] num_alive_axons = new int[1], num_alive_axons_dev;

        bool[] axon_is_init_insult;

        bool[] axon_is_large; // For display purposes

        ushort im_size;

        // ====================================
        //        Model Preprocessing
        // ====================================
        ushort calc_im_siz()
        {
            return (ushort)((mdl_nerve_r * setts.resolution + 2) * 2);
        }

        // Requires full Model info and assigns tox, rate, etc
        private void preprocess_model()
        {
            

            
            // =======================================
            //              Init Parameters
            // =======================================

            float res = setts.resolution;
            mdl_nerve_r = mdl.nerve_scale_ratio * mdl_real_nerve_r;

            float rate_conv = pow2(res / min_res);
            float temp = 1F / rate_conv;
            if (setts.rate_live > temp ||
                setts.rate_bound > temp ||
                setts.rate_dead > temp ||
                setts.rate_extra > temp)
            {
                append_stat_ln("Bad diffusion rate (>0.2)! Increase min_res. Preprocess aborted.");
                return;
            }

            // "setts" are user input with physical units

            // 1 - real detox rate to reduce computation ->  tox[x_y] *= detox[x_y]
            k_detox_intra = 1F - setts.detox_intra;
            k_detox_extra = 1F - setts.detox_extra;
            k_tox_prod = setts.tox_prod;

            // User inputs 0 to 1 for rate values 
            k_rate_live_axon = setts.rate_live / 5F * rate_conv;
            k_rate_boundary = setts.rate_bound / 5F * rate_conv;
            k_rate_dead_axon = setts.rate_dead / 5F * rate_conv;
            k_rate_extra = setts.rate_extra / 5F * rate_conv;

            // 
            death_tox_thres = setts.death_tox_thres;
            insult_tox = setts.insult_tox;
            on_death_tox = setts.on_death_tox;

            prep_prof.time(0);
            tic();

            update_bottom_stat("Preprocessing ...");

            im_size = calc_im_siz();
            update_image_siz_lbl();

            init_bmp_write();

            // ======== Pixel Properties =========
            rate = rate_init = new float[im_size, im_size, 4];
            detox = detox_init = new float[im_size, im_size];
            tox = tox_init = new float[im_size, im_size];
            tox_prod = tox_prod_init = new float[im_size, im_size];

            axon_mask = axon_mask_init = new byte[im_size, im_size]; // for display. if 1: axon is live, if 2: axon is dead, otherwise 0.
            pix_idx = new int[im_size * im_size]; // linear index of pixels within nerve

            // ======== Axon Properties =========

            axons_cent_pix = new uint[mdl.n_axons];
            axon_is_alive = axon_is_alive_init = Enumerable.Repeat(true, mdl.n_axons).ToArray(); // init to true

            // temp variable
            int max_pixels_in_nerve = (int)(pow2(mdl_nerve_r * res) * (1 - pow2(mdl_vessel_ratio)) * Math.PI);

            axons_inside_pix = new uint[max_pixels_in_nerve * 3 / 4];
            axons_inside_pix_idx = new uint[mdl.n_axons + 1];

            axons_surr_rate = new uint[max_pixels_in_nerve / 5];
            axons_surr_rate_idx = new uint[mdl.n_axons + 1];

            axon_is_init_insult = new bool[mdl.n_axons];

            axon_is_large = new bool[mdl.n_axons]; // For display purposes

            death_itr = new uint[mdl.n_axons];

            axon_lbl = new axon_lbl_class[mdl.n_axons]; // for GUI

            // ======== Local Constants =========
            int nerve_cent_pix = im_size / 2;
            int nerve_r_pix = (int)(mdl_nerve_r * res);
            int vein_r_pix = (int)(mdl_vessel_ratio * mdl_nerve_r * res);

            int nerve_r_pix_2 = pow2(nerve_r_pix);
            int vein_r_pix_2 = pow2(vein_r_pix);

            // GPU init for prep0 and prep1
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.FreeAll(); gpu.Synchronize();

            pix_idx_num = 0;
            rate_dev = gpu.Allocate<float>(im_size, im_size, 4);
            detox_dev = gpu.Allocate<float>(im_size, im_size);
            pix_idx_dev = gpu.Allocate<int>(im_size * im_size);
            byte[] pix_out_of_nerve_dev = gpu.Allocate<byte>(im_size * im_size);
            byte[] pix_out_of_nerve = new byte[im_size * im_size];

            int prep_siz = 32;
            dim3 block_siz_prep = new dim3(prep_siz, prep_siz);
            int tmp = (int)Math.Ceiling((double)im_size / (double)prep_siz);
            dim3 grid_siz_prep = new dim3(tmp, tmp);

            gpu.Launch(grid_siz_prep, block_siz_prep).cuda_pre_prep(im_size, nerve_cent_pix, nerve_r_pix_2, vein_r_pix_2, k_rate_extra, k_detox_extra,
                pix_out_of_nerve_dev, rate_dev, detox_dev);

            gpu.Synchronize();

            gpu.CopyFromDevice(pix_out_of_nerve_dev, pix_out_of_nerve);
            gpu.CopyFromDevice(rate_dev, rate);
            gpu.CopyFromDevice(detox_dev, detox);

            prep_prof.time(1);

            for (int idx = 0; idx < im_size * im_size; idx++)
                if (pix_out_of_nerve[idx] == 0)
                    pix_idx[pix_idx_num++] = idx;

            prep_prof.time(2);

            float[,] AxCorPix = new float[mdl.n_axons, 3]; // axon coordinate in pixels

            // ======== Individual Axon Properties Initialization =========

            int[] box_y_min = new int[mdl.n_axons],
                box_y_max = new int[mdl.n_axons],
                box_x_min = new int[mdl.n_axons],
                box_x_max = new int[mdl.n_axons],
                box_siz_x = new int[mdl.n_axons],
                box_siz_y = new int[mdl.n_axons];


            for (int i = 0; i < mdl.n_axons; i++)
            {
                axon_is_large[i] = mdl.axon_coor[i][2] > axon_max_r_mean;

                axon_is_init_insult[i] = pow2(insult_r - mdl.axon_coor[i][2]) > pow2(insult_x - mdl.axon_coor[i][0]) + pow2(insult_y - mdl.axon_coor[i][1]);

                // Change coordinates from um to pixels
                float xc = nerve_cent_pix + mdl.axon_coor[i][0] * res;
                float yc = nerve_cent_pix + mdl.axon_coor[i][1] * res;
                float rc = mdl.axon_coor[i][2] * res;
                AxCorPix[i, 0] = xc; AxCorPix[i, 1] = yc; AxCorPix[i, 2] = rc;
                death_itr[i] = 0;
                axons_cent_pix[i] = (uint)xc * im_size + (uint)yc;
                axon_lbl[i] = new axon_lbl_class { lbl = "", x = xc, y = yc };
                axons_surr_rate_idx[i + 1] = axons_surr_rate_idx[i];

                float rc_1 = rc + 1;
                box_y_min[i] = Max((int)(yc - rc_1), 0);
                box_y_max[i] = Min((int)(yc + rc_1), im_size - 1);
                box_x_min[i] = Max((int)(xc - rc_1), 0);
                box_x_max[i] = Min((int)(xc + rc_1), im_size - 1);
                box_siz_x[i] = box_y_max[i] - box_y_min[i] + 2;
                box_siz_y[i] = box_x_max[i] - box_x_min[i] + 2;
            }
            prep_prof.time(3);

            for (int i = 0; i < mdl.n_axons; i++)
            {
                bool[,] is_inside_this_axon = new bool[box_siz_x[i], box_siz_y[i]];
                axons_inside_pix_idx[i + 1] = axons_inside_pix_idx[i];

                for (int y = box_y_min[i]; y <= box_y_max[i]; y++)
                    for (int x = box_x_min[i]; x <= box_x_max[i]; x++)
                    {
                        float dx = (float)x - AxCorPix[i, 0];
                        float dy = (float)y - AxCorPix[i, 1];
                        bool inside = AxCorPix[i, 2] * AxCorPix[i, 2] - (dx * dx + dy * dy) > 0;
                        if (inside)
                        { // inside axon
                            is_inside_this_axon[x - box_x_min[i], y - box_y_min[i]] = true;
                            axon_mask[x, y] = 1; // alive
                            axons_inside_pix[axons_inside_pix_idx[i + 1]++] = (uint)x * (uint)im_size + (uint)y;
                            tox_prod[x, y] = k_tox_prod;
                            detox[x, y] = k_detox_intra;
                            if (axon_is_init_insult[i])
                                tox[x, y] = insult_tox;
                        }
                    }
                prep_prof.time(4);

                for (int y = box_y_min[i] + 1; y < box_y_max[i]; y++)
                    for (int x = box_x_min[i] + 1; x < box_x_max[i]; x++)
                    {
                        int x_rel = x - box_x_min[i];
                        int y_rel = y - box_y_min[i];

                        int[] neighbors_x = new int[] { x_rel + 1, x_rel - 1, x_rel, x_rel };
                        int[] neighbors_y = new int[] { y_rel, y_rel, y_rel + 1, y_rel - 1 };

                        for (uint k = 0; k < 4; k++)
                        {
                            bool xy_inside = is_inside_this_axon[x_rel, y_rel];
                            bool neigh_k_inside = is_inside_this_axon[neighbors_x[k], neighbors_y[k]];

                            if (xy_inside != neigh_k_inside)
                            {
                                rate[x, y, k] = k_rate_boundary;
                                if (neigh_k_inside)
                                    axons_surr_rate[axons_surr_rate_idx[i + 1]++] = (uint)(((uint)x * (uint)im_size + (uint)y) * 4 + (uint)k);
                            }
                            else if (xy_inside)
                                rate[x, y, k] = k_rate_live_axon;
                        }
                    }
                prep_prof.time(5);
                // Verify radius
                // Debug.WriteLine("{0} vs {1}", (Math.Pow(mdl.axon_coor[i][2] * res, 2) * Math.PI).ToString("0.0"), axons_inside_pix_idx[i + 1] - axons_inside_pix_idx[i]);
            }

            gpu.Launch(grid_siz_prep, block_siz_prep).cuda_prep(im_size, pix_out_of_nerve_dev, rate_dev);
            gpu.CopyFromDevice(rate_dev, rate);

            prep_prof.time(6);

            // Keep backup of inital state 

            //tox_init = null; tox_init = (float[,])tox.Clone();
            //rate_init = null; rate_init = (float[,,])rate.Clone();
            //detox_init = null; detox_init = (float[,])detox.Clone();
            //tox_init = null; tox_init = null; tox_prod_init = (float[,])tox_prod.Clone();
            //axon_mask_init = null; axon_mask_init = (byte[,])axon_mask.Clone();
            //axon_is_alive_init = null; axon_is_alive_init = (bool[])axon_is_alive.Clone();

            reset_state();

            // variable size study
            //((rate.Length + tox.Length + detox.Length + tox_prod.Length + axon_mask.Length + axon_is_alive.Length)*4)/1024/1024 // MB

            update_bottom_stat("Preprocess Done! (" + (toc() / 1000).ToString("0.0") + " secs)");
            // Debug.WriteLine("inside: {0} vs allocated {1}", axons_inside_pix_idx[mdl.n_axons - 1], axons_inside_pix.Length);

            prep_prof.report();
        }
    }
}
