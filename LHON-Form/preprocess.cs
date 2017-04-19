using System;
using System.Diagnostics;

namespace LHON_Form
{
    public partial class Main_Form
    {

        /* Units:
        Constant         UI Unit                  Algorithm Unit          Conversion Factor
        k_detox           1 / sec                   1 / itr               K * resolution ^ 2
        k_rate            1 / sec                   1 / itr               
        k_tox_prod        micromol / um^2 / sec     tox / pix / itr       
        death_tox_lim     micromol / um^2           tox / pix             

            sec = (CONSTANT / resolution ^ 2) * itr
            micromol / um^2 = (CONSTANT * resolution ^ 2) * tox / pix
        */

        float k_detox_intra, k_detox_extra, k_tox_prod, death_tox_lim,
            k_rate_live_axon, k_rate_boundary, k_rate_dead_axon, k_rate_extra;


        // ====================================
        //              Variables
        // ====================================

        float[,] tox, tox_init, tox_dev; // Tox
        float[,,] rate, rate_init, rate_dev; // Rate
        float[,] detox, detox_init, detox_dev; // Detox
        float[,] tox_prod, tox_prod_init, tox_prod_dev; // Tox_prod
        uint[] axons_cent_pix, axons_cent_pix_dev; // Center pixel of each axon
        uint[] axons_inside_pix, axons_inside_pix_dev; // 1D array for all axons
        uint[] axons_inside_pix_idx, axons_inside_pix_idx_dev; // indices for the above 1D array
        uint[] axons_surr_rate, axon_surr_rate_dev; // 1D indices of rate array that have the boundary rate and are outside axons
        uint[] axons_surr_rate_idx, axon_surr_rate_idx_dev; // indices for above array
        uint[] death_itr, death_itr_dev; // death iteration of each axon
        byte[,] axon_mask, axon_mask_dev;
        //uint[] axons_inside_npix; // for debugging


        // Index of pixels inside the nerve (for linear indexing of GPU threads)
        int[] pix_idx, pix_idx_dev;
        int pix_idx_num; // number of pixels inside the nerve

        bool[] axon_is_alive, axon_is_alive_dev;
        int[] num_alive_axons = new int[1], num_alive_axons_dev;

        float[,] axons_coor; // Final result of model generation
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
            if (mdl.n_axons == 0)
            {
                append_stat_ln("No axons in the model! Preprocess aborted.");
                return;
            }

            mdl_nerve_r = mdl.nerve_scale_ratio * mdl_real_nerve_r;

            // Init constants

            float max_res = 10F;

            float res2_fact = pow2f(setts.resolution / max_res);
            float rate_and_detox_conv = 1F / res2_fact;
            float tox_prod_conv = 1F / pow2f(res2_fact);

            // "setts" are user input with physical units

            // 1 - real detox rate to reduce computation ->  tox[x_y] *= detox[x_y]
            k_detox_intra = (1F - setts.detox_intra) * rate_and_detox_conv;
            k_detox_extra = (1F - setts.detox_extra) * rate_and_detox_conv;
            k_tox_prod = setts.tox_prod * tox_prod_conv;

            // User inputs 0 to 1 for rate values
            k_rate_live_axon = setts.rate_live / 5F * rate_and_detox_conv;
            k_rate_boundary = setts.rate_bound / 5F * rate_and_detox_conv;
            k_rate_dead_axon = setts.rate_dead / 5F * rate_and_detox_conv;
            k_rate_extra = setts.rate_extra / 5F * rate_and_detox_conv;
            death_tox_lim = setts.death_tox_lim * rate_and_detox_conv;

            alg_prof.time(0);
            tic();

            update_bottom_stat("Preprocessing ...");

            im_size = calc_im_siz();
            update_image_siz_lbl();

            init_bmp_write();

            // ======== Image Properties =========
            rate = new float[im_size, im_size, 4];
            rate_init = new float[im_size, im_size, 4];

            detox = new float[im_size, im_size];
            detox_init = new float[im_size, im_size];

            tox = new float[im_size, im_size];
            tox_init = new float[im_size, im_size];

            tox_prod = new float[im_size, im_size];
            tox_prod_init = new float[im_size, im_size];

            axon_mask = new byte[im_size, im_size];

            axons_cent_pix = new uint[mdl.n_axons];

            // ======== Image Properties Initialization =========
            int nerve_cent_pix = im_size / 2;
            int nerve_r_pix = (int)(mdl_nerve_r * setts.resolution);
            int vein_r_pix = (int)(mdl.vessel_ratio * mdl_nerve_r * setts.resolution);
            Func<int, int, int, int> within_circle2_int = (x, y, r) =>
            {
                int dx = x - nerve_cent_pix;
                int dy = y - nerve_cent_pix;
                return r * r - (dx * dx + dy * dy);
            };
            
            // ======== Common Axon Properties (+Initialization) =========

            update_num_axons_lbl();

            // Assign max memory
            int max_pixels_in_nerve = (int)(Math.Pow(mdl_nerve_r * setts.resolution, 2) * Math.PI) -
                (int)(Math.Pow(mdl_nerve_r * mdl.vessel_ratio * setts.resolution, 2) * Math.PI);

            pix_idx = new int[im_size * im_size];
            pix_idx_num = 0;

            bool[,] pix_out_of_nerve = new bool[im_size, im_size];

            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                {
                    pix_out_of_nerve[x, y] = within_circle2_int(x, y, nerve_r_pix) < 0 || within_circle2_int(x, y, vein_r_pix) > 0;
                    if (!pix_out_of_nerve[x, y])
                    {
                        pix_idx[pix_idx_num++] += x * im_size + y;
                        for (uint k = 0; k < 4; k++)
                            rate[x, y, k] = k_rate_extra;
                        detox[x, y] = k_detox_extra;
                    }
                }

            alg_prof.time(1);

            // ======== Individual Axon Properties =========

            axons_inside_pix = new uint[max_pixels_in_nerve * 3 / 4];
            axons_inside_pix_idx = new uint[mdl.n_axons + 1];
            //axons_inside_npix = new uint[mdl.n_axons];

            axons_surr_rate = new uint[max_pixels_in_nerve / 2];
            axons_surr_rate_idx = new uint[mdl.n_axons + 1];

            axon_is_large = new bool[mdl.n_axons]; // For display purposes

            axon_is_alive = new bool[mdl.n_axons];
            death_itr = new uint[mdl.n_axons];

            axon_lbl = new axon_lbl_class[mdl.n_axons];

            axons_coor = new float[mdl.n_axons, 3];

            // ======== Individual Axon Properties Initialization =========

            for (int i = 0; i < mdl.n_axons; i++)
            {
                axon_is_large[i] = mdl.axon_coor[i][2] > axon_max_r_mean;

                bool[,] is_inside_this_axon = new bool[im_size, im_size];

                // Change coordinates from um to pixels
                float xc = nerve_cent_pix + mdl.axon_coor[i][0] * setts.resolution;
                float yc = nerve_cent_pix + mdl.axon_coor[i][1] * setts.resolution;
                float rc = mdl.axon_coor[i][2] * setts.resolution;

                axons_coor[i, 0] = xc; axons_coor[i, 1] = yc; axons_coor[i, 2] = rc;

                axon_is_alive[i] = true;
                death_itr[i] = 0;

                axons_cent_pix[i] = (uint)xc * im_size + (uint)yc;

                axon_lbl[i] = new axon_lbl_class { lbl = "", x = xc, y = yc };

                axons_inside_pix_idx[i + 1] = axons_inside_pix_idx[i];
                axons_surr_rate_idx[i + 1] = axons_surr_rate_idx[i];

                float extra_siz = 1 + rc;

                int[] box_y = new int[] { Max((int)(yc - extra_siz), 0), Min((int)(yc + extra_siz), im_size) };
                int[] box_x = new int[] { Max((int)(xc - extra_siz), 0), Min((int)(xc + extra_siz), im_size) };

                int[] box_siz = new int[] { box_y[1] - box_y[0] + 1, box_x[1] - box_x[0] + 1 };

                float[,] dist = new float[box_siz[0], box_siz[1]];

                for (int y = box_y[0]; y <= box_y[1]; y++)
                    for (int x = box_x[0]; x <= box_x[1]; x++)
                    {
                        bool inside = within_circle2(x, y, xc, yc, rc) > 0;

                        if (inside)
                        { // inside axon
                            is_inside_this_axon[x, y] = true;
                            axon_mask[x, y] = 1; // alive
                            //axons_inside_npix[i]++;
                            axons_inside_pix[axons_inside_pix_idx[i + 1]++] = (uint)(x * im_size + y);
                            tox_prod[x, y] = k_tox_prod;
                            detox[x, y] = k_detox_intra;
                            tox[x, y] = death_tox_lim * 0.95F; // For debugging
                        }
                    }
                alg_prof.time(2);

                for (int y = box_y[0]; y <= box_y[1]; y++)
                    for (int x = box_x[0]; x <= box_x[1]; x++)
                    {
                        int[,] neighbors = new int[,] { { x + 1, y }, { x - 1, y }, { x, y + 1 }, { x, y - 1 } };
                        for (uint k = 0; k < 4; k++)
                        {
                            bool xy_inside = is_inside_this_axon[x, y];
                            bool neigh_k_inside = is_inside_this_axon[neighbors[k, 0], neighbors[k, 1]];

                            if (xy_inside != neigh_k_inside)
                            {
                                rate[x, y, k] = k_rate_boundary;
                                if (neigh_k_inside)
                                    axons_surr_rate[axons_surr_rate_idx[i + 1]++] = ((uint)x * (uint)im_size + (uint)y) * 4 + k;
                            }
                            else if (xy_inside)
                                rate[x, y, k] = k_rate_live_axon;
                        }
                    }
                alg_prof.time(3);
                // Verify radius
                // Debug.WriteLine("{0} vs {1}", (Math.Pow(mdl.axon_coor[i][2] * setts.resolution, 2) * Math.PI).ToString("0.0"), axons_inside_pix_idx[i + 1] - axons_inside_pix_idx[i]);
            }           

            // Set nerve boundary rates to 0
            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                {
                    int[,] neighbors = new int[,] { { x + 1, y }, { x - 1, y }, { x, y + 1 }, { x, y - 1 } };
                    for (uint k = 0; k < 4; k++)
                        if (pix_out_of_nerve[x, y] || pix_out_of_nerve[neighbors[k, 0], neighbors[k, 1]])
                            rate[x, y, k] = 0;
                }

            alg_prof.time(4);

            areal_progress_lim = 0;
            int temp = 0;

            // Keep backup of inital state
            tox_init = (float[,])tox.Clone();
            rate_init = (float[,,])rate.Clone();
            detox_init = (float[,])detox.Clone();
            tox_prod_init = (float[,])tox_prod.Clone();

            /* NO_GUI
            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                    if (within_circle2_int(x, y, nerve_r_pix) > 0)
                    {
                        temp++;
                        if (tox[x, y] > 0) areal_progress_lim++;
                    }
            */

            areal_progress_lim = areal_progress_lim / temp * 0.7F;

            reset_state();

            update_bottom_stat("Preprocess Done! (" + (toc() / 1000).ToString("0.0") + " secs)");

            // Debug.WriteLine("inside: {0} vs allocated {1}", axons_inside_pix_idx[mdl.n_axons - 1], axons_inside_pix.Length);

            alg_prof.report();

        }
    }
}
