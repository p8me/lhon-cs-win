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
        uint[] death_itr, death_itr_dev; // death iteration of each axon
        uint[] axons_inside_npix; // for debugging

        // Index of pixels inside the nerve (for linear indexing of GPU threads)
        uint[] pix_idx, pix_idx_dev;
        uint pix_idx_num; // number of pixels inside the nerve

        bool[] axon_is_alive, axon_is_alive_dev;
        int[] num_alive_axons = new int[1], num_alive_axons_dev;

        float[,] axons_coor; // Final result of model generation
        bool[] axon_is_large; // For display purpuses

        // ====================================
        //        Model Preprocessing
        // ====================================

        ushort calc_im_siz()
        {
            return (ushort)((((mdl.nerve_r * setts.resolution) * 2 + 1) / threads_per_block_1D + 1) * threads_per_block_1D);
        }


        // Requires full Model info and assigns tox, rate, etc
        private void preprocess_model()
        {
            if (mdl.n_axons == 0)
            {
                append_stat_ln("No axons in the model! Preprocess aborted.");
                return;
            }

            // Init constants

            float max_res = 10F;

            float res2_fact = setts.resolution * setts.resolution / (max_res * max_res);

            float rate_and_detox_conv = 1F / res2_fact;

            float tox_prod_conv = 1F / (res2_fact * res2_fact);

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

            axons_cent_pix = new uint[mdl.n_axons];

            // ======== Image Properties Initialization =========
            int nerve_cent_pix = im_size / 2;
            int nerve_r_pix = (int)(mdl.nerve_r * setts.resolution);
            int vein_r_pix = (int)(mdl.vessel_rat * mdl.nerve_r * setts.resolution);
            Func<int, int, int, int> within_circle2_int = (x, y, r) =>
            {
                int dx = x - nerve_cent_pix;
                int dy = y - nerve_cent_pix;
                return r * r - (dx * dx + dy * dy);
            };

            alg_prof.time(1);

            // ======== Common Axon Properties (+Initialization) =========

            update_num_axons_lbl();

            // Assign max memory
            int max_pixels_in_nerve = (int)(Math.Pow(mdl.nerve_r * setts.resolution, 2) * Math.PI) -
                (int)(Math.Pow(mdl.nerve_r * mdl.vessel_rat * setts.resolution, 2) * Math.PI);

            // ======== Individual Axon Properties =========

            axons_inside_pix = new uint[max_pixels_in_nerve];
            axons_inside_pix_idx = new uint[mdl.n_axons + 1];
            axons_inside_npix = new uint[mdl.n_axons];

            axon_is_large = new bool[mdl.n_axons];

            axon_is_alive = new bool[mdl.n_axons];
            death_itr = new uint[mdl.n_axons];

            axon_lbl = new axon_lbl_class[mdl.n_axons];

            axons_coor = new float[mdl.n_axons, 3];

            // ======== Individual Axon Properties Initialization =========

            bool[,] occupied = new bool[im_size, im_size];

            int inside_axon_arr_cnt = 0;

            for (int i = 0; i < mdl.n_axons; i++)
            {
                axon_is_large[i] = mdl.axon_coor[i][2] > mdl.max_r;

                // Change coordinates from um to pixels
                float xc = nerve_cent_pix + mdl.axon_coor[i][0] * setts.resolution;
                float yc = nerve_cent_pix + mdl.axon_coor[i][1] * setts.resolution;
                float rc = mdl.axon_coor[i][2] * setts.resolution;

                if (rc > 10 * mdl.min_r_abs) axon_is_large[i] = true;

                axons_coor[i, 0] = xc; axons_coor[i, 1] = yc; axons_coor[i, 2] = rc;

                axon_is_alive[i] = true;
                death_itr[i] = 0;

                axons_cent_pix[i] = (uint)xc * im_size + (uint)yc;

                axon_lbl[i] = new axon_lbl_class { lbl = "", x = xc, y = yc };

                axons_inside_pix_idx[i + 1] = axons_inside_pix_idx[i];

                float extra_siz = 2 + rc;

                int[] box_y = new int[] { Max((int)(yc - extra_siz), 0), Min((int)(yc + extra_siz), im_size) };
                int[] box_x = new int[] { Max((int)(xc - extra_siz), 0), Min((int)(xc + extra_siz), im_size) };

                int[] box_siz = new int[] { box_y[1] - box_y[0] + 1, box_x[1] - box_x[0] + 1 };

                float[,] dist = new float[box_siz[0], box_siz[1]];

                for (int y = box_y[0]; y <= box_y[1]; y++)
                    for (int x = box_x[0]; x <= box_x[1]; x++)
                        dist[y - box_y[0], x - box_x[0]] = within_circle2(x, y, xc, yc, rc);

                for (int y = box_y[0]; y <= box_y[1]; y++)
                    for (int x = box_x[0]; x <= box_x[1]; x++)
                    {
                        Func<int, int, float> dst = (X, Y) => dist[Y - box_y[0], X - box_x[0]];

                        bool inside = dst(x, y) > 0;

                        if (inside)
                        { // inside axon
                            axons_inside_npix[i]++;
                            axons_inside_pix[inside_axon_arr_cnt++] = (uint)x * (uint)im_size + (uint)y;
                            axons_inside_pix_idx[i + 1]++;

                            occupied[x, y] = true;
                            // tox[x, y] = 1F;

                            tox_prod[x, y] = k_tox_prod;
                            detox[x, y] = k_detox_intra;
                        }
                    }
                // Verify radius
                // Debug.WriteLine("{0} vs {1}", (Math.Pow(mdl.axon_coor[i][2] * setts.resolution, 2) * Math.PI).ToString("0.0"), axons_inside_pix_idx[i + 1] - axons_inside_pix_idx[i]);
            }

            alg_prof.time(3);

            pix_idx = new uint[im_size * im_size];
            pix_idx_num = 0;

            // ================== Assign Rates

            bool[,] pix_out_of_nerve = new bool[im_size, im_size];

            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                {
                    pix_out_of_nerve[x, y] = within_circle2_int(x, y, nerve_r_pix) < 0 || within_circle2_int(x, y, vein_r_pix) > 0;
                    if (!occupied[x, y])
                        detox[x, y] = k_detox_extra;
                }

            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                {
                    int[,] arr = new int[,] { { x + 1, y }, { x - 1, y }, { x, y + 1 }, { x, y - 1 } };

                    if (pix_out_of_nerve[x, y]) // Outside nerve area
                        for (int k = 0; k < 4; k++)
                            rate[x, y, k] = 0;
                    else // pix_within_nerve area
                    {
                        pix_idx[pix_idx_num++] += (uint)x * (uint)im_size + (uint)y;

                        for (int k = 0; k < 4; k++)
                        {
                            if (pix_out_of_nerve[arr[k, 0], arr[k, 1]])
                                rate[x, y, k] = 0;
                            else if (occupied[x, y])
                                rate[x, y, k] = occupied[arr[k, 0], arr[k, 1]] ? k_rate_live_axon : k_rate_boundary;
                            else
                                rate[x, y, k] = occupied[arr[k, 0], arr[k, 1]] ? k_rate_boundary : k_rate_extra;
                        }
                    }
                }


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

            alg_prof.time(5);

            areal_progress_lim = areal_progress_lim / temp * 0.7F;

            reset_state();

            update_bottom_stat("Preprocess Done! (" + (toc() / 1000).ToString("0.0") + " secs)");

            Debug.WriteLine("inside: {0} vs allocated {1}", inside_axon_arr_cnt, axons_inside_pix.Length);

            alg_prof.report();

        }

    }
}
