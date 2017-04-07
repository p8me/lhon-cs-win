using System;
using System.Diagnostics;

namespace LHON_Form
{
    public partial class Main_Form
    {

        /* Units:
        Constant         UI Unit        Algorithm Unit
        k_detox          1 / sec         1 / itr
        k_rate           1 / sec         1 / itr
        k_tox_prod       c / sec         c / itr
        */

        float k_detox_intra;

        float k_detox_extra;

        float k_tox_prod;

        float k_rate_live_axon;

        float k_rate_boundary;

        float k_rate_dead_axon;

        float k_rate_extra = 1F / 5F; // extracellular rate

        // ====================================
        //              Variables
        // ====================================

        // Tox
        float[,] tox, tox_init, tox_dev;

        // Rate
        float[,,] rate, rate_init, rate_dev;

        // Detox
        float[,] detox, detox_init, detox_dev;

        // Tox_prod
        float[,] tox_prod, tox_prod_init, tox_prod_dev;

        // Center pixel of each axon
        uint[] axons_cent_pix, axons_cent_pix_dev;

        uint[] axons_inside_pix, axons_inside_pix_dev;

        uint[] axons_inside_pix_idx, axons_inside_pix_idx_dev;

        uint[] axons_inside_npix, death_itr, axons_inside_npix_dev, death_itr_dev;

        uint[] pix_idx, pix_idx_dev;
        uint pix_idx_num;

        bool[] live_neur, live_neur_dev;
        int[] num_live_neur = new int[1], num_live_neur_dev;

        float[,] axons_coor;
        bool[] show_neur_lvl;

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
            // Init constants

            float max_res = 10;

            float res2 = setts.resolution * setts.resolution;

            float rate_and_detox_conv = res2 / (max_res * max_res);

            float tox_prod_conv = res2 * res2 / (max_res * max_res * max_res * max_res);

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

            axons_cent_pix = new uint[im_size * im_size];

            // ======== Image Properties Initialization =========
            int nerve_cent_pix = im_size / 2;
            int nerve_r_pix = (int)(mdl.nerve_r * setts.resolution);
            int vein_r_pix = (int)(mdl.vein_rat * mdl.nerve_r * setts.resolution);
            Func<int, int, int, int> within_circle2_int = (x, y, r) =>
            {
                int dx = x - nerve_cent_pix;
                int dy = y - nerve_cent_pix;
                return r * r - (dx * dx + dy * dy);
            };

            alg_prof.time(1);

            // ======== Common Neuron Properties (+Initialization) =========

            update_n_neur_lbl();

            // Assign max memory
            int max_pixels_in_nerve = (int)(Math.Pow(mdl.nerve_r * setts.resolution, 2) * Math.PI) -
                (int)(Math.Pow(mdl.nerve_r * mdl.vein_rat * setts.resolution, 2) * Math.PI);

            // ======== Individual Neuron Properties =========

            axons_inside_pix = new uint[max_pixels_in_nerve];
            axons_inside_pix_idx = new uint[mdl.n_neurs + 1];
            axons_inside_npix = new uint[mdl.n_neurs];

            show_neur_lvl = new bool[mdl.n_neurs];

            live_neur = new bool[mdl.n_neurs];
            death_itr = new uint[mdl.n_neurs];

            neur_lbl = new neur_lbl_class[mdl.n_neurs];

            axons_coor = new float[mdl.n_neurs, 3];

            // ======== Individual Neuron Properties Initialization =========

            bool[,] occupied = new bool[im_size, im_size];

            int inside_axon_arr_cnt = 0;

            for (int i = 0; i < mdl.n_neurs; i++)
            {
                show_neur_lvl[i] = mdl.neur_cor[i][2] > mdl.max_r;

                // Change coordinates from um to pixels
                float xc = nerve_cent_pix + mdl.neur_cor[i][0] * setts.resolution;
                float yc = nerve_cent_pix + mdl.neur_cor[i][1] * setts.resolution;
                float rc = mdl.neur_cor[i][2] * setts.resolution;

                axons_coor[i, 0] = xc; axons_coor[i, 1] = yc; axons_coor[i, 2] = rc;

                live_neur[i] = true;
                death_itr[i] = 0;

                axons_cent_pix[i] = (uint)xc * im_size + (uint)yc;

                neur_lbl[i] = new neur_lbl_class { lbl = "", x = xc, y = yc };

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
                            axons_inside_pix[inside_axon_arr_cnt] = (uint)x * im_size + (uint)y;
                            axons_inside_pix_idx[i + 1]++;
                            inside_axon_arr_cnt++;

                            occupied[x, y] = true;
                            //tox[x, y] = 1F;

                            tox_prod[x, y] = k_tox_prod;
                            detox[x, y] = k_detox_intra;
                        }
                        else // outside axon
                            detox[x, y] = k_detox_extra;
                    }
                // Verify radius
                // Debug.WriteLine("{0} vs {1}", (Math.Pow(mdl.neur_cor[i][2] * setts.resolution, 2) * Math.PI).ToString("0.0"), axons_inside_pix_idx[i + 1] - axons_inside_pix_idx[i]);
            }

            alg_prof.time(3);

            pix_idx = new uint[im_size * im_size];
            pix_idx_num = 0;

            // ================== Assign Rates
            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                {
                    if (within_circle2_int(x, y, nerve_r_pix) < 0 || within_circle2_int(x, y, vein_r_pix) > 0)
                    { // Outside nerve area
                        for (int k = 0; k < 4; k++)
                            rate[x, y, k] = 0;
                    }
                    else
                    {  // Inside nerve area
                        pix_idx[pix_idx_num++] += (uint)x * im_size + (uint)y;
                        int[,] arr = new int[,] { { x + 1, y }, { x - 1, y }, { x, y + 1 }, { x, y - 1 } };

                        if (occupied[x, y])
                            for (int k = 0; k < 4; k++)
                                rate[x, y, k] = occupied[arr[k, 0], arr[k, 1]] ? k_rate_live_axon : k_rate_boundary;
                        else
                            for (int k = 0; k < 4; k++)
                                rate[x, y, k] = occupied[arr[k, 0], arr[k, 1]] ? k_rate_boundary : k_rate_extra;
                    }
                }


            areal_progress_lim = 0;
            int temp = 0;

            // Keep backup of inital state
            tox_init = (float[,])tox.Clone();
            rate_init = (float[,,])rate.Clone();
            detox_init = (float[,])detox.Clone();
            tox_prod_init = (float[,])tox_prod.Clone();

            /* SCREW_GUI
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
