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
using MathNet.Numerics.Distributions;

namespace LHON_Form
{
    public partial class Main_Form
    {
        // =============================== Model Preprocessing

        bool touch_mode_8 = false; // false: 4, true: 8

        int calc_im_siz()
        {
            return (int)(((mdl.nerve_r * setts.resolution) * 2 + 1) / threads_per_block_1D + 1) * threads_per_block_1D;
        }

        private void preprocess_model()
        {
            // Requires full Model info and assigns tox, rate, etc

            alg_prof.time(0);
            tic();

            update_bottom_stat("Preprocessing ...");

            im_size = calc_im_siz();

            update_image_siz_lbl();

            init_bmp_write();

            // ======== Image Properties =========
            rate = new float[im_size, im_size];
            detox = new float[im_size, im_size];
            locked_pix = new byte[im_size, im_size];
            tox_init = new float[im_size, im_size];
            rate_init = new float[im_size, im_size];
            detox_init = new float[im_size, im_size];
            locked_pix_init = new byte[im_size, im_size];

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
            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                {
                    if (within_circle2_int(x, y, nerve_r_pix) < 0 || within_circle2_int(x, y, vein_r_pix) > 0)
                    { rate[x, y] = 0; locked_pix[x, y]++; }
                    else rate[x, y] = 1F;
                }

            alg_prof.time(2);

            // ======== Common Neuron Properties (+Initialization) =========

            update_n_neur_lbl();

            // Assign max memory
            max_set_size_bound = (int)(2.2 * mdl.max_r_abs * setts.resolution * 3.14);
            max_set_size_bound_touch = (int)(mdl.max_r_abs * setts.resolution * 3.14 * 20);
            int max_pixels_in_nerve = (int)(Math.Pow(mdl.nerve_r * setts.resolution, 2) * Math.PI) -
                (int)(Math.Pow(mdl.nerve_r * mdl.vein_rat * setts.resolution, 2) * Math.PI);

            // ======== Individual Neuron Properties =========

            //int[,,] axons_bound_pix = new int[mdl.n_neurs, max_set_size_bound, 2];
            int[] axons_bound_npix = new int[mdl.n_neurs];
            axons_inside_pix = new ushort[max_pixels_in_nerve, 2];
            axons_inside_pix_idx = new int[mdl.n_neurs + 1];

            axons_inside_npix = new uint[mdl.n_neurs];
            axons_bound_touch_pix = new int[mdl.n_neurs, max_set_size_bound_touch, 2];
            axons_bound_touch_npix = new uint[mdl.n_neurs];

            show_neur_lvl = new bool[mdl.n_neurs];

            live_neur = new bool[mdl.n_neurs];
            death_itr = new uint[mdl.n_neurs];
            neur_tol = new float[mdl.n_neurs];
            tox_touch_neur = new float[mdl.n_neurs];
            tox_touch_neur_last = new float[mdl.n_neurs];

            neur_lbl = new neur_lbl_class[mdl.n_neurs];


            // ======== Individual Neuron Properties Initialization =========

            axons_coor = new float[mdl.n_neurs, 3];

            int inside_neur_arr_cnt = 0;

            //try
            {
                for (int i = 0; i < mdl.n_neurs; i++)
                {
                    show_neur_lvl[i] = mdl.neur_cor[i][2] > mdl.max_r;

                    float detox_resolution = detox_val / (float)Math.Pow(setts.resolution, 2);
                    
                    float xc = nerve_cent_pix + mdl.neur_cor[i][0] * setts.resolution;
                    float yc = nerve_cent_pix + mdl.neur_cor[i][1] * setts.resolution;
                    float rc = mdl.neur_cor[i][2] * setts.resolution;

                    axons_coor[i, 0] = xc; axons_coor[i, 1] = yc; axons_coor[i, 2] = rc;

                    live_neur[i] = true;
                    death_itr[i] = 0;

                    neur_tol[i] = mdl.neur_cor[i][2] * mdl.neur_cor[i][2] * setts.neur_tol_coeff * setts.resolution;

                    neur_lbl[i] = new neur_lbl_class { lbl = "", x = xc, y = yc };

                    axons_inside_pix_idx[i + 1] = axons_inside_pix_idx[i];

                    float extra_radius;
                    if (!touch_mode_8)
                        extra_radius = 1; // to find touch
                    else
                        extra_radius = 2; // to find touch


                    float extra_box_siz = 2 * extra_radius;
                    Func<float, float> extra_rad = (rad) => (2 * extra_radius * rad + extra_radius * extra_radius);

                    float extra_siz = rc + extra_box_siz;

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
                            bool on_bound = dst(x, y) <= 0 && dst(x, y) > -extra_rad(rc);

                            if (inside)
                            {
                                axons_inside_npix[i]++;

                                axons_inside_pix[inside_neur_arr_cnt, 0] = (ushort)x;
                                axons_inside_pix[inside_neur_arr_cnt, 1] = (ushort)y;
                                axons_inside_pix_idx[i + 1]++;
                                inside_neur_arr_cnt++;

                                locked_pix[x, y]++;
                                tox[x, y] = 1;

                                rate[x, y] = setts.neur_rate;
                            }
                            if (on_bound) // if it's a cell boundary
                            {
                                //axons_bound_pix[i, axons_bound_npix[i], 0] = x;
                                //axons_bound_pix[i, axons_bound_npix[i]++, 1] = y;

                                int[,] arr;
                                if (!touch_mode_8)
                                    arr = new int[,] { { x + 1, y }, { x - 1, y }, { x, y + 1 }, { x, y - 1 } };
                                else
                                    arr = new int[,] { { x + 1, y }, { x + 1, y + 1 }, { x + 1, y - 1 }, { x - 1, y }, { x - 1, y + 1 }, { x - 1, y - 1 }, { x, y + 1 }, { x, y - 1 } };
                                
                                for (int k = 0; k < arr.GetLength(0); k++)
                                {
                                    if (dst(arr[k, 0], arr[k, 1]) > 0)
                                    {
                                        axons_bound_touch_pix[i, axons_bound_touch_npix[i], 0] = x;
                                        axons_bound_touch_pix[i, axons_bound_touch_npix[i]++, 1] = y;
                                        touch_pix[x, y] = 1;
                                        detox[x, y] = detox_resolution;
                                    }
                                }
                            }
                        }
                    // Verify radius
                    //Debug.WriteLine("{0} vs {1}", (Math.Pow(mdl.neur_cor[i][2] * setts.resolution, 2) * Math.PI).ToString("0.0"), axons_inside_pix_idx[i + 1] - axons_inside_pix_idx[i]);
                }

                alg_prof.time(3);
            }

            //catch (Exception e)
            //{
            //    MessageBox.Show(e.ToString() + "\nChange Settings!");
            //    setts.Show();
            //    return;
            //}

            areal_progress_lim = 0;
            int temp = 0;

            // Keep back up of inital state
            tox_init = (float[,])tox.Clone();
            rate_init = (float[,])rate.Clone();
            detox_init = (float[,])detox.Clone();
            locked_pix_init = (byte[,])locked_pix.Clone();

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

            Debug.WriteLine("inside: {0} vs allocated {1}", inside_neur_arr_cnt, axons_inside_pix.Length / 2);
            Debug.WriteLine("bound: {0} vs allocated {1}", axons_bound_npix.Max(), max_set_size_bound);
            Debug.WriteLine("bound-touch: {0} vs allocated {1}", axons_bound_touch_npix.Max(), max_set_size_bound_touch);


            alg_prof.report();

        }

    }
}
