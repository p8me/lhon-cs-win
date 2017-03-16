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

        private void preprocess_model()
        {
            // Requires full Model info and assigns tox, rate, etc

            tic();

            update_bottom_stat("Preprocessing ...");

            im_size = (int)(((mdl.nerve_r + nerve_clear) * setts.resolution) * 2);

            update_image_siz_lbl();

            init_bmp_write();

            // ======== Image Properties =========
            rate = new float[im_size, im_size];
            locked_pix = new uint[im_size, im_size];
            tox_init = new float[im_size, im_size];
            rate_init = new float[im_size, im_size];
            locked_pix_init = new uint[im_size, im_size];
            bool[,] cant_be_touch_pix = new bool[im_size, im_size];

            // ======== Image Properties Initialization =========
            int nerve_cent = im_size / 2;
            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                {
                    rate[x, y] = 1F;
                    if (within_circle2(x, y, nerve_cent, nerve_cent, (int)(mdl.nerve_r * setts.resolution)) < 0
                        || within_circle2(x, y, nerve_cent, nerve_cent, (int)(mdl.vein_rat * mdl.nerve_r * setts.resolution)) > 0)
                    { rate[x, y] = 0; locked_pix[x, y]++; }
                }

            // ======== Common Neuron Properties (+Initialization) =========

            update_n_neur_lbl();

            // Assign max memory
            max_set_size_bound = (int)(2.2 * mdl.max_r_abs * setts.resolution * 3.14);
            max_set_size_bound_touch = (int)(2.8 * mdl.max_r_abs * setts.resolution * 3.14);
            int max_pixels_in_nerve = (int)(Math.Pow(mdl.nerve_r * setts.resolution, 2) * Math.PI) -
                (int)(Math.Pow(mdl.nerve_r * mdl.vein_rat * setts.resolution, 2) * Math.PI);

            // ======== Individual Neuron Properties =========

            int[,,] neurs_bound_pix = new int[mdl.n_neurs, max_set_size_bound, 2];
            int[] neurs_bound_npix = new int[mdl.n_neurs];
            neurs_inside_pix = new ushort[max_pixels_in_nerve, 2];
            neurs_inside_pix_idx = new int[mdl.n_neurs + 1];

            neurs_inside_npix = new uint[mdl.n_neurs];
            neurs_bound_touch_pix = new int[mdl.n_neurs, max_set_size_bound_touch, 2];
            neurs_bound_touch_npix = new uint[mdl.n_neurs];

            show_neur_lvl = new bool[mdl.n_neurs];

            live_neur = new bool[mdl.n_neurs];
            death_itr = new uint[mdl.n_neurs];
            neur_tol = new float[mdl.n_neurs];
            tox_touch_neur = new float[mdl.n_neurs];
            tox_touch_neur_last = new float[mdl.n_neurs];

            neur_lbl = new neur_lbl_class[mdl.n_neurs];


            // ======== Individual Neuron Properties Initialization =========

            neurs_coor = new float[mdl.n_neurs, 3];

            int inside_neur_arr_cnt = 0;

            //try
            {
                for (int i = 0; i < mdl.n_neurs; i++)
                {
                    show_neur_lvl[i] = mdl.neur_cor[i][2] > mdl.max_r;

                    float xc = ((mdl.neur_cor[i][0] + mdl.nerve_r + nerve_clear) * setts.resolution);
                    float yc = ((mdl.neur_cor[i][1] + mdl.nerve_r + nerve_clear) * setts.resolution);
                    float rc = (mdl.neur_cor[i][2] * setts.resolution);

                    neurs_coor[i, 0] = xc; neurs_coor[i, 1] = yc; neurs_coor[i, 2] = rc;

                    live_neur[i] = true;
                    death_itr[i] = 0;

                    neur_tol[i] = mdl.neur_cor[i][2] * mdl.neur_cor[i][2] * setts.neur_tol_coeff * setts.resolution;

                    neur_lbl[i] = new neur_lbl_class { lbl = "", x = xc, y = yc };

                    neurs_inside_pix_idx[i + 1] = neurs_inside_pix_idx[i];

                    for (int y = Max((int)(yc - rc) - 1, 0); y <= yc + rc + 1 && y < im_size; y++)
                        for (int x = Max((int)(xc - rc) - 1, 0); x <= xc + rc + 1 && x < im_size; x++)
                        {
                            float wc = within_circle2(x, y, xc, yc, rc);
                            bool inside = wc > 0;
                            bool on_bound = Math.Abs(wc) <= rc;

                            if (inside || on_bound)
                            {
                                neurs_inside_npix[i]++;

                                neurs_inside_pix[inside_neur_arr_cnt, 0] = (ushort)x;
                                neurs_inside_pix[inside_neur_arr_cnt, 1] = (ushort)y;
                                neurs_inside_pix_idx[i + 1]++;
                                inside_neur_arr_cnt++;

                                locked_pix[x, y]++;
                                tox[x, y] = 1;
                                cant_be_touch_pix[x, y] = true;

                                if (on_bound) // if it's a cell boundary
                                {
                                    neurs_bound_pix[i, neurs_bound_npix[i], 0] = x;
                                    neurs_bound_pix[i, neurs_bound_npix[i]++, 1] = y;
                                }
                                else
                                    rate[x, y] = setts.neur_rate;
                            }
                        }
                }

                for (int i = 0; i < mdl.n_neurs; i++)
                {
                    for (int m = 0; m < neurs_bound_npix[i]; m++)
                    {
                        int x = neurs_bound_pix[i, m, 0];
                        int y = neurs_bound_pix[i, m, 1];

                        int[,] arr = new int[4, 2] { { x + 1, y }, { x - 1, y }, { x, y + 1 }, { x, y - 1 } };

                        for (int k = 0; k < 4; k++)
                        {
                            if (within_circle2(arr[k, 0], arr[k, 1],
                                neurs_coor[i, 0], neurs_coor[i, 1], neurs_coor[i, 2]) < 0)
                            {
                                if (!cant_be_touch_pix[arr[k, 0], arr[k, 1]])
                                {
                                    neurs_bound_touch_pix[i, neurs_bound_touch_npix[i], 0] = arr[k, 0];
                                    neurs_bound_touch_pix[i, neurs_bound_touch_npix[i]++, 1] = arr[k, 1];
                                    touch_pix[arr[k, 0], arr[k, 1]] = 1;
                                }
                            }
                        }
                    }
                }
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
            for (int y = 0; y < im_size; y++)
                for (int x = 0; x < im_size; x++)
                {
                    tox_init[x, y] = tox[x, y];
                    rate_init[x, y] = rate[x, y];
                    locked_pix_init[x, y] = locked_pix[x, y];
                    if (within_circle2(x, y, nerve_cent, nerve_cent, (int)(mdl.nerve_r * setts.resolution)) > 0)
                    {
                        temp++;
                        if (tox[x, y] > 0) areal_progress_lim++;
                    }
                }

            areal_progress_lim = areal_progress_lim / temp * 0.7F;

            reset_state();

            update_bottom_stat("Preprocess Done! (" + (toc() / 1000).ToString("0.0") + " secs)");

            Debug.WriteLine("inside: {0} vs allocated {1}", inside_neur_arr_cnt, neurs_inside_pix.Length / 2);
            Debug.WriteLine("bound: {0} vs allocated {1}", neurs_bound_npix.Max(), max_set_size_bound);
            Debug.WriteLine("bound-touch: {0} vs allocated {1}", neurs_bound_touch_npix.Max(), max_set_size_bound_touch);
        }

    }
}
