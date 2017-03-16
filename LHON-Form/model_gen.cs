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
    public partial class Main_Form : Form
    {
        // =============================== Model Generation

        float sample_raduis(float mean)
        {
            double alpha = 3;
            double beta = (mean - mdl.min_r_abs) / (mdl.max_r_abs - mdl.min_r_abs) * (alpha - 1);
            //double nrm_rnd = (1 / Gamma.Sample(alpha, 1 / beta));
            float res;
            do
            {
                double samp = InverseGamma.Sample(alpha, beta);
                //double samp = 1 / Gamma.Sample(alpha, 1 / beta);
                res = (float)(mdl.min_r_abs + samp * (mdl.max_r_abs - mdl.min_r_abs));
            }
            while (res >= mdl.max_r_abs);

            return res;
        }


        float Xc = 0, Yc = 0, Rc, Rcl = 0, Rc_avg; // in length unit
        float Xcll = 0, Xcl = 0, Ycll = 0, Ycl = 0; // in length unit
        float angle = 0;
        int xc = 0, yc = 0, rc = 0; // in pixels
        int rc_clear, rc_clear2, rc2, box_x0, box_y0, box_x1, box_y1;

        float mdl_resolution;

        bool[,] mdl_occupied;

        List<float[]> mdl_neur_cor;
        int mdl_n_neurs;

        neur_lbl_class[] mdl_neur_lbl;


        // takes Xc, Yc and Rc and updates all pixel-based params
        void update_box()
        {
            xc = cor_to_pix(Xc);
            yc = cor_to_pix(Yc);
            rc = (int)(Rc * mdl_resolution);
            rc2 = rc * rc;

            rc_clear = rc + (int)(mdl.clearance * mdl_resolution);
            rc_clear2 = (rc_clear - 1) * (rc_clear - 1);
            box_x0 = Max(xc - rc_clear, 0);
            box_y0 = Max(yc - rc_clear, 0);
            box_x1 = Min(xc + rc_clear + 1, im_size);
            box_y1 = Min(yc + rc_clear + 1, im_size);
        }

        bool check_overlap()
        {
            for (int y = box_y0; y < box_y1; y++)
                for (int x = box_x0; x < box_x1; x++)
                {
                    float dx = x - xc;
                    float dy = y - yc;
                    if (rc_clear2 - (dx * dx + dy * dy) > 0)
                        if (mdl_occupied[x, y])
                            return true;
                }
            return false;
        }

        int cor_to_pix(float X)
        {
            return (int)((X + mdl.nerve_r) * mdl_resolution);
        }

        private void new_model()
        {
            // new_model is resolution independent. Numbers are length unit.
            // It gets the eye nerve raduis and min_r, max_r as inputs
            // and generates a bunch of circles as [x, y, raduis] in mdl.neur_cor

            sim_stat = sim_stat_enum.Running;

            mdl_neur_cor = new List<float[]>();
            mdl_n_neurs = 0;

            bool strict_mod = chk_strict_rad.Checked;

            // float base_mean_r = mdl.min_r + (mdl.max_r - mdl.min_r) * (float)((1.0 - 0.19) / (6.87 - 0.19));
            //double alpha = 3;
            //double denum = (mdl.max_r - mdl.min_r) / (alpha - 1);
            //Func<double, double> beta = (mean) => (mean - mdl.min_r) / denum;

            update_bottom_stat("Generating Model...");

            //double[,] statistical_mean = new double[,]
            //       {{1.04,   1.19,    1.15,    1.21,    1.28,    1.27 },
            //        { .97,    .95,    1.01,    1.16,    1.24,    1.26 },
            //        { .93,    .92,    1.07,    1.16,    1.22,    1.28 },
            //        { .94,    .93,    1.08,    1.19,    1.18,    1.17 },
            //        { .92,   1.02,    1.02,    1.14,    1.16,    1.24 },
            //        {1.01,    1.1,    1.10,    1.22,    1.22,    1.21 } };

            mdl_resolution = 20; // 8 / mdl.clearance;

            im_size = (int)(mdl.nerve_r * 2 * mdl_resolution);

            mdl_occupied = new bool[im_size, im_size];

            // Generate random coordinates and raduis
            Random random = new Random();
            Func<float, float, float> get_rand = (float m, float M) => (float)random.NextDouble() * (M - m) + m;

            float rc_slope = 1 / (mdl.nerve_r * 2) * (mdl.max_r - mdl.min_r);
            // Could be a function of Y too
            Func<float, float, float> find_average = (float X, float Y) => mdl.min_r + (X + mdl.nerve_r) * rc_slope;

            Xc = 0; Yc = 0; Xcll = 0; Xcl = 0; Ycll = 0; Ycl = 0; Rcl = 0;
            angle = 0;

            float angle_step = 0.08F;
            float distance_step = mdl.clearance / 2;

            tic();

            int num_tries = (int)(mdl.num_tries * mdl.nerve_r * mdl.nerve_r);

            for (int i = 0; i < num_tries; i++)
            {
                // =================================

                if (i % 20 == 0) update_mdl_prog((float)i / num_tries);

                Rc_avg = find_average(Xcl, Ycl);
                if (strict_mod)
                    Rc = Rc_avg;
                else
                    Rc = sample_raduis(Rc_avg);

                if (i > 0)
                {
                    float dist = Rc + Rcl + mdl.clearance;
                    Xc = Xcl + (float)Math.Cos(angle) * dist;
                    Yc = Ycl + (float)Math.Sin(angle) * dist;

                    if (i > 1)
                    {
                        bool intitial_overlap = true;
                        float Xcg = 0, Ycg = 0; // last successful coords
                        float angle_init = angle;
                        while (true)
                        {
                            Xc = Xcl + (float)Math.Cos(angle) * dist;
                            Yc = Ycl + (float)Math.Sin(angle) * dist;

                            update_box();
                            if (check_overlap())
                            {
                                if (intitial_overlap)
                                {
                                    angle -= angle_step;
                                    if (Math.Abs(angle - angle_init) >= Math.PI)
                                    {
                                        dist += distance_step;
                                        angle = angle_init;
                                    }
                                }
                                else
                                {
                                    // revert to the last non-overlapping case
                                    Xc = Xcg;
                                    Yc = Ycg;
                                    angle -= angle_step;
                                    break;
                                }
                            }
                            else
                            {
                                intitial_overlap = false;
                                Xcg = Xc;
                                Ycg = Yc;
                                angle += angle_step;
                            }
                        }
                    }
                }

                update_box();

                // Add neuron
                for (int y = box_y0; y < box_y1; y++)
                    for (int x = box_x0; x < box_x1; x++)
                    {
                        float dx = x - xc;
                        float dy = y - yc;
                        //if (rc_clear2 - (dx * dx + dy * dy) > 0)
                        if (within_circle2(x, y, xc, yc, rc) > 0)
                            mdl_occupied[x, y] = true;
                    }
                mdl_neur_cor.Add(new float[3] { Xc, Yc, Rc });
                mdl_n_neurs++;
                update_n_neur_lbl();

                Xcll = Xcl; Xcl = Xc;
                Ycll = Ycl; Ycl = Yc;
                Rcl = Rc;

            }
            mdl.neur_cor = new List<float[]>();
            mdl.n_neurs = 0;

            // REMOVE 2
            //mdl_neur_lbl = new neur_lbl_class[mdl_n_neurs];

            //int idx = 0;
            for (int i = 0; i < mdl_n_neurs; i++)
            {
                float cent_dis = (float)Math.Sqrt(mdl_neur_cor[i][0] * mdl_neur_cor[i][0] +
                    mdl_neur_cor[i][1] * mdl_neur_cor[i][1]);
                if (cent_dis + mdl_neur_cor[i][2] > mdl.nerve_r) continue;
                if (cent_dis - mdl_neur_cor[i][2] < mdl.vein_rat * mdl.nerve_r) continue;

                mdl.neur_cor.Add(mdl_neur_cor[i]);
                mdl.n_neurs++;

                // REMOVE
                //float tempx = (mdl_neur_cor[i][0] + mdl.nerve_r + nerve_clear) * setts.resolution;
                //float tempy = (mdl_neur_cor[i][1] + mdl.nerve_r + nerve_clear) * setts.resolution;
                //mdl_neur_lbl[idx] = new neur_lbl_class { lbl = (i + 1).ToString("0"), x = tempx, y = tempy };
                //idx++;
            }

            append_stat("Model Generated in " + (toc() / 1000).ToString("0.0") + " secs\n");

            Debug.WriteLine("model done");

            first_neur_idx = 0;
            preprocess_model();
            Debug.WriteLine("prep done");

            sim_stat = sim_stat_enum.None;
        }
    }
}
