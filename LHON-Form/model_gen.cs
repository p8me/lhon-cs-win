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

        string axon_radius_cdf_file = @"../../../../Matlab\pdf_from_graph\axon_radius_cdf.tbl";

        bool show_axon_order_mdl_gen = false;

        // =============================== Model Generation

        void read_model_cdf_file()
        {
            if (File.Exists(axon_radius_cdf_file))
            {
                using (BinaryReader reader = new BinaryReader(File.Open(axon_radius_cdf_file, FileMode.Open)))
                {
                    UInt32 num_diameters = reader.ReadUInt32();
                    UInt32 num_cum_probabilities = reader.ReadUInt32() - 1;
                    float[,] cdf_value = new float[num_diameters, num_cum_probabilities];
                    float[] cdf_axon_radius = new float[num_diameters];
                    
                    for (int r = 0; r < num_diameters; r++)
                        cdf_axon_radius[r] = reader.ReadSingle();
                    for (int r = 0; r < num_diameters; r++)
                        for (int c = 0; c < num_cum_probabilities; c++)
                            cdf_value[r, c] = reader.ReadSingle();
                }
            }
            else
                append_stat_ln("axon_radius_cdf_file could not be found.");
        }

        float sample_radius(float mean)
        {
            double alpha = 3;
            double beta = (mean - mdl.min_r_abs) / (mdl.max_r_abs - mdl.min_r_abs) * (alpha - 1);
            //double nrm_rnd = (1 / Gamma.Sample(alpha, 1 / beta));
            float res;
            do
            {
                //MathNet.Numerics.Distributions.Normal
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

        List<float[]> mdl_axons_coor;
        int mdl_n_axons;

        axon_lbl_class[] mdl_axon_lbl;


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
                    if (rc_clear2 - (dx * dx + dy * dy) >= 0)
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
            // and generates a bunch of circles as [x, y, raduis] in mdl.axon_coor

            sim_stat = sim_stat_enum.Running;

            mdl_axons_coor = new List<float[]>();
            mdl_n_axons = 0;

            bool strict_mod = chk_strict_rad.Checked;

            // float base_mean_r = mdl.min_r + (mdl.max_r - mdl.min_r) * (float)((1.0 - 0.19) / (6.87 - 0.19));
            //double alpha = 3;
            //double denum = (mdl.max_r - mdl.min_r) / (alpha - 1);
            //Func<double, double> beta = (mean) => (mean - mdl.min_r) / denum;

            update_bottom_stat("Generating Model...");
            
            mdl_resolution = 25; // 8 / mdl.clearance;

            im_size = (ushort)(mdl.nerve_r * 2 * mdl_resolution);

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

            int num_tries = (int)(mdl.circ_gen_ratio * mdl.nerve_r * mdl.nerve_r);

            for (int i = 0; i < num_tries; i++)
            {
                // =================================

                if (i % 20 == 0) update_mdl_prog((float)i / num_tries);

                Rc_avg = find_average(Xcl, Ycl);
                if (strict_mod)
                    Rc = Rc_avg;
                else
                    Rc = sample_radius(Rc_avg);

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
                        if (within_circle2(x, y, xc, yc, rc) > 0)
                            mdl_occupied[x, y] = true;

                mdl_axons_coor.Add(new float[3] { Xc, Yc, Rc });
                mdl_n_axons++;
                update_num_axons_lbl();

                Xcll = Xcl; Xcl = Xc;
                Ycll = Ycl; Ycl = Yc;
                Rcl = Rc;

            }
            mdl.axon_coor = new List<float[]>();
            mdl.n_axons = 0;

            if (show_axon_order_mdl_gen)
                mdl_axon_lbl = new axon_lbl_class[mdl_n_axons];

            int idx = 0;
            for (int i = 0; i < mdl_n_axons; i++)
            {
                float cent_dis = (float)Math.Sqrt(mdl_axons_coor[i][0] * mdl_axons_coor[i][0] +
                    mdl_axons_coor[i][1] * mdl_axons_coor[i][1]);
                if (cent_dis + mdl_axons_coor[i][2] > mdl.nerve_r) continue;
                if (cent_dis - mdl_axons_coor[i][2] < mdl.vessel_rat * mdl.nerve_r) continue;

                mdl.axon_coor.Add(mdl_axons_coor[i]);
                mdl.n_axons++;

                int tmp_im_siz = im_size = calc_im_siz();

                if (show_axon_order_mdl_gen)
                {
                    float tempx = mdl_axons_coor[i][0] * setts.resolution + tmp_im_siz / 2;
                    float tempy = mdl_axons_coor[i][1] * setts.resolution + tmp_im_siz / 2;
                    mdl_axon_lbl[idx] = new axon_lbl_class { lbl = (i + 1).ToString("0"), x = tempx, y = tempy };
                    idx++;
                }
            }

            append_stat("Model Generated in " + (toc() / 1000).ToString("0.0") + " secs\n");

            Debug.WriteLine("model done");

            first_axon_idx = 0;
            preprocess_model();

            sim_stat = sim_stat_enum.None;
        }
    }
}
