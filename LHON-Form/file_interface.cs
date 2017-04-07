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
        // ============= Save Progress Image

        private void Take_Progress_Snapshot(byte[,,] dest, uint frame)
        {
            if (InvokeRequired)
                Invoke(new Action(() => Take_Progress_Snapshot(dest, frame)));
            else
            {
                gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);

                gpu.Set(progression_image_sum_float_dev);
                gpu.Set(progress_image_num_averaged_pix_dev);
                //gpu.Launch(blocks_per_grid, threads_per_block).gpu_progress_image_1(tox_dev, locked_pix_dev, progression_image_sum_float_dev, progress_image_num_averaged_pix_dev, resolution_reduction_ratio);
                //gpu.Launch(new dim3(prog_im_siz, prog_im_siz), 1).gpu_progress_image_2(tox_dev, locked_pix_dev, progression_image_sum_float_dev, progress_image_num_averaged_pix_dev, progression_image_dev, prog_im_siz);

                byte[,] progression_image = new byte[prog_im_siz, prog_im_siz];
                gpu.CopyFromDevice(progression_image_dev, progression_image);
                gpu.Synchronize();

                for (int i = 0; i < prog_im_siz; i++)
                    for (int j = 0; j < prog_im_siz; j++)
                        dest[frame, i, j] = progression_image[i, j];

                //Rectangle bounds = picB.Bounds;
                //var org = picB.PointToScreen(new Point(0, 0));

                //using (Bitmap bitmap = new Bitmap(bounds.Width, bounds.Height))
                //{
                //    using (Graphics g = Graphics.FromImage(bitmap))
                //    {
                //        g.CopyFromScreen(org, Point.Empty, bounds.Size);
                //    }
                //    string pth = ProjectOutputDir + @"Recordings\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + ".jpg";
                //    bitmap.Save(pth, ImageFormat.Jpeg);
                //}
            }
        }

        private void Save_Progress(string progression_fil_name)
        {
            using (FileStream fileStream = new FileStream(progression_fil_name, FileMode.Append, FileAccess.Write, FileShare.None))
            {
                using (BinaryWriter writer = new BinaryWriter(fileStream))
                {
                    writer.Write(mdl.nerve_r);
                    writer.Write(mdl.max_r);
                    writer.Write(mdl.min_r);
                    writer.Write(mdl.clearance);
                    writer.Write(mdl.n_neurs);
                    writer.Write(setts.resolution);
                    writer.Write(im_size);
                    writer.Write(prog_im_siz);

                    writer.Write(init_insult[0]);
                    writer.Write(init_insult[1]);
                    writer.Write(progress_num_frames);
                    writer.Write(tt_sim.read());
                    writer.Write(last_itr);

                    for (int m = 0; m < progress_num_frames; m++)
                        writer.Write(areal_progress_chron_val[m]);

                    for (int m = 0; m < progress_num_frames; m++)
                        writer.Write(chron_progress_areal_val[m]);

                    gpu.CopyFromDevice(death_itr_dev, death_itr);

                    for (int m = 0; m < mdl.n_neurs; m++)
                    {
                        float x = (mdl.neur_cor[m][0] / mdl.nerve_r / 2 + 0.5F) * 256F;
                        writer.Write((byte)(x));
                        float y = (mdl.neur_cor[m][1] / mdl.nerve_r / 2 + 0.5F) * 256F;
                        writer.Write((byte)(y));
                        writer.Write((byte)(mdl.neur_cor[m][2] * 40));
                        float r = (float)death_itr[m] / (float)last_itr * 256F;
                        writer.Write((byte)(r));
                    }

                    for (int m = 0; m < progress_num_frames; m++)
                        for (int i = 0; i < prog_im_siz; i++)
                            for (int j = 0; j < prog_im_siz; j++)
                                writer.Write(areal_progression_image_stack[m, i, j]);

                    for (int m = 0; m < progress_num_frames; m++)
                        for (int i = 0; i < prog_im_siz; i++)
                            for (int j = 0; j < prog_im_siz; j++)
                                writer.Write(chron_progression_image_stack[m, i, j]);

                    writer.Flush();

                    //append_stat_ln("Sim Progress saved to " + progression_fil_name);

                }
            }
        }


        // ====================== Matlab Interface

        void Export_model() // no death info, text file
        {
            string path = ProjectOutputDir + @"Exported\" + DateTime.Now.ToString("yyyy - MM - dd @HH - mm - ss") + ".txt";
            using (StreamWriter file = new StreamWriter(path, true))
            {
                file.WriteLine("{0}, {1}, {2}", mdl.nerve_r, mdl.vein_rat, mdl.clearance);
                for (int i = 0; i < mdl.n_neurs; i++)
                    file.WriteLine("{0}, {1}, {2}", mdl.neur_cor[i][0], mdl.neur_cor[i][1], mdl.neur_cor[i][2]);
            }
            append_stat_ln("Model exported to " + path);
        }

    }
}
