//
//  file_interface.cs
//  LHON-Form
//
//  Created by Pooya Merat in 2016.
//

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
using System.Runtime.Serialization.Formatters.Binary;

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
                //gpu.Launch(blocks_per_grid_pix, threads_per_block).gpu_progress_image_1(tox_dev, locked_pix_dev, progression_image_sum_float_dev, progress_image_num_averaged_pix_dev, resolution_reduction_ratio);
                //gpu.Launch(new dim3(prog_im_siz, prog_im_siz), 1).gpu_progress_image_2(tox_dev, locked_pix_dev, progression_image_sum_float_dev, progress_image_num_averaged_pix_dev, progression_image_dev, prog_im_siz);

                byte[,] progression_image = new byte[prog_im_siz, prog_im_siz];
                gpu.CopyFromDevice(progression_image_dev, progression_image);
                gpu.Synchronize();

                for (int i = 0; i < prog_im_siz; i++)
                    for (int j = 0; j < prog_im_siz; j++)
                        dest[frame, i, j] = progression_image[i, j];

                /* From Picture Box
                Rectangle bounds = picB.Bounds;
                var org = picB.PointToScreen(new Point(0, 0));

                using (Bitmap bitmap = new Bitmap(bounds.Width, bounds.Height))
                {
                    using (Graphics g = Graphics.FromImage(bitmap))
                    {
                        g.CopyFromScreen(org, Point.Empty, bounds.Size);
                    }
                    string pth = ProjectOutputDir + @"Recordings\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + ".jpg";
                    bitmap.Save(pth, ImageFormat.Jpeg);
                }
                */
            }
        }

        private void Save_Progress(string progression_fil_name)
        {
            float[] Dia = new float[mdl.n_axons];
            for (int a = 0; a < mdl.n_axons; a++)
                Dia[a] = mdl.axon_coor[a][2];

            var sorted = Dia.Select((x, i) => new KeyValuePair<float, int>(x, i)).OrderBy(x => x.Key).ToList();

            List<float> DiaSorted = sorted.Select(x => x.Key).ToList();
            List<int> SortIdx = sorted.Select(x => x.Value).ToList();

            float medium_large_thrs = DiaSorted[2 * mdl.n_axons / 3];
            float small_medium_thrs = DiaSorted[mdl.n_axons / 3];
            int every_itr = 50;

            int[] size_idx = new int[mdl.n_axons];
            int[] n_axons = new int[3];

            float sum_rad_alive = 0;

            for (int a = 0; a < mdl.n_axons; a++)
            {
                float r = mdl.axon_coor[a][2];
                if (r < small_medium_thrs) { size_idx[a] = 0; n_axons[0]++; }
                else if (r > medium_large_thrs) { size_idx[a] = 2; n_axons[2]++; }
                else { size_idx[a] = 1; n_axons[1]++; }
                sum_rad_alive += r;
            }

            // Death iteration
            gpu.CopyFromDevice(death_itr_dev, death_itr);

            int[,] alive = new int[iteration, 3];
            int[,] dead = new int[iteration, 3];
            float[] mean_dia_dead = new float[iteration];
            float[] mean_dia_alive = new float[iteration];

            float sum_rad_dead = 0;

            for (int i = 1; i < iteration; i++)
            {
                for (int j = 0; j < 3; j++)
                    dead[i, j] = dead[i - 1, j];

                mean_dia_alive[i] = mean_dia_alive[i - 1];
                mean_dia_dead[i] = mean_dia_dead[i - 1];

                for (int a = 0; a < mdl.n_axons; a++)
                    if (i == death_itr[a])
                    {
                        int j = size_idx[a];
                        dead[i, j]++;
                        float r = mdl.axon_coor[a][2];
                        sum_rad_alive -= r;
                        sum_rad_dead += r;
                        mean_dia_alive[i] = sum_rad_alive / (n_axons[j] - dead[i, j]);
                        mean_dia_dead[i] = sum_rad_dead / dead[i, j];
                    }

                for (int j = 0; j < 3; j++)
                    alive[i, j] = n_axons[j] - dead[i, j];
            }

            string timeStr = DateTime.Now.ToString("yyyy - MM - dd @HH - mm - ss");
            string path = ProjectOutputDir + @"Exported\" + timeStr + ".csv";
            using (StreamWriter file = new StreamWriter(path, true))
            {

                file.WriteLine("Iteration, Small_Alive, Small_Dead, Medium_Alive, Medium_Dead, Large_Alive, Large_Dead, Mean_Diameter_Alive, Mean_Diameter_Dead");

                for (int i = 1; i < iteration; i += every_itr)
                {
                    file.Write("{0}, ", i);
                    for (int j = 0; j < 3; j++)
                        file.Write("{0}, {1}, ", alive[i, j], dead[i, j]);
                    file.Write("{0}, {1}\n", mean_dia_alive[i] * 2, mean_dia_dead[i] * 2);
                }
            }
            append_stat_ln("Model exported to " + path);
            path = ProjectOutputDir + @"Exported\" + timeStr + ".txt";
            using (StreamWriter file = new StreamWriter(path, true))
            {
                file.WriteLine("Scale = {0}, Clearance = {1}", mdl.nerve_scale_ratio, mdl_clearance);
                file.WriteLine("resolution = {0}, rate_live = {1}, rate_dead = {2}, rate_bound = {3}, rate_extra = {4}", setts.resolution, setts.rate_live, setts.rate_dead, setts.rate_bound, setts.rate_extra);
                file.WriteLine("tox_prod = {0}, detox_intra = {1}, detox_extra = {2}, death_tox_thres = {3}", setts.tox_prod, setts.detox_intra, setts.detox_extra, setts.death_tox_thres);
            }
            /*
            using (FileStream fileStream = new FileStream(progression_fil_name, FileMode.Append, FileAccess.Write, FileShare.None))
            {
                using (BinaryWriter writer = new BinaryWriter(fileStream))
                {
                    // Model ID
                    Debug.WriteLine(model_id);
                    writer.Write(model_id);

                    // Setts
                    writer.Write(setts.resolution);

                    writer.Write(setts.rate_live);
                    writer.Write(setts.rate_dead);
                    writer.Write(setts.rate_bound);
                    writer.Write(setts.rate_extra);

                    writer.Write(setts.tox_prod);
                    writer.Write(setts.detox_intra);
                    writer.Write(setts.detox_extra);

                    writer.Write(setts.death_tox_thres);

                    writer.Write(insult_x);
                    writer.Write(insult_y);
                    writer.Write(insult_r);

                    writer.Write(setts.insult_tox);

                    for (int m = 0; m < mdl.n_axons; m++)
                        writer.Write(death_itr[m]);

                    writer.Flush();

                    append_stat_ln("Sim Progress saved to " + progression_fil_name);
                }
            }
            */
        }

        // ====================== Binary reader . writer

        public static void WriteToBinaryFile<T>(string filePath, T objectToWrite)
        {
            using (Stream stream = File.Create(filePath))
            {
                var binaryFormatter = new BinaryFormatter();
                binaryFormatter.Serialize(stream, objectToWrite);
            }
        }
        public static T ReadFromBinaryFile<T>(string filePath)
        {
            using (Stream stream = File.Open(filePath, FileMode.Open))
            {
                var binaryFormatter = new BinaryFormatter();
                return (T)binaryFormatter.Deserialize(stream);
            }
        }


        // ====================== Matlab Interface

        void Export_model() // no death info, text file
        {
            string path = ProjectOutputDir + @"Exported\" + DateTime.Now.ToString("yyyy - MM - dd @HH - mm - ss") + ".txt";
            using (StreamWriter file = new StreamWriter(path, true))
            {
                file.WriteLine("{0}, {1}, {2}", mdl.nerve_scale_ratio, mdl_vessel_ratio, mdl_clearance);
                for (int i = 0; i < mdl.n_axons; i++)
                    file.WriteLine("{0}, {1}, {2}", mdl.axon_coor[i][0], mdl.axon_coor[i][1], mdl.axon_coor[i][2]);
            }
            append_stat_ln("Model exported to " + path);
        }

    }
}
