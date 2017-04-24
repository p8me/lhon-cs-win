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

                    // Death iteration
                    gpu.CopyFromDevice(death_itr_dev, death_itr);

                    for (int m = 0; m < mdl.n_axons; m++)
                        writer.Write(death_itr[m]);
                    
                    writer.Flush();

                    append_stat_ln("Sim Progress saved to " + progression_fil_name);
                }
            }
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
