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
using System.Windows.Media.Imaging;

namespace LHON_Form
{
    public partial class Main_Form : Form
    {
        ushort bmp_im_size = 1024; // will be rounded to a multiple of threads_per_block_bmp

        int threads_per_block_bmp_1D = 32;

        Bitmap bmp;
        IntPtr bmp_scan0;
        byte[,,] bmp_bytes, bmp_bytes_dev;

        byte[,] init_insult_mask_dev;

        float bmp_image_compression_ratio;

        bool[] show_opts = new bool[2],
            show_opts_dev = new bool[2];

        int blocks_per_grid_bmp;

        dim3 update_bmp_gride_size_2D, update_bmp_block_size_2D;

        void init_bmp_write()
        {
            bmp_im_size = (ushort)((bmp_im_size / threads_per_block_bmp_1D) * threads_per_block_bmp_1D);
            bmp_image_compression_ratio = (float)im_size / (float)bmp_im_size;
            blocks_per_grid_bmp = bmp_im_size / threads_per_block_bmp_1D;

            update_bmp_gride_size_2D = new dim3(blocks_per_grid_bmp, blocks_per_grid_bmp);
            update_bmp_block_size_2D = new dim3(threads_per_block_bmp_1D, threads_per_block_bmp_1D);

            bmp = new Bitmap(bmp_im_size, bmp_im_size);
            Rectangle rect = new Rectangle(0, 0, bmp_im_size, bmp_im_size);
            BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadWrite, bmp.PixelFormat);
            bmp_scan0 = bmpData.Scan0;
            bmp.UnlockBits(bmpData);
            bmp_bytes = new byte[bmp_im_size, bmp_im_size, 4];

            for (int y = 0; y < bmp_im_size; y++)
                for (int x = 0; x < bmp_im_size; x++)
                    bmp_bytes[y, x, 3] = 255;
        }

        [DllImport("kernel32.dll", EntryPoint = "CopyMemory", SetLastError = false)]
        public static extern void CopyMemory(IntPtr dest, IntPtr src, uint count);
        unsafe private void update_bmp_image()
        {
            if (InvokeRequired)
                Invoke(new Action(() => update_bmp_image()));
            else
            {
                gpu.Launch(update_bmp_gride_size_2D, update_bmp_block_size_2D).cuda_update_image(im_size, bmp_im_size, bmp_image_compression_ratio,
                    bmp_bytes_dev, tox_dev, axon_mask_dev, init_insult_mask_dev, death_tox_thres, show_opts_dev);

                gpu.CopyFromDevice(bmp_bytes_dev, bmp_bytes);

                fixed (byte* dat = &bmp_bytes[0, 0, 0])
                    CopyMemory(bmp_scan0, (IntPtr)dat, (uint)bmp_bytes.Length);
                picB.Image = bmp;

            }
        }

        void record_bmp_gif()
        {
            if (InvokeRequired)
                Invoke(new Action(() => record_bmp_gif()));
            else
            {
                // AVI:
                //aviStream.AddFrame((Bitmap)bmp.Clone());
                // GIF:
                var bmh = bmp.GetHbitmap();
                var opts = BitmapSizeOptions.FromWidthAndHeight(bmp_im_size, bmp_im_size);
                var src = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(bmh, IntPtr.Zero, System.Windows.Int32Rect.Empty, opts);
                gifEnc.Frames.Add(BitmapFrame.Create(src));
            }
        }

        float insult_x, insult_y, insult_r; // in um

        void update_init_insult()
        {
            int insult_x_p = bmp_im_size - ((int)(insult_y * setts.resolution / bmp_image_compression_ratio) + bmp_im_size / 2);
            int insult_y_p = (int)(insult_x * setts.resolution / bmp_image_compression_ratio) + bmp_im_size / 2;
            int insult_r2_p = (int)(pow2(insult_r * setts.resolution / bmp_image_compression_ratio));

            gpu.Launch(update_bmp_gride_size_2D, update_bmp_block_size_2D).cuda_update_init_insult(
                bmp_im_size, insult_x_p, insult_y_p, insult_r2_p, init_insult_mask_dev);
        }

        int picB_offx, picB_offy;
        float picB_ratio;

        private void picB_Resize(object sender, EventArgs e)
        {
            float picW = picB.Size.Width;
            float picH = picB.Size.Height;

            float asp_im = (float)im_size / (float)im_size,
                asp_box = picW / picH;

            if (asp_im > asp_box)
            {
                picB_ratio = picW / im_size;
                picB_offx = 0;
                picB_offy = (int)((picH - picB_ratio * (float)im_size) / 2f);
            }
            else
            {
                picB_ratio = picH / im_size;
                picB_offx = (int)((picW - picB_ratio * (float)im_size) / 2f);
                picB_offy = 0;
            }
        }

        float[] get_mouse_click_um(MouseEventArgs e)
        {
            float[] um = new float[2];
            int x = (int)((e.X - picB_offx) / picB_ratio);
            int y = (int)((e.Y - picB_offy) / picB_ratio);
            if (x >= 0 && x < im_size && y >= 0 && y < im_size)
            {
                // Sets the initial insult location
                um[1] = (float)(im_size - y - 1) / setts.resolution - mdl_nerve_r;
                um[0] = (float)(x - 1) / setts.resolution - mdl_nerve_r;
            }
            return um;
        }

        void mouse_click(MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right || sim_stat == sim_stat_enum.Running) return;

            // Sets the initial insult location
            float[] um = get_mouse_click_um(e);
            insult_x = um[0];
            insult_y = um[1];

            Debug.WriteLine(insult_x + "  " + insult_y);

            reset_state();
        }

        // picB.Pain += (s, e) => {...}
        private void picB_Paint(object sender, PaintEventArgs e)
        {
            if (!show_axon_order_mdl_gen && axon_lbl != null)
            {
                // the X on the first axon
                //var nlbl0 = axon_lbl[first_axon_idx];
                //SizeF textSize0 = e.Graphics.MeasureString(nlbl0.lbl, this.Font);
                //e.Graphics.DrawString(nlbl0.lbl, this.Font, Brushes.Beige, nlbl0.x * picB_ratio + picB_offx - (textSize0.Width / 2), nlbl0.y * picB_ratio + picB_offy - (textSize0.Height / 2));
            }

            if (show_axon_order_mdl_gen)
            {
                if (mdl_axon_lbl != null && mdl_axon_lbl.Length > 0)
                    for (int i = 0; i < mdl_n_axons; i++)
                    {
                        var lbli = mdl_axon_lbl[i];
                        if (lbli != null)
                        {
                            SizeF textSize = e.Graphics.MeasureString(lbli.lbl, this.Font);
                            float x = lbli.x * picB_ratio + picB_offx - (textSize.Width / 2);
                            float y = lbli.y * picB_ratio + picB_offy - (textSize.Height / 2);
                            e.Graphics.DrawString(lbli.lbl, this.Font, Brushes.White, x, y);
                        }
                    }
            }
        }

    }
}
