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
        ushort bmp_im_size = 512; // will be rounded to a multiple of threads_per_block_bmp

        int threads_per_block_bmp_1D = 32;

        Bitmap bmp;
        IntPtr bmp_scan0;
        byte[,,] bmp_bytes;

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
                    bmp_bytes_dev, tox_dev, axon_mask_dev, death_tox_lim, show_opts_dev);

                gpu.CopyFromDevice(bmp_bytes_dev, bmp_bytes);
                gpu.CopyFromDevice(tox_dev, tox);

                fixed (byte* dat = &bmp_bytes[0, 0, 0])
                    CopyMemory(bmp_scan0, (IntPtr)dat, (uint)bmp_bytes.Length);
                picB.Image = bmp;

                if (sim_stat == sim_stat_enum.Running && chk_rec_avi.Checked)
                {
                    // AVI:
                    //aviStream.AddFrame((Bitmap)bmp.Clone());
                    // GIF:
                    var bmh = bmp.GetHbitmap();
                    var opts = BitmapSizeOptions.FromWidthAndHeight(200, 200);
                    var src = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(bmh, IntPtr.Zero, System.Windows.Int32Rect.Empty, opts);
                    gifEnc.Frames.Add(BitmapFrame.Create(src));
                }
            }
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

        private void picB_Paint(object sender, PaintEventArgs e)
        {

            if (!show_axon_order_mdl_gen && axon_lbl != null)
            {
                // the X on the first axon
                //var nlbl0 = axon_lbl[first_axon_idx];
                //SizeF textSize0 = e.Graphics.MeasureString(nlbl0.lbl, this.Font);
                //e.Graphics.DrawString(nlbl0.lbl, this.Font, Brushes.Beige, nlbl0.x * picB_ratio + picB_offx - (textSize0.Width / 2), nlbl0.y * picB_ratio + picB_offy - (textSize0.Height / 2));

                if (chk_axons_tox_lvl.Checked)
                    for (int i = 0; i < mdl.n_axons; i++)
                    {
                        var nlbl = axon_lbl[i];
                        if (axon_is_large[i] && i != first_axon_idx && nlbl.lbl.Length > 0)
                        {
                            SizeF textSize = e.Graphics.MeasureString(nlbl.lbl, this.Font);
                            e.Graphics.DrawString(nlbl.lbl, this.Font, Brushes.Red, nlbl.x * picB_ratio + picB_offx - (textSize.Width / 2), nlbl.y * picB_ratio + picB_offy - (textSize.Height / 2));
                        }
                    }
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

        private void picB_Click(object sender, EventArgs e)
        {
            //var mouseEventArgs = e as MouseEventArgs;
            //if (mouseEventArgs != null)
            //{
            //    int x = (int)((mouseEventArgs.X - picB_offx) / picB_ratio);
            //    int y = (int)((mouseEventArgs.Y - picB_offy) / picB_ratio);
            //    if (x >= 0 && x < im_size && y >= 0 && y < im_size)
            //    {
            //        if (sim_stat == sim_stat_enum.Running) return;

            //        // Sets the initial insult location
            //        init_insult[0] = (float)(x - 1) / setts.resolution - mdl_nerve_r;
            //        init_insult[1] = (float)(y - 1) / setts.resolution - mdl_nerve_r;

            //        reset_state();
            //    }
            //}
        }


    }
}
