using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.Diagnostics;
using System.Drawing.Design;
using System.Windows.Forms;
using System.Windows.Forms.Design;
using System.Drawing.Drawing2D;
using System.Runtime.InteropServices;


namespace CustomControls
{
    public partial class CueTextBox : TextBox
    {
        [Localizable(true)]
        public string Cue
        {
            get { return mCue; }
            set { mCue = value; updateCue(); }
        }

        private void updateCue()
        {
            if (this.IsHandleCreated && mCue != null)
            {
                SendMessage(this.Handle, 0x1501, (IntPtr)1, mCue);
            }
        }
        protected override void OnHandleCreated(EventArgs e)
        {
            base.OnHandleCreated(e);
            updateCue();
        }
        private string mCue;

        // PInvoke
        [DllImport("user32.dll", CharSet = CharSet.Unicode)]
        private static extern IntPtr SendMessage(IntPtr hWnd, int msg, IntPtr wp, string lp);
    }

    public class PictureBoxWithInterpolationMode : PictureBox
    {
        public InterpolationMode InterpolationMode { get; set; }

        protected override void OnPaint(PaintEventArgs paintEventArgs)
        {
            InterpolationMode = InterpolationMode.NearestNeighbor;
            paintEventArgs.Graphics.InterpolationMode = InterpolationMode;
            base.OnPaint(paintEventArgs);
        }
    }
}
