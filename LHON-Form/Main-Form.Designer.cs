﻿using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace LHON_Form
{
    partial class Main_Form
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        private class PictureBoxWithInterpolationMode : PictureBox
        {
            public InterpolationMode InterpolationMode { get; set; }

            protected override void OnPaint(PaintEventArgs paintEventArgs)
            {
                paintEventArgs.Graphics.InterpolationMode = InterpolationMode;
                base.OnPaint(paintEventArgs);
            }
        }

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Main_Form));
            this.picB = new LHON_Form.Main_Form.PictureBoxWithInterpolationMode();
            this.btn_start = new System.Windows.Forms.Button();
            this.btn_reset = new System.Windows.Forms.Button();
            this.btn_save_model = new System.Windows.Forms.Button();
            this.btn_load_model = new System.Windows.Forms.Button();
            this.btn_redraw = new System.Windows.Forms.Button();
            this.chk_neur_lvl = new System.Windows.Forms.CheckBox();
            this.label4 = new System.Windows.Forms.Label();
            this.lbl_n_Neurs = new System.Windows.Forms.Label();
            this.chk_show_bound = new System.Windows.Forms.CheckBox();
            this.chk_show_tox = new System.Windows.Forms.CheckBox();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.btn_export = new System.Windows.Forms.Button();
            this.chk_strict_rad = new System.Windows.Forms.CheckBox();
            this.lbl_mdl_prog = new System.Windows.Forms.Label();
            this.label16 = new System.Windows.Forms.Label();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.txt_clearance = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.txt_nerve_rad = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.txt_min_rad = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.txt_max_rad = new System.Windows.Forms.TextBox();
            this.txt_vein_rad = new System.Windows.Forms.TextBox();
            this.label15 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.txt_Tol = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.txt_neur_rate = new System.Windows.Forms.TextBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.label12 = new System.Windows.Forms.Label();
            this.lbl_image_siz = new System.Windows.Forms.Label();
            this.btn_preprocess = new System.Windows.Forms.Button();
            this.btn_load_setts = new System.Windows.Forms.Button();
            this.btn_save_setts = new System.Windows.Forms.Button();
            this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
            this.txt_resolution = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel4 = new System.Windows.Forms.TableLayoutPanel();
            this.chk_save_prog = new System.Windows.Forms.CheckBox();
            this.btn_sweep = new System.Windows.Forms.Button();
            this.cmb_sw_sel1 = new System.Windows.Forms.ComboBox();
            this.txt_sw_range1 = new CueTextBox();
            this.chk_save_sw_prog = new System.Windows.Forms.CheckBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel3 = new System.Windows.Forms.TableLayoutPanel();
            this.chk_rec_avi = new System.Windows.Forms.CheckBox();
            this.tableLayoutPanel9 = new System.Windows.Forms.TableLayoutPanel();
            this.tableLayoutPanel8 = new System.Windows.Forms.TableLayoutPanel();
            this.label17 = new System.Windows.Forms.Label();
            this.lbl_chron_progress = new System.Windows.Forms.Label();
            this.tableLayoutPanel10 = new System.Windows.Forms.TableLayoutPanel();
            this.label14 = new System.Windows.Forms.Label();
            this.lbl_areal_progress = new System.Windows.Forms.Label();
            this.tableLayoutPanel5 = new System.Windows.Forms.TableLayoutPanel();
            this.label1 = new System.Windows.Forms.Label();
            this.lbl_itr = new System.Windows.Forms.Label();
            this.tableLayoutPanel7 = new System.Windows.Forms.TableLayoutPanel();
            this.label3 = new System.Windows.Forms.Label();
            this.lbl_itr_s = new System.Windows.Forms.Label();
            this.tableLayoutPanel12 = new System.Windows.Forms.TableLayoutPanel();
            this.label18 = new System.Windows.Forms.Label();
            this.lbl_rem_time = new System.Windows.Forms.Label();
            this.tableLayoutPanel6 = new System.Windows.Forms.TableLayoutPanel();
            this.label2 = new System.Windows.Forms.Label();
            this.lbl_tox = new System.Windows.Forms.Label();
            this.tableLayoutPanel13 = new System.Windows.Forms.TableLayoutPanel();
            this.label19 = new System.Windows.Forms.Label();
            this.lbl_el_time = new System.Windows.Forms.Label();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.statlbl = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel3 = new System.Windows.Forms.ToolStripStatusLabel();
            this.statlbl_sweep = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.txt_status = new System.Windows.Forms.TextBox();
            this.btn_clr = new System.Windows.Forms.Button();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel11 = new System.Windows.Forms.TableLayoutPanel();
            this.label21 = new System.Windows.Forms.Label();
            this.label20 = new System.Windows.Forms.Label();
            this.cmb_sw_sel2 = new System.Windows.Forms.ComboBox();
            this.txt_sw_range2 = new CueTextBox();
            this.txt_min_first_r = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.btn_snapshot = new System.Windows.Forms.Button();
            this.tableLayoutPanel14 = new System.Windows.Forms.TableLayoutPanel();
            this.label22 = new System.Windows.Forms.Label();
            this.lbl_live_neur_perc = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.picB)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.tableLayoutPanel1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.tableLayoutPanel2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.tableLayoutPanel4.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.tableLayoutPanel3.SuspendLayout();
            this.tableLayoutPanel9.SuspendLayout();
            this.tableLayoutPanel8.SuspendLayout();
            this.tableLayoutPanel10.SuspendLayout();
            this.tableLayoutPanel5.SuspendLayout();
            this.tableLayoutPanel7.SuspendLayout();
            this.tableLayoutPanel12.SuspendLayout();
            this.tableLayoutPanel6.SuspendLayout();
            this.tableLayoutPanel13.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.tableLayoutPanel11.SuspendLayout();
            this.tableLayoutPanel14.SuspendLayout();
            this.SuspendLayout();
            // 
            // picB
            // 
            this.picB.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.picB.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            this.picB.Location = new System.Drawing.Point(16, 196);
            this.picB.Name = "picB";
            this.picB.Size = new System.Drawing.Size(721, 529);
            this.picB.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.picB.TabIndex = 0;
            this.picB.TabStop = false;
            this.picB.Click += new System.EventHandler(this.picB_Click);
            this.picB.Paint += new System.Windows.Forms.PaintEventHandler(this.picB_Paint);
            this.picB.MouseDown += new System.Windows.Forms.MouseEventHandler(this.picB_MouseDown);
            this.picB.MouseMove += new System.Windows.Forms.MouseEventHandler(this.picB_MouseMove);
            this.picB.MouseUp += new System.Windows.Forms.MouseEventHandler(this.picB_MouseUp);
            this.picB.Resize += new System.EventHandler(this.picB_Resize);
            // 
            // btn_start
            // 
            this.btn_start.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.btn_start.Location = new System.Drawing.Point(7, 4);
            this.btn_start.Name = "btn_start";
            this.btn_start.Size = new System.Drawing.Size(108, 27);
            this.btn_start.TabIndex = 2;
            this.btn_start.Text = "&Start";
            this.btn_start.UseVisualStyleBackColor = true;
            // 
            // btn_reset
            // 
            this.btn_reset.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.btn_reset.Location = new System.Drawing.Point(133, 4);
            this.btn_reset.Name = "btn_reset";
            this.btn_reset.Size = new System.Drawing.Size(105, 27);
            this.btn_reset.TabIndex = 5;
            this.btn_reset.Text = "&Reset";
            this.btn_reset.UseVisualStyleBackColor = true;
            // 
            // btn_save_model
            // 
            this.btn_save_model.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.btn_save_model.Location = new System.Drawing.Point(35, 243);
            this.btn_save_model.Name = "btn_save_model";
            this.btn_save_model.Size = new System.Drawing.Size(105, 36);
            this.btn_save_model.TabIndex = 9;
            this.btn_save_model.Text = "Save Model";
            this.btn_save_model.UseVisualStyleBackColor = true;
            // 
            // btn_load_model
            // 
            this.btn_load_model.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.btn_load_model.Location = new System.Drawing.Point(146, 243);
            this.btn_load_model.Name = "btn_load_model";
            this.btn_load_model.Size = new System.Drawing.Size(105, 36);
            this.btn_load_model.TabIndex = 10;
            this.btn_load_model.Text = "Load Model";
            this.btn_load_model.UseVisualStyleBackColor = true;
            // 
            // btn_redraw
            // 
            this.btn_redraw.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.btn_redraw.Location = new System.Drawing.Point(257, 243);
            this.btn_redraw.Name = "btn_redraw";
            this.btn_redraw.Size = new System.Drawing.Size(89, 36);
            this.btn_redraw.TabIndex = 11;
            this.btn_redraw.Text = "Redraw";
            this.btn_redraw.UseVisualStyleBackColor = true;
            // 
            // chk_neur_lvl
            // 
            this.chk_neur_lvl.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.chk_neur_lvl.AutoSize = true;
            this.chk_neur_lvl.Location = new System.Drawing.Point(3, 37);
            this.chk_neur_lvl.Name = "chk_neur_lvl";
            this.chk_neur_lvl.Size = new System.Drawing.Size(123, 21);
            this.chk_neur_lvl.TabIndex = 12;
            this.chk_neur_lvl.Text = "Neuron Labels";
            this.chk_neur_lvl.UseVisualStyleBackColor = true;
            // 
            // label4
            // 
            this.label4.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label4.AutoSize = true;
            this.label4.ForeColor = System.Drawing.Color.Maroon;
            this.label4.Location = new System.Drawing.Point(34, 213);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(136, 17);
            this.label4.TabIndex = 19;
            this.label4.Text = "Number of Neurons:";
            // 
            // lbl_n_Neurs
            // 
            this.lbl_n_Neurs.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.lbl_n_Neurs.AutoSize = true;
            this.lbl_n_Neurs.ForeColor = System.Drawing.Color.Maroon;
            this.lbl_n_Neurs.Location = new System.Drawing.Point(168, 213);
            this.lbl_n_Neurs.Name = "lbl_n_Neurs";
            this.lbl_n_Neurs.Size = new System.Drawing.Size(16, 17);
            this.lbl_n_Neurs.TabIndex = 18;
            this.lbl_n_Neurs.Text = "0";
            // 
            // chk_show_bound
            // 
            this.chk_show_bound.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.chk_show_bound.AutoSize = true;
            this.chk_show_bound.Checked = true;
            this.chk_show_bound.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chk_show_bound.Location = new System.Drawing.Point(151, 5);
            this.chk_show_bound.Name = "chk_show_bound";
            this.chk_show_bound.Size = new System.Drawing.Size(71, 21);
            this.chk_show_bound.TabIndex = 20;
            this.chk_show_bound.Text = "Bound";
            this.chk_show_bound.UseVisualStyleBackColor = true;
            // 
            // chk_show_tox
            // 
            this.chk_show_tox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.chk_show_tox.AutoSize = true;
            this.chk_show_tox.Checked = true;
            this.chk_show_tox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chk_show_tox.Location = new System.Drawing.Point(151, 37);
            this.chk_show_tox.Name = "chk_show_tox";
            this.chk_show_tox.Size = new System.Drawing.Size(53, 21);
            this.chk_show_tox.TabIndex = 22;
            this.chk_show_tox.Text = "Tox";
            this.chk_show_tox.UseVisualStyleBackColor = true;
            // 
            // groupBox1
            // 
            this.groupBox1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox1.Controls.Add(this.btn_export);
            this.groupBox1.Controls.Add(this.chk_strict_rad);
            this.groupBox1.Controls.Add(this.lbl_mdl_prog);
            this.groupBox1.Controls.Add(this.label16);
            this.groupBox1.Controls.Add(this.tableLayoutPanel1);
            this.groupBox1.Controls.Add(this.btn_redraw);
            this.groupBox1.Controls.Add(this.btn_save_model);
            this.groupBox1.Controls.Add(this.btn_load_model);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.lbl_n_Neurs);
            this.groupBox1.Location = new System.Drawing.Point(743, 12);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(373, 285);
            this.groupBox1.TabIndex = 25;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Model";
            // 
            // btn_export
            // 
            this.btn_export.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.btn_export.Location = new System.Drawing.Point(277, 164);
            this.btn_export.Name = "btn_export";
            this.btn_export.Size = new System.Drawing.Size(69, 36);
            this.btn_export.TabIndex = 35;
            this.btn_export.Text = "Export";
            this.btn_export.UseVisualStyleBackColor = true;
            // 
            // chk_strict_rad
            // 
            this.chk_strict_rad.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.chk_strict_rad.AutoSize = true;
            this.chk_strict_rad.ForeColor = System.Drawing.SystemColors.HotTrack;
            this.chk_strict_rad.Location = new System.Drawing.Point(35, 173);
            this.chk_strict_rad.Name = "chk_strict_rad";
            this.chk_strict_rad.Size = new System.Drawing.Size(152, 21);
            this.chk_strict_rad.TabIndex = 34;
            this.chk_strict_rad.Text = "Strict Raduis Model";
            this.chk_strict_rad.UseVisualStyleBackColor = false;
            // 
            // lbl_mdl_prog
            // 
            this.lbl_mdl_prog.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.lbl_mdl_prog.AutoSize = true;
            this.lbl_mdl_prog.ForeColor = System.Drawing.Color.Maroon;
            this.lbl_mdl_prog.Location = new System.Drawing.Point(318, 213);
            this.lbl_mdl_prog.Name = "lbl_mdl_prog";
            this.lbl_mdl_prog.Size = new System.Drawing.Size(13, 17);
            this.lbl_mdl_prog.TabIndex = 21;
            this.lbl_mdl_prog.Text = "-";
            // 
            // label16
            // 
            this.label16.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label16.AutoSize = true;
            this.label16.ForeColor = System.Drawing.Color.Maroon;
            this.label16.Location = new System.Drawing.Point(233, 213);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(84, 17);
            this.label16.TabIndex = 20;
            this.label16.Text = "Model Prog:";
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.tableLayoutPanel1.ColumnCount = 2;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel1.Controls.Add(this.txt_clearance, 1, 4);
            this.tableLayoutPanel1.Controls.Add(this.label10, 0, 4);
            this.tableLayoutPanel1.Controls.Add(this.txt_nerve_rad, 1, 0);
            this.tableLayoutPanel1.Controls.Add(this.label11, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.label6, 0, 2);
            this.tableLayoutPanel1.Controls.Add(this.txt_min_rad, 1, 2);
            this.tableLayoutPanel1.Controls.Add(this.label5, 0, 3);
            this.tableLayoutPanel1.Controls.Add(this.txt_max_rad, 1, 3);
            this.tableLayoutPanel1.Controls.Add(this.txt_vein_rad, 1, 1);
            this.tableLayoutPanel1.Controls.Add(this.label15, 0, 1);
            this.tableLayoutPanel1.Location = new System.Drawing.Point(34, 21);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 5;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(292, 146);
            this.tableLayoutPanel1.TabIndex = 12;
            // 
            // txt_clearance
            // 
            this.txt_clearance.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_clearance.Location = new System.Drawing.Point(149, 120);
            this.txt_clearance.Name = "txt_clearance";
            this.txt_clearance.Size = new System.Drawing.Size(68, 22);
            this.txt_clearance.TabIndex = 30;
            // 
            // label10
            // 
            this.label10.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(38, 122);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(105, 17);
            this.label10.TabIndex = 29;
            this.label10.Text = "Clearance (um)";
            // 
            // txt_nerve_rad
            // 
            this.txt_nerve_rad.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_nerve_rad.Location = new System.Drawing.Point(149, 3);
            this.txt_nerve_rad.Name = "txt_nerve_rad";
            this.txt_nerve_rad.Size = new System.Drawing.Size(68, 22);
            this.txt_nerve_rad.TabIndex = 3;
            // 
            // label11
            // 
            this.label11.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(34, 6);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(109, 17);
            this.label11.TabIndex = 1;
            this.label11.Text = "Nerve Rad (um)";
            // 
            // label6
            // 
            this.label6.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(15, 64);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(128, 17);
            this.label6.TabIndex = 22;
            this.label6.Text = "Min Neur Rad (um)";
            // 
            // txt_min_rad
            // 
            this.txt_min_rad.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_min_rad.Location = new System.Drawing.Point(149, 61);
            this.txt_min_rad.Name = "txt_min_rad";
            this.txt_min_rad.Size = new System.Drawing.Size(68, 22);
            this.txt_min_rad.TabIndex = 23;
            // 
            // label5
            // 
            this.label5.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(12, 93);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(131, 17);
            this.label5.TabIndex = 24;
            this.label5.Text = "Max Neur Rad (um)";
            // 
            // txt_max_rad
            // 
            this.txt_max_rad.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_max_rad.Location = new System.Drawing.Point(149, 90);
            this.txt_max_rad.Name = "txt_max_rad";
            this.txt_max_rad.Size = new System.Drawing.Size(68, 22);
            this.txt_max_rad.TabIndex = 25;
            // 
            // txt_vein_rad
            // 
            this.txt_vein_rad.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_vein_rad.Location = new System.Drawing.Point(149, 32);
            this.txt_vein_rad.Name = "txt_vein_rad";
            this.txt_vein_rad.Size = new System.Drawing.Size(68, 22);
            this.txt_vein_rad.TabIndex = 3;
            // 
            // label15
            // 
            this.label15.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(26, 35);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(117, 17);
            this.label15.TabIndex = 1;
            this.label15.Text = "Vein Ratio (0→1)";
            // 
            // label8
            // 
            this.label8.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(5, 38);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(137, 17);
            this.label8.TabIndex = 20;
            this.label8.Text = "Neur Tolerance (>0)";
            // 
            // txt_Tol
            // 
            this.txt_Tol.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_Tol.Location = new System.Drawing.Point(148, 35);
            this.txt_Tol.Name = "txt_Tol";
            this.txt_Tol.Size = new System.Drawing.Size(68, 22);
            this.txt_Tol.TabIndex = 21;
            // 
            // label7
            // 
            this.label7.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(25, 69);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(117, 17);
            this.label7.TabIndex = 26;
            this.label7.Text = "Neur Rate (0→1)";
            // 
            // txt_neur_rate
            // 
            this.txt_neur_rate.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_neur_rate.Location = new System.Drawing.Point(148, 67);
            this.txt_neur_rate.Name = "txt_neur_rate";
            this.txt_neur_rate.Size = new System.Drawing.Size(68, 22);
            this.txt_neur_rate.TabIndex = 27;
            // 
            // groupBox2
            // 
            this.groupBox2.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox2.Controls.Add(this.label12);
            this.groupBox2.Controls.Add(this.lbl_image_siz);
            this.groupBox2.Controls.Add(this.btn_preprocess);
            this.groupBox2.Controls.Add(this.btn_load_setts);
            this.groupBox2.Controls.Add(this.btn_save_setts);
            this.groupBox2.Controls.Add(this.tableLayoutPanel2);
            this.groupBox2.Location = new System.Drawing.Point(743, 303);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(373, 207);
            this.groupBox2.TabIndex = 26;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Settings";
            // 
            // label12
            // 
            this.label12.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label12.AutoSize = true;
            this.label12.ForeColor = System.Drawing.Color.Maroon;
            this.label12.Location = new System.Drawing.Point(38, 132);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(81, 17);
            this.label12.TabIndex = 31;
            this.label12.Text = "Image Size:";
            // 
            // lbl_image_siz
            // 
            this.lbl_image_siz.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.lbl_image_siz.AutoSize = true;
            this.lbl_image_siz.ForeColor = System.Drawing.Color.Maroon;
            this.lbl_image_siz.Location = new System.Drawing.Point(125, 132);
            this.lbl_image_siz.Name = "lbl_image_siz";
            this.lbl_image_siz.Size = new System.Drawing.Size(13, 17);
            this.lbl_image_siz.TabIndex = 30;
            this.lbl_image_siz.Text = "-";
            // 
            // btn_preprocess
            // 
            this.btn_preprocess.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.btn_preprocess.Location = new System.Drawing.Point(256, 160);
            this.btn_preprocess.Name = "btn_preprocess";
            this.btn_preprocess.Size = new System.Drawing.Size(89, 36);
            this.btn_preprocess.TabIndex = 29;
            this.btn_preprocess.Text = "Update";
            this.btn_preprocess.UseVisualStyleBackColor = true;
            // 
            // btn_load_setts
            // 
            this.btn_load_setts.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.btn_load_setts.Location = new System.Drawing.Point(145, 160);
            this.btn_load_setts.Name = "btn_load_setts";
            this.btn_load_setts.Size = new System.Drawing.Size(105, 36);
            this.btn_load_setts.TabIndex = 29;
            this.btn_load_setts.Text = "Load Settings";
            this.btn_load_setts.UseVisualStyleBackColor = true;
            // 
            // btn_save_setts
            // 
            this.btn_save_setts.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.btn_save_setts.Location = new System.Drawing.Point(34, 160);
            this.btn_save_setts.Name = "btn_save_setts";
            this.btn_save_setts.Size = new System.Drawing.Size(105, 36);
            this.btn_save_setts.TabIndex = 20;
            this.btn_save_setts.Text = "Save Settings";
            this.btn_save_setts.UseVisualStyleBackColor = true;
            // 
            // tableLayoutPanel2
            // 
            this.tableLayoutPanel2.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.tableLayoutPanel2.ColumnCount = 2;
            this.tableLayoutPanel2.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 51.22807F));
            this.tableLayoutPanel2.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 48.77193F));
            this.tableLayoutPanel2.Controls.Add(this.txt_resolution, 1, 0);
            this.tableLayoutPanel2.Controls.Add(this.txt_neur_rate, 1, 2);
            this.tableLayoutPanel2.Controls.Add(this.label8, 0, 1);
            this.tableLayoutPanel2.Controls.Add(this.label7, 0, 2);
            this.tableLayoutPanel2.Controls.Add(this.txt_Tol, 1, 1);
            this.tableLayoutPanel2.Controls.Add(this.label9, 0, 0);
            this.tableLayoutPanel2.Location = new System.Drawing.Point(41, 27);
            this.tableLayoutPanel2.Name = "tableLayoutPanel2";
            this.tableLayoutPanel2.RowCount = 3;
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 14.28571F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 20F));
            this.tableLayoutPanel2.Size = new System.Drawing.Size(285, 94);
            this.tableLayoutPanel2.TabIndex = 12;
            // 
            // txt_resolution
            // 
            this.txt_resolution.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_resolution.Location = new System.Drawing.Point(148, 4);
            this.txt_resolution.Name = "txt_resolution";
            this.txt_resolution.Size = new System.Drawing.Size(68, 22);
            this.txt_resolution.TabIndex = 32;
            // 
            // label9
            // 
            this.label9.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(13, 7);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(129, 17);
            this.label9.TabIndex = 29;
            this.label9.Text = "Resolution (pix/um)";
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.tableLayoutPanel4);
            this.groupBox3.Location = new System.Drawing.Point(16, 12);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(406, 67);
            this.groupBox3.TabIndex = 27;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Simulation";
            // 
            // tableLayoutPanel4
            // 
            this.tableLayoutPanel4.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.tableLayoutPanel4.ColumnCount = 3;
            this.tableLayoutPanel4.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 31.47208F));
            this.tableLayoutPanel4.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 31.72589F));
            this.tableLayoutPanel4.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 36.80203F));
            this.tableLayoutPanel4.Controls.Add(this.chk_save_prog, 2, 0);
            this.tableLayoutPanel4.Controls.Add(this.btn_start, 0, 0);
            this.tableLayoutPanel4.Controls.Add(this.btn_reset, 1, 0);
            this.tableLayoutPanel4.Location = new System.Drawing.Point(12, 24);
            this.tableLayoutPanel4.Name = "tableLayoutPanel4";
            this.tableLayoutPanel4.RowCount = 1;
            this.tableLayoutPanel4.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel4.Size = new System.Drawing.Size(394, 36);
            this.tableLayoutPanel4.TabIndex = 24;
            // 
            // chk_save_prog
            // 
            this.chk_save_prog.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.chk_save_prog.Appearance = System.Windows.Forms.Appearance.Button;
            this.chk_save_prog.AutoSize = true;
            this.chk_save_prog.Location = new System.Drawing.Point(265, 4);
            this.chk_save_prog.Name = "chk_save_prog";
            this.chk_save_prog.Size = new System.Drawing.Size(111, 27);
            this.chk_save_prog.TabIndex = 38;
            this.chk_save_prog.Text = "Save Progress";
            this.chk_save_prog.UseVisualStyleBackColor = true;
            // 
            // btn_sweep
            // 
            this.btn_sweep.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.btn_sweep.Location = new System.Drawing.Point(12, 4);
            this.btn_sweep.Name = "btn_sweep";
            this.btn_sweep.Size = new System.Drawing.Size(108, 27);
            this.btn_sweep.TabIndex = 24;
            this.btn_sweep.Text = "S&weep";
            this.btn_sweep.UseVisualStyleBackColor = true;
            this.btn_sweep.Click += new System.EventHandler(this.btn_sweep_Click);
            // 
            // cmb_sw_sel1
            // 
            this.cmb_sw_sel1.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.cmb_sw_sel1.FormattingEnabled = true;
            this.cmb_sw_sel1.Items.AddRange(new object[] {
            "Repeat",
            "Nerve Rad",
            "Min Rad",
            "Max Rad",
            "Clearance",
            "Resolution",
            "Tolerance",
            "Neur Rate",
            "Insult Rad",
            "Insult Peri"});
            this.cmb_sw_sel1.Location = new System.Drawing.Point(183, 5);
            this.cmb_sw_sel1.Name = "cmb_sw_sel1";
            this.cmb_sw_sel1.Size = new System.Drawing.Size(79, 24);
            this.cmb_sw_sel1.TabIndex = 34;
            // 
            // txt_sw_range1
            // 
            this.txt_sw_range1.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.txt_sw_range1.Cue = "#itrs , start , end";
            this.txt_sw_range1.Location = new System.Drawing.Point(274, 6);
            this.txt_sw_range1.Name = "txt_sw_range1";
            this.txt_sw_range1.Size = new System.Drawing.Size(110, 22);
            this.txt_sw_range1.TabIndex = 24;
            // 
            // chk_save_sw_prog
            // 
            this.chk_save_sw_prog.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.chk_save_sw_prog.Appearance = System.Windows.Forms.Appearance.Button;
            this.chk_save_sw_prog.AutoSize = true;
            this.chk_save_sw_prog.ForeColor = System.Drawing.SystemColors.HotTrack;
            this.chk_save_sw_prog.Location = new System.Drawing.Point(18, 39);
            this.chk_save_sw_prog.Name = "chk_save_sw_prog";
            this.chk_save_sw_prog.Size = new System.Drawing.Size(96, 27);
            this.chk_save_sw_prog.TabIndex = 37;
            this.chk_save_sw_prog.Text = "Save Sweep";
            this.chk_save_sw_prog.UseVisualStyleBackColor = true;
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.tableLayoutPanel3);
            this.groupBox4.Location = new System.Drawing.Point(428, 12);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(309, 98);
            this.groupBox4.TabIndex = 28;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Display";
            // 
            // tableLayoutPanel3
            // 
            this.tableLayoutPanel3.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.tableLayoutPanel3.ColumnCount = 2;
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel3.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel3.Controls.Add(this.chk_rec_avi, 0, 0);
            this.tableLayoutPanel3.Controls.Add(this.chk_neur_lvl, 0, 1);
            this.tableLayoutPanel3.Controls.Add(this.chk_show_bound, 1, 0);
            this.tableLayoutPanel3.Controls.Add(this.chk_show_tox, 1, 1);
            this.tableLayoutPanel3.Location = new System.Drawing.Point(6, 24);
            this.tableLayoutPanel3.Name = "tableLayoutPanel3";
            this.tableLayoutPanel3.RowCount = 2;
            this.tableLayoutPanel3.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel3.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel3.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 20F));
            this.tableLayoutPanel3.Size = new System.Drawing.Size(297, 64);
            this.tableLayoutPanel3.TabIndex = 12;
            // 
            // chk_rec_avi
            // 
            this.chk_rec_avi.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.chk_rec_avi.Appearance = System.Windows.Forms.Appearance.Button;
            this.chk_rec_avi.AutoSize = true;
            this.chk_rec_avi.ForeColor = System.Drawing.SystemColors.HotTrack;
            this.chk_rec_avi.Location = new System.Drawing.Point(3, 3);
            this.chk_rec_avi.Name = "chk_rec_avi";
            this.chk_rec_avi.Size = new System.Drawing.Size(89, 26);
            this.chk_rec_avi.TabIndex = 24;
            this.chk_rec_avi.Text = "Record AVI";
            this.chk_rec_avi.UseVisualStyleBackColor = true;
            // 
            // tableLayoutPanel9
            // 
            this.tableLayoutPanel9.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.tableLayoutPanel9.ColumnCount = 4;
            this.tableLayoutPanel9.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 19.72222F));
            this.tableLayoutPanel9.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 24.02778F));
            this.tableLayoutPanel9.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 26.35229F));
            this.tableLayoutPanel9.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 29.95839F));
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel14, 0, 1);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel8, 0, 0);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel10, 0, 0);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel5, 0, 0);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel7, 0, 1);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel12, 3, 0);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel6, 1, 1);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel13, 3, 1);
            this.tableLayoutPanel9.Location = new System.Drawing.Point(16, 731);
            this.tableLayoutPanel9.Name = "tableLayoutPanel9";
            this.tableLayoutPanel9.RowCount = 2;
            this.tableLayoutPanel9.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel9.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel9.Size = new System.Drawing.Size(721, 49);
            this.tableLayoutPanel9.TabIndex = 29;
            // 
            // tableLayoutPanel8
            // 
            this.tableLayoutPanel8.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel8.ColumnCount = 2;
            this.tableLayoutPanel8.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 68.92655F));
            this.tableLayoutPanel8.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 31.07345F));
            this.tableLayoutPanel8.Controls.Add(this.label17, 0, 0);
            this.tableLayoutPanel8.Controls.Add(this.lbl_chron_progress, 1, 0);
            this.tableLayoutPanel8.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel8.Location = new System.Drawing.Point(145, 3);
            this.tableLayoutPanel8.Name = "tableLayoutPanel8";
            this.tableLayoutPanel8.RowCount = 1;
            this.tableLayoutPanel8.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel8.Size = new System.Drawing.Size(167, 18);
            this.tableLayoutPanel8.TabIndex = 36;
            // 
            // label17
            // 
            this.label17.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(5, 0);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(107, 17);
            this.label17.TabIndex = 16;
            this.label17.Text = "Chron Progress";
            // 
            // lbl_chron_progress
            // 
            this.lbl_chron_progress.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_chron_progress.AutoSize = true;
            this.lbl_chron_progress.Location = new System.Drawing.Point(118, 0);
            this.lbl_chron_progress.Name = "lbl_chron_progress";
            this.lbl_chron_progress.Size = new System.Drawing.Size(16, 17);
            this.lbl_chron_progress.TabIndex = 15;
            this.lbl_chron_progress.Text = "0";
            // 
            // tableLayoutPanel10
            // 
            this.tableLayoutPanel10.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel10.ColumnCount = 2;
            this.tableLayoutPanel10.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 69.46107F));
            this.tableLayoutPanel10.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 30.53892F));
            this.tableLayoutPanel10.Controls.Add(this.label14, 0, 0);
            this.tableLayoutPanel10.Controls.Add(this.lbl_areal_progress, 1, 0);
            this.tableLayoutPanel10.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel10.Location = new System.Drawing.Point(3, 3);
            this.tableLayoutPanel10.Name = "tableLayoutPanel10";
            this.tableLayoutPanel10.RowCount = 1;
            this.tableLayoutPanel10.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel10.Size = new System.Drawing.Size(136, 18);
            this.tableLayoutPanel10.TabIndex = 35;
            // 
            // label14
            // 
            this.label14.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(26, 0);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(65, 18);
            this.label14.TabIndex = 16;
            this.label14.Text = "Areal Progress";
            // 
            // lbl_areal_progress
            // 
            this.lbl_areal_progress.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_areal_progress.AutoSize = true;
            this.lbl_areal_progress.Location = new System.Drawing.Point(97, 0);
            this.lbl_areal_progress.Name = "lbl_areal_progress";
            this.lbl_areal_progress.Size = new System.Drawing.Size(16, 17);
            this.lbl_areal_progress.TabIndex = 15;
            this.lbl_areal_progress.Text = "0";
            // 
            // tableLayoutPanel5
            // 
            this.tableLayoutPanel5.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel5.ColumnCount = 2;
            this.tableLayoutPanel5.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel5.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel5.Controls.Add(this.label1, 0, 0);
            this.tableLayoutPanel5.Controls.Add(this.lbl_itr, 1, 0);
            this.tableLayoutPanel5.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel5.Location = new System.Drawing.Point(318, 3);
            this.tableLayoutPanel5.Name = "tableLayoutPanel5";
            this.tableLayoutPanel5.RowCount = 1;
            this.tableLayoutPanel5.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel5.Size = new System.Drawing.Size(183, 18);
            this.tableLayoutPanel5.TabIndex = 32;
            // 
            // label1
            // 
            this.label1.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(25, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(63, 17);
            this.label1.TabIndex = 7;
            this.label1.Text = "Iteration:";
            // 
            // lbl_itr
            // 
            this.lbl_itr.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_itr.AutoSize = true;
            this.lbl_itr.Location = new System.Drawing.Point(94, 0);
            this.lbl_itr.Name = "lbl_itr";
            this.lbl_itr.Size = new System.Drawing.Size(16, 17);
            this.lbl_itr.TabIndex = 3;
            this.lbl_itr.Text = "0";
            // 
            // tableLayoutPanel7
            // 
            this.tableLayoutPanel7.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel7.ColumnCount = 2;
            this.tableLayoutPanel7.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 45.05495F));
            this.tableLayoutPanel7.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 54.94505F));
            this.tableLayoutPanel7.Controls.Add(this.label3, 0, 0);
            this.tableLayoutPanel7.Controls.Add(this.lbl_itr_s, 1, 0);
            this.tableLayoutPanel7.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel7.Location = new System.Drawing.Point(3, 27);
            this.tableLayoutPanel7.Name = "tableLayoutPanel7";
            this.tableLayoutPanel7.RowCount = 1;
            this.tableLayoutPanel7.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel7.Size = new System.Drawing.Size(136, 19);
            this.tableLayoutPanel7.TabIndex = 33;
            // 
            // label3
            // 
            this.label3.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(23, 1);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(35, 17);
            this.label3.TabIndex = 16;
            this.label3.Text = "itr/s:";
            // 
            // lbl_itr_s
            // 
            this.lbl_itr_s.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_itr_s.AutoSize = true;
            this.lbl_itr_s.Location = new System.Drawing.Point(64, 1);
            this.lbl_itr_s.Name = "lbl_itr_s";
            this.lbl_itr_s.Size = new System.Drawing.Size(16, 17);
            this.lbl_itr_s.TabIndex = 15;
            this.lbl_itr_s.Text = "0";
            // 
            // tableLayoutPanel12
            // 
            this.tableLayoutPanel12.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel12.ColumnCount = 2;
            this.tableLayoutPanel12.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel12.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel12.Controls.Add(this.label18, 0, 0);
            this.tableLayoutPanel12.Controls.Add(this.lbl_rem_time, 1, 0);
            this.tableLayoutPanel12.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel12.Location = new System.Drawing.Point(507, 3);
            this.tableLayoutPanel12.Name = "tableLayoutPanel12";
            this.tableLayoutPanel12.RowCount = 1;
            this.tableLayoutPanel12.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel12.Size = new System.Drawing.Size(211, 18);
            this.tableLayoutPanel12.TabIndex = 38;
            // 
            // label18
            // 
            this.label18.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(27, 0);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(75, 17);
            this.label18.TabIndex = 7;
            this.label18.Text = "Remaining";
            // 
            // lbl_rem_time
            // 
            this.lbl_rem_time.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_rem_time.AutoSize = true;
            this.lbl_rem_time.Location = new System.Drawing.Point(108, 0);
            this.lbl_rem_time.Name = "lbl_rem_time";
            this.lbl_rem_time.Size = new System.Drawing.Size(16, 17);
            this.lbl_rem_time.TabIndex = 3;
            this.lbl_rem_time.Text = "0";
            // 
            // tableLayoutPanel6
            // 
            this.tableLayoutPanel6.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel6.ColumnCount = 2;
            this.tableLayoutPanel6.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel6.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel6.Controls.Add(this.label2, 0, 0);
            this.tableLayoutPanel6.Controls.Add(this.lbl_tox, 1, 0);
            this.tableLayoutPanel6.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel6.Location = new System.Drawing.Point(318, 27);
            this.tableLayoutPanel6.Name = "tableLayoutPanel6";
            this.tableLayoutPanel6.RowCount = 1;
            this.tableLayoutPanel6.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel6.Size = new System.Drawing.Size(183, 19);
            this.tableLayoutPanel6.TabIndex = 34;
            // 
            // label2
            // 
            this.label2.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(37, 1);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(51, 17);
            this.label2.TabIndex = 8;
            this.label2.Text = "tox_lvl:";
            // 
            // lbl_tox
            // 
            this.lbl_tox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_tox.AutoSize = true;
            this.lbl_tox.Location = new System.Drawing.Point(94, 1);
            this.lbl_tox.Name = "lbl_tox";
            this.lbl_tox.Size = new System.Drawing.Size(16, 17);
            this.lbl_tox.TabIndex = 1;
            this.lbl_tox.Text = "0";
            // 
            // tableLayoutPanel13
            // 
            this.tableLayoutPanel13.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel13.ColumnCount = 2;
            this.tableLayoutPanel13.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel13.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel13.Controls.Add(this.label19, 0, 0);
            this.tableLayoutPanel13.Controls.Add(this.lbl_el_time, 1, 0);
            this.tableLayoutPanel13.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel13.Location = new System.Drawing.Point(507, 27);
            this.tableLayoutPanel13.Name = "tableLayoutPanel13";
            this.tableLayoutPanel13.RowCount = 1;
            this.tableLayoutPanel13.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel13.Size = new System.Drawing.Size(211, 18);
            this.tableLayoutPanel13.TabIndex = 39;
            // 
            // label19
            // 
            this.label19.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(43, 0);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(59, 17);
            this.label19.TabIndex = 7;
            this.label19.Text = "Elapsed";
            // 
            // lbl_el_time
            // 
            this.lbl_el_time.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_el_time.AutoSize = true;
            this.lbl_el_time.Location = new System.Drawing.Point(108, 0);
            this.lbl_el_time.Name = "lbl_el_time";
            this.lbl_el_time.Size = new System.Drawing.Size(16, 17);
            this.lbl_el_time.TabIndex = 3;
            this.lbl_el_time.Text = "0";
            // 
            // statusStrip1
            // 
            this.statusStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.statlbl,
            this.toolStripStatusLabel3,
            this.statlbl_sweep,
            this.toolStripStatusLabel1,
            this.toolStripStatusLabel2});
            this.statusStrip1.Location = new System.Drawing.Point(0, 782);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(1137, 25);
            this.statusStrip1.TabIndex = 30;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // statlbl
            // 
            this.statlbl.Name = "statlbl";
            this.statlbl.Size = new System.Drawing.Size(73, 20);
            this.statlbl.Text = "                ";
            // 
            // toolStripStatusLabel3
            // 
            this.toolStripStatusLabel3.Name = "toolStripStatusLabel3";
            this.toolStripStatusLabel3.Size = new System.Drawing.Size(117, 20);
            this.toolStripStatusLabel3.Text = "                           ";
            // 
            // statlbl_sweep
            // 
            this.statlbl_sweep.Name = "statlbl_sweep";
            this.statlbl_sweep.Size = new System.Drawing.Size(105, 20);
            this.statlbl_sweep.Text = "                        ";
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(758, 20);
            this.toolStripStatusLabel1.Spring = true;
            // 
            // toolStripStatusLabel2
            // 
            this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
            this.toolStripStatusLabel2.Size = new System.Drawing.Size(69, 20);
            this.toolStripStatusLabel2.Text = "               ";
            // 
            // txt_status
            // 
            this.txt_status.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txt_status.Location = new System.Drawing.Point(743, 516);
            this.txt_status.Multiline = true;
            this.txt_status.Name = "txt_status";
            this.txt_status.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.txt_status.Size = new System.Drawing.Size(373, 255);
            this.txt_status.TabIndex = 31;
            // 
            // btn_clr
            // 
            this.btn_clr.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btn_clr.Location = new System.Drawing.Point(1073, 516);
            this.btn_clr.Name = "btn_clr";
            this.btn_clr.Size = new System.Drawing.Size(43, 27);
            this.btn_clr.TabIndex = 32;
            this.btn_clr.Text = "clr";
            this.btn_clr.UseVisualStyleBackColor = true;
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.tableLayoutPanel11);
            this.groupBox5.Location = new System.Drawing.Point(16, 85);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(406, 105);
            this.groupBox5.TabIndex = 33;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "Parameter Sweep";
            // 
            // tableLayoutPanel11
            // 
            this.tableLayoutPanel11.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.tableLayoutPanel11.ColumnCount = 4;
            this.tableLayoutPanel11.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 38.32487F));
            this.tableLayoutPanel11.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 47F));
            this.tableLayoutPanel11.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 24.61929F));
            this.tableLayoutPanel11.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 36.80203F));
            this.tableLayoutPanel11.Controls.Add(this.label21, 1, 1);
            this.tableLayoutPanel11.Controls.Add(this.label20, 1, 0);
            this.tableLayoutPanel11.Controls.Add(this.btn_sweep, 0, 0);
            this.tableLayoutPanel11.Controls.Add(this.cmb_sw_sel1, 2, 0);
            this.tableLayoutPanel11.Controls.Add(this.txt_sw_range1, 3, 0);
            this.tableLayoutPanel11.Controls.Add(this.chk_save_sw_prog, 0, 1);
            this.tableLayoutPanel11.Controls.Add(this.cmb_sw_sel2, 2, 1);
            this.tableLayoutPanel11.Controls.Add(this.txt_sw_range2, 3, 1);
            this.tableLayoutPanel11.Location = new System.Drawing.Point(12, 24);
            this.tableLayoutPanel11.Name = "tableLayoutPanel11";
            this.tableLayoutPanel11.RowCount = 2;
            this.tableLayoutPanel11.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel11.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel11.Size = new System.Drawing.Size(394, 70);
            this.tableLayoutPanel11.TabIndex = 24;
            // 
            // label21
            // 
            this.label21.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.label21.AutoSize = true;
            this.label21.Location = new System.Drawing.Point(140, 44);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(32, 17);
            this.label21.TabIndex = 40;
            this.label21.Text = "2nd";
            // 
            // label20
            // 
            this.label20.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.label20.AutoSize = true;
            this.label20.Location = new System.Drawing.Point(143, 9);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(27, 17);
            this.label20.TabIndex = 37;
            this.label20.Text = "1st";
            // 
            // cmb_sw_sel2
            // 
            this.cmb_sw_sel2.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.cmb_sw_sel2.FormattingEnabled = true;
            this.cmb_sw_sel2.Items.AddRange(new object[] {
            "Repeat",
            "Nerve Rad",
            "Min Rad",
            "Max Rad",
            "Clearance",
            "Resolution",
            "Tolerance",
            "Neur Rate",
            "Insult Rad",
            "Insult Peri"});
            this.cmb_sw_sel2.Location = new System.Drawing.Point(183, 40);
            this.cmb_sw_sel2.Name = "cmb_sw_sel2";
            this.cmb_sw_sel2.Size = new System.Drawing.Size(79, 24);
            this.cmb_sw_sel2.TabIndex = 38;
            // 
            // txt_sw_range2
            // 
            this.txt_sw_range2.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.txt_sw_range2.Cue = "#itrs , start , end";
            this.txt_sw_range2.Location = new System.Drawing.Point(274, 41);
            this.txt_sw_range2.Name = "txt_sw_range2";
            this.txt_sw_range2.Size = new System.Drawing.Size(110, 22);
            this.txt_sw_range2.TabIndex = 39;
            // 
            // txt_min_first_r
            // 
            this.txt_min_first_r.Location = new System.Drawing.Point(545, 111);
            this.txt_min_first_r.Name = "txt_min_first_r";
            this.txt_min_first_r.Size = new System.Drawing.Size(68, 22);
            this.txt_min_first_r.TabIndex = 35;
            this.txt_min_first_r.Text = "1";
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(434, 113);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(107, 17);
            this.label13.TabIndex = 34;
            this.label13.Text = "min_first_r (um)";
            // 
            // btn_snapshot
            // 
            this.btn_snapshot.Location = new System.Drawing.Point(626, 108);
            this.btn_snapshot.Name = "btn_snapshot";
            this.btn_snapshot.Size = new System.Drawing.Size(105, 27);
            this.btn_snapshot.TabIndex = 36;
            this.btn_snapshot.Text = "Snapshot";
            this.btn_snapshot.UseVisualStyleBackColor = true;
            // 
            // tableLayoutPanel14
            // 
            this.tableLayoutPanel14.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel14.ColumnCount = 2;
            this.tableLayoutPanel14.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel14.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel14.Controls.Add(this.label22, 0, 0);
            this.tableLayoutPanel14.Controls.Add(this.lbl_live_neur_perc, 1, 0);
            this.tableLayoutPanel14.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel14.Location = new System.Drawing.Point(145, 27);
            this.tableLayoutPanel14.Name = "tableLayoutPanel14";
            this.tableLayoutPanel14.RowCount = 1;
            this.tableLayoutPanel14.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel14.Size = new System.Drawing.Size(167, 19);
            this.tableLayoutPanel14.TabIndex = 40;
            // 
            // label22
            // 
            this.label22.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label22.AutoSize = true;
            this.label22.Location = new System.Drawing.Point(7, 0);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(73, 19);
            this.label22.TabIndex = 8;
            this.label22.Text = "live neurs (%):";
            // 
            // lbl_live_neur_perc
            // 
            this.lbl_live_neur_perc.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_live_neur_perc.AutoSize = true;
            this.lbl_live_neur_perc.Location = new System.Drawing.Point(86, 1);
            this.lbl_live_neur_perc.Name = "lbl_live_neur_perc";
            this.lbl_live_neur_perc.Size = new System.Drawing.Size(16, 17);
            this.lbl_live_neur_perc.TabIndex = 1;
            this.lbl_live_neur_perc.Text = "0";
            // 
            // Main_Form
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1137, 807);
            this.Controls.Add(this.btn_snapshot);
            this.Controls.Add(this.txt_min_first_r);
            this.Controls.Add(this.label13);
            this.Controls.Add(this.groupBox5);
            this.Controls.Add(this.btn_clr);
            this.Controls.Add(this.txt_status);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.tableLayoutPanel9);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.picB);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "Main_Form";
            this.Text = "LHON";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Main_Form_FormClosing);
            this.Load += new System.EventHandler(this.Main_Form_Load);
            ((System.ComponentModel.ISupportInitialize)(this.picB)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.tableLayoutPanel1.ResumeLayout(false);
            this.tableLayoutPanel1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.tableLayoutPanel2.ResumeLayout(false);
            this.tableLayoutPanel2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.tableLayoutPanel4.ResumeLayout(false);
            this.tableLayoutPanel4.PerformLayout();
            this.groupBox4.ResumeLayout(false);
            this.tableLayoutPanel3.ResumeLayout(false);
            this.tableLayoutPanel3.PerformLayout();
            this.tableLayoutPanel9.ResumeLayout(false);
            this.tableLayoutPanel8.ResumeLayout(false);
            this.tableLayoutPanel8.PerformLayout();
            this.tableLayoutPanel10.ResumeLayout(false);
            this.tableLayoutPanel10.PerformLayout();
            this.tableLayoutPanel5.ResumeLayout(false);
            this.tableLayoutPanel5.PerformLayout();
            this.tableLayoutPanel7.ResumeLayout(false);
            this.tableLayoutPanel7.PerformLayout();
            this.tableLayoutPanel12.ResumeLayout(false);
            this.tableLayoutPanel12.PerformLayout();
            this.tableLayoutPanel6.ResumeLayout(false);
            this.tableLayoutPanel6.PerformLayout();
            this.tableLayoutPanel13.ResumeLayout(false);
            this.tableLayoutPanel13.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            this.tableLayoutPanel11.ResumeLayout(false);
            this.tableLayoutPanel11.PerformLayout();
            this.tableLayoutPanel14.ResumeLayout(false);
            this.tableLayoutPanel14.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private PictureBoxWithInterpolationMode picB;
        private System.Windows.Forms.Button btn_start;
        private Button btn_reset;
        private Button btn_save_model;
        private Button btn_load_model;
        private Button btn_redraw;
        private CheckBox chk_neur_lvl;
        private Label label4;
        private Label lbl_n_Neurs;
        private CheckBox chk_show_bound;
        private CheckBox chk_show_tox;
        private GroupBox groupBox1;
        private Label label8;
        private TextBox txt_Tol;
        private Label label6;
        private TextBox txt_min_rad;
        private Label label5;
        private TextBox txt_max_rad;
        private Label label7;
        private TextBox txt_neur_rate;
        private TableLayoutPanel tableLayoutPanel1;
        private GroupBox groupBox2;
        private TableLayoutPanel tableLayoutPanel2;
        private GroupBox groupBox3;
        private TableLayoutPanel tableLayoutPanel4;
        private GroupBox groupBox4;
        private TableLayoutPanel tableLayoutPanel3;
        private TextBox txt_nerve_rad;
        private Label label11;
        private Button btn_load_setts;
        private Button btn_save_setts;
        private TextBox txt_resolution;
        private Label label9;
        private Button btn_preprocess;
        private TextBox txt_clearance;
        private Label label10;
        private TableLayoutPanel tableLayoutPanel9;
        private TableLayoutPanel tableLayoutPanel7;
        private Label label3;
        private Label lbl_itr_s;
        private Button btn_sweep;
        private StatusStrip statusStrip1;
        private ToolStripStatusLabel statlbl;
        private ToolStripStatusLabel toolStripStatusLabel3;
        private ToolStripStatusLabel statlbl_sweep;
        private ToolStripStatusLabel toolStripStatusLabel1;
        private ToolStripStatusLabel toolStripStatusLabel2;
        private TextBox txt_status;
        private Button btn_clr;
        private ComboBox cmb_sw_sel1;
        private CueTextBox txt_sw_range1;
        private Label label12;
        private Label lbl_image_siz;
        private CheckBox chk_rec_avi;
        private CheckBox chk_save_sw_prog;
        private GroupBox groupBox5;
        private TableLayoutPanel tableLayoutPanel11;
        private CheckBox chk_save_prog;
        private TextBox txt_vein_rad;
        private Label label15;
        private Label lbl_mdl_prog;
        private Label label16;
        private CheckBox chk_strict_rad;
        private Button btn_export;
        private TextBox txt_min_first_r;
        private Label label13;
        private TableLayoutPanel tableLayoutPanel8;
        private Label label17;
        private Label lbl_chron_progress;
        private TableLayoutPanel tableLayoutPanel10;
        private Label label14;
        private Label lbl_areal_progress;
        private TableLayoutPanel tableLayoutPanel5;
        private Label label1;
        private Label lbl_itr;
        private TableLayoutPanel tableLayoutPanel12;
        private Label label18;
        private Label lbl_rem_time;
        private TableLayoutPanel tableLayoutPanel6;
        private Label label2;
        private Label lbl_tox;
        private TableLayoutPanel tableLayoutPanel13;
        private Label label19;
        private Label lbl_el_time;
        private Button btn_snapshot;
        private Label label21;
        private Label label20;
        private ComboBox cmb_sw_sel2;
        private CueTextBox txt_sw_range2;
        private TableLayoutPanel tableLayoutPanel14;
        private Label label22;
        private Label lbl_live_neur_perc;
    }
}

