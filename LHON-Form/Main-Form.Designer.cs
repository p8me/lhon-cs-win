using System.Drawing.Drawing2D;
using System.Windows.Forms;
using CustomControls;

namespace LHON_Form
{
    partial class Main_Form
    {
        const bool first_compile = true;

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

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Main_Form));
            this.picB = new CustomControls.PictureBoxWithInterpolationMode();
            this.txt_sw_range1 = new CustomControls.CueTextBox();
            this.txt_sw_range2 = new CustomControls.CueTextBox();
            this.btn_start = new System.Windows.Forms.Button();
            this.btn_reset = new System.Windows.Forms.Button();
            this.btn_save_model = new System.Windows.Forms.Button();
            this.btn_load_model = new System.Windows.Forms.Button();
            this.btn_redraw = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.lbl_num_axons = new System.Windows.Forms.Label();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.label15 = new System.Windows.Forms.Label();
            this.txt_clearance = new System.Windows.Forms.TextBox();
            this.txt_nerve_scale = new System.Windows.Forms.TextBox();
            this.label11 = new System.Windows.Forms.Label();
            this.lbl_nerve_siz = new System.Windows.Forms.Label();
            this.chk_strict_rad = new System.Windows.Forms.CheckBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.txt_on_death_tox = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.txt_insult_tox = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.txt_tox_prod_rate = new System.Windows.Forms.TextBox();
            this.label29 = new System.Windows.Forms.Label();
            this.label36 = new System.Windows.Forms.Label();
            this.txt_death_tox_threshold = new System.Windows.Forms.TextBox();
            this.groupBox7 = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel2 = new System.Windows.Forms.TableLayoutPanel();
            this.label26 = new System.Windows.Forms.Label();
            this.txt_rate_live = new System.Windows.Forms.TextBox();
            this.txt_rate_bound = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.txt_rate_dead = new System.Windows.Forms.TextBox();
            this.label25 = new System.Windows.Forms.Label();
            this.txt_rate_extra = new System.Windows.Forms.TextBox();
            this.label30 = new System.Windows.Forms.Label();
            this.label31 = new System.Windows.Forms.Label();
            this.label32 = new System.Windows.Forms.Label();
            this.label33 = new System.Windows.Forms.Label();
            this.groupBox6 = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel15 = new System.Windows.Forms.TableLayoutPanel();
            this.txt_detox_extra = new System.Windows.Forms.TextBox();
            this.label27 = new System.Windows.Forms.Label();
            this.label28 = new System.Windows.Forms.Label();
            this.txt_detox_intra = new System.Windows.Forms.TextBox();
            this.label34 = new System.Windows.Forms.Label();
            this.label35 = new System.Windows.Forms.Label();
            this.txt_resolution = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.label12 = new System.Windows.Forms.Label();
            this.lbl_image_siz = new System.Windows.Forms.Label();
            this.btn_preprocess = new System.Windows.Forms.Button();
            this.btn_load_setts = new System.Windows.Forms.Button();
            this.btn_save_setts = new System.Windows.Forms.Button();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel4 = new System.Windows.Forms.TableLayoutPanel();
            this.btn_save_prog = new System.Windows.Forms.Button();
            this.btn_sweep = new System.Windows.Forms.Button();
            this.cmb_sw_sel1 = new System.Windows.Forms.ComboBox();
            this.chk_save_sw_prog = new System.Windows.Forms.CheckBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.txt_rec_inerval = new System.Windows.Forms.TextBox();
            this.chk_show_axons = new System.Windows.Forms.CheckBox();
            this.label17 = new System.Windows.Forms.Label();
            this.chk_show_tox = new System.Windows.Forms.CheckBox();
            this.chk_rec_avi = new System.Windows.Forms.CheckBox();
            this.tableLayoutPanel9 = new System.Windows.Forms.TableLayoutPanel();
            this.tableLayoutPanel14 = new System.Windows.Forms.TableLayoutPanel();
            this.lbl_sim_time = new System.Windows.Forms.Label();
            this.label19 = new System.Windows.Forms.Label();
            this.tableLayoutPanel8 = new System.Windows.Forms.TableLayoutPanel();
            this.label14 = new System.Windows.Forms.Label();
            this.lbl_real_time = new System.Windows.Forms.Label();
            this.tableLayoutPanel10 = new System.Windows.Forms.TableLayoutPanel();
            this.lbl_itr = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.tableLayoutPanel5 = new System.Windows.Forms.TableLayoutPanel();
            this.label22 = new System.Windows.Forms.Label();
            this.lbl_alive_axons_perc = new System.Windows.Forms.Label();
            this.tableLayoutPanel7 = new System.Windows.Forms.TableLayoutPanel();
            this.label3 = new System.Windows.Forms.Label();
            this.lbl_itr_s = new System.Windows.Forms.Label();
            this.tableLayoutPanel12 = new System.Windows.Forms.TableLayoutPanel();
            this.tableLayoutPanel6 = new System.Windows.Forms.TableLayoutPanel();
            this.label2 = new System.Windows.Forms.Label();
            this.lbl_tox = new System.Windows.Forms.Label();
            this.tableLayoutPanel13 = new System.Windows.Forms.TableLayoutPanel();
            this.lbl_rem_time = new System.Windows.Forms.Label();
            this.label18 = new System.Windows.Forms.Label();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.statlbl = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel3 = new System.Windows.Forms.ToolStripStatusLabel();
            this.statlbl_sweep = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.tableLayoutPanel11 = new System.Windows.Forms.TableLayoutPanel();
            this.label21 = new System.Windows.Forms.Label();
            this.label20 = new System.Windows.Forms.Label();
            this.cmb_sw_sel2 = new System.Windows.Forms.ComboBox();
            this.txt_delay_ms = new System.Windows.Forms.TextBox();
            this.label13 = new System.Windows.Forms.Label();
            this.btn_snapshot = new System.Windows.Forms.Button();
            this.txt_stop_itr = new System.Windows.Forms.TextBox();
            this.label16 = new System.Windows.Forms.Label();
            this.txt_stop_time = new System.Windows.Forms.TextBox();
            this.label24 = new System.Windows.Forms.Label();
            this.txt_block_siz = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.groupBox8 = new System.Windows.Forms.GroupBox();
            this.btn_clr = new System.Windows.Forms.Button();
            this.txt_status = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.picB)).BeginInit();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox7.SuspendLayout();
            this.tableLayoutPanel2.SuspendLayout();
            this.groupBox6.SuspendLayout();
            this.tableLayoutPanel15.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.tableLayoutPanel4.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.tableLayoutPanel9.SuspendLayout();
            this.tableLayoutPanel14.SuspendLayout();
            this.tableLayoutPanel8.SuspendLayout();
            this.tableLayoutPanel10.SuspendLayout();
            this.tableLayoutPanel5.SuspendLayout();
            this.tableLayoutPanel7.SuspendLayout();
            this.tableLayoutPanel6.SuspendLayout();
            this.tableLayoutPanel13.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.tableLayoutPanel11.SuspendLayout();
            this.groupBox8.SuspendLayout();
            this.SuspendLayout();
            // 
            // picB
            // 
            this.picB.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.picB.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            this.picB.Location = new System.Drawing.Point(16, 196);
            this.picB.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.picB.Name = "picB";
            this.picB.Size = new System.Drawing.Size(721, 529);
            this.picB.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.picB.TabIndex = 0;
            this.picB.TabStop = false;
            this.picB.Resize += new System.EventHandler(this.picB_Resize);
            // 
            // txt_sw_range1
            // 
            this.txt_sw_range1.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.txt_sw_range1.Cue = null;
            this.txt_sw_range1.Location = new System.Drawing.Point(274, 6);
            this.txt_sw_range1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_sw_range1.Name = "txt_sw_range1";
            this.txt_sw_range1.Size = new System.Drawing.Size(111, 22);
            this.txt_sw_range1.TabIndex = 24;
            // 
            // txt_sw_range2
            // 
            this.txt_sw_range2.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.txt_sw_range2.Cue = null;
            this.txt_sw_range2.Location = new System.Drawing.Point(274, 41);
            this.txt_sw_range2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_sw_range2.Name = "txt_sw_range2";
            this.txt_sw_range2.Size = new System.Drawing.Size(111, 22);
            this.txt_sw_range2.TabIndex = 39;
            // 
            // btn_start
            // 
            this.btn_start.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.btn_start.Location = new System.Drawing.Point(8, 4);
            this.btn_start.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_start.Name = "btn_start";
            this.btn_start.Size = new System.Drawing.Size(108, 27);
            this.btn_start.TabIndex = 2;
            this.btn_start.Text = "&Start";
            this.btn_start.UseVisualStyleBackColor = true;
            // 
            // btn_reset
            // 
            this.btn_reset.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.btn_reset.Location = new System.Drawing.Point(134, 4);
            this.btn_reset.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_reset.Name = "btn_reset";
            this.btn_reset.Size = new System.Drawing.Size(105, 27);
            this.btn_reset.TabIndex = 5;
            this.btn_reset.Text = "&Reset";
            this.btn_reset.UseVisualStyleBackColor = true;
            // 
            // btn_save_model
            // 
            this.btn_save_model.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.btn_save_model.Location = new System.Drawing.Point(35, 117);
            this.btn_save_model.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_save_model.Name = "btn_save_model";
            this.btn_save_model.Size = new System.Drawing.Size(105, 36);
            this.btn_save_model.TabIndex = 9;
            this.btn_save_model.Text = "Save Model";
            this.btn_save_model.UseVisualStyleBackColor = true;
            // 
            // btn_load_model
            // 
            this.btn_load_model.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.btn_load_model.Location = new System.Drawing.Point(147, 117);
            this.btn_load_model.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_load_model.Name = "btn_load_model";
            this.btn_load_model.Size = new System.Drawing.Size(105, 36);
            this.btn_load_model.TabIndex = 10;
            this.btn_load_model.Text = "Load Model";
            this.btn_load_model.UseVisualStyleBackColor = true;
            // 
            // btn_redraw
            // 
            this.btn_redraw.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.btn_redraw.Location = new System.Drawing.Point(257, 117);
            this.btn_redraw.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_redraw.Name = "btn_redraw";
            this.btn_redraw.Size = new System.Drawing.Size(89, 36);
            this.btn_redraw.TabIndex = 11;
            this.btn_redraw.Text = "Generate";
            this.btn_redraw.UseVisualStyleBackColor = true;
            // 
            // label4
            // 
            this.label4.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.label4.AutoSize = true;
            this.label4.ForeColor = System.Drawing.Color.Maroon;
            this.label4.Location = new System.Drawing.Point(37, 90);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(99, 17);
            this.label4.TabIndex = 19;
            this.label4.Text = "Num of Axons:";
            // 
            // lbl_num_axons
            // 
            this.lbl_num_axons.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.lbl_num_axons.AutoSize = true;
            this.lbl_num_axons.ForeColor = System.Drawing.Color.Maroon;
            this.lbl_num_axons.Location = new System.Drawing.Point(144, 90);
            this.lbl_num_axons.Name = "lbl_num_axons";
            this.lbl_num_axons.Size = new System.Drawing.Size(16, 17);
            this.lbl_num_axons.TabIndex = 18;
            this.lbl_num_axons.Text = "0";
            // 
            // groupBox1
            // 
            this.groupBox1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox1.Controls.Add(this.label15);
            this.groupBox1.Controls.Add(this.txt_clearance);
            this.groupBox1.Controls.Add(this.txt_nerve_scale);
            this.groupBox1.Controls.Add(this.label11);
            this.groupBox1.Controls.Add(this.lbl_nerve_siz);
            this.groupBox1.Controls.Add(this.chk_strict_rad);
            this.groupBox1.Controls.Add(this.btn_redraw);
            this.groupBox1.Controls.Add(this.btn_save_model);
            this.groupBox1.Controls.Add(this.btn_load_model);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.lbl_num_axons);
            this.groupBox1.Location = new System.Drawing.Point(743, 12);
            this.groupBox1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox1.Size = new System.Drawing.Size(373, 159);
            this.groupBox1.TabIndex = 25;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Model";
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(169, 61);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(93, 17);
            this.label15.TabIndex = 44;
            this.label15.Text = "clearance um";
            // 
            // txt_clearance
            // 
            this.txt_clearance.Location = new System.Drawing.Point(267, 60);
            this.txt_clearance.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_clearance.Name = "txt_clearance";
            this.txt_clearance.Size = new System.Drawing.Size(52, 22);
            this.txt_clearance.TabIndex = 43;
            this.txt_clearance.Text = "0";
            // 
            // txt_nerve_scale
            // 
            this.txt_nerve_scale.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.txt_nerve_scale.Location = new System.Drawing.Point(154, 26);
            this.txt_nerve_scale.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_nerve_scale.Name = "txt_nerve_scale";
            this.txt_nerve_scale.Size = new System.Drawing.Size(68, 22);
            this.txt_nerve_scale.TabIndex = 39;
            // 
            // label11
            // 
            this.label11.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(37, 28);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(111, 17);
            this.label11.TabIndex = 38;
            this.label11.Text = "Nerve Scale (%)";
            // 
            // lbl_nerve_siz
            // 
            this.lbl_nerve_siz.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.lbl_nerve_siz.AutoSize = true;
            this.lbl_nerve_siz.ForeColor = System.Drawing.Color.Maroon;
            this.lbl_nerve_siz.Location = new System.Drawing.Point(244, 27);
            this.lbl_nerve_siz.Name = "lbl_nerve_siz";
            this.lbl_nerve_siz.Size = new System.Drawing.Size(39, 17);
            this.lbl_nerve_siz.TabIndex = 37;
            this.lbl_nerve_siz.Text = "0 um";
            // 
            // chk_strict_rad
            // 
            this.chk_strict_rad.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.chk_strict_rad.AutoSize = true;
            this.chk_strict_rad.ForeColor = System.Drawing.SystemColors.HotTrack;
            this.chk_strict_rad.Location = new System.Drawing.Point(40, 57);
            this.chk_strict_rad.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.chk_strict_rad.Name = "chk_strict_rad";
            this.chk_strict_rad.Size = new System.Drawing.Size(110, 21);
            this.chk_strict_rad.TabIndex = 34;
            this.chk_strict_rad.Text = "Strict Raduis";
            this.chk_strict_rad.UseVisualStyleBackColor = false;
            // 
            // groupBox2
            // 
            this.groupBox2.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox2.Controls.Add(this.txt_on_death_tox);
            this.groupBox2.Controls.Add(this.label10);
            this.groupBox2.Controls.Add(this.txt_insult_tox);
            this.groupBox2.Controls.Add(this.label6);
            this.groupBox2.Controls.Add(this.txt_tox_prod_rate);
            this.groupBox2.Controls.Add(this.label29);
            this.groupBox2.Controls.Add(this.label36);
            this.groupBox2.Controls.Add(this.txt_death_tox_threshold);
            this.groupBox2.Controls.Add(this.groupBox7);
            this.groupBox2.Controls.Add(this.groupBox6);
            this.groupBox2.Controls.Add(this.txt_resolution);
            this.groupBox2.Controls.Add(this.label9);
            this.groupBox2.Controls.Add(this.label12);
            this.groupBox2.Controls.Add(this.lbl_image_siz);
            this.groupBox2.Controls.Add(this.btn_preprocess);
            this.groupBox2.Controls.Add(this.btn_load_setts);
            this.groupBox2.Controls.Add(this.btn_save_setts);
            this.groupBox2.Location = new System.Drawing.Point(743, 184);
            this.groupBox2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox2.Size = new System.Drawing.Size(373, 459);
            this.groupBox2.TabIndex = 26;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Settings";
            // 
            // txt_on_death_tox
            // 
            this.txt_on_death_tox.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.txt_on_death_tox.Location = new System.Drawing.Point(267, 285);
            this.txt_on_death_tox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_on_death_tox.Name = "txt_on_death_tox";
            this.txt_on_death_tox.Size = new System.Drawing.Size(65, 22);
            this.txt_on_death_tox.TabIndex = 46;
            // 
            // label10
            // 
            this.label10.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(58, 288);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(181, 17);
            this.label10.TabIndex = 45;
            this.label10.Text = "On death [micromole/um^2]";
            // 
            // txt_insult_tox
            // 
            this.txt_insult_tox.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.txt_insult_tox.Location = new System.Drawing.Point(267, 255);
            this.txt_insult_tox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_insult_tox.Name = "txt_insult_tox";
            this.txt_insult_tox.Size = new System.Drawing.Size(65, 22);
            this.txt_insult_tox.TabIndex = 44;
            // 
            // label6
            // 
            this.label6.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(84, 258);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(155, 17);
            this.label6.TabIndex = 43;
            this.label6.Text = "Insult [micromole/um^2]";
            // 
            // txt_tox_prod_rate
            // 
            this.txt_tox_prod_rate.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.txt_tox_prod_rate.Location = new System.Drawing.Point(267, 227);
            this.txt_tox_prod_rate.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_tox_prod_rate.Name = "txt_tox_prod_rate";
            this.txt_tox_prod_rate.Size = new System.Drawing.Size(65, 22);
            this.txt_tox_prod_rate.TabIndex = 42;
            // 
            // label29
            // 
            this.label29.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label29.AutoSize = true;
            this.label29.Location = new System.Drawing.Point(61, 230);
            this.label29.Name = "label29";
            this.label29.Size = new System.Drawing.Size(178, 17);
            this.label29.TabIndex = 41;
            this.label29.Text = "Production [micromole/sec]";
            // 
            // label36
            // 
            this.label36.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label36.AutoSize = true;
            this.label36.Location = new System.Drawing.Point(54, 202);
            this.label36.Name = "label36";
            this.label36.Size = new System.Drawing.Size(186, 17);
            this.label36.TabIndex = 39;
            this.label36.Text = "Death Thr [micromole/um^2]";
            // 
            // txt_death_tox_threshold
            // 
            this.txt_death_tox_threshold.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.txt_death_tox_threshold.Location = new System.Drawing.Point(267, 200);
            this.txt_death_tox_threshold.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_death_tox_threshold.Name = "txt_death_tox_threshold";
            this.txt_death_tox_threshold.Size = new System.Drawing.Size(65, 22);
            this.txt_death_tox_threshold.TabIndex = 40;
            // 
            // groupBox7
            // 
            this.groupBox7.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox7.Controls.Add(this.tableLayoutPanel2);
            this.groupBox7.Cursor = System.Windows.Forms.Cursors.Default;
            this.groupBox7.Location = new System.Drawing.Point(8, 52);
            this.groupBox7.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox7.Name = "groupBox7";
            this.groupBox7.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox7.Size = new System.Drawing.Size(359, 135);
            this.groupBox7.TabIndex = 35;
            this.groupBox7.TabStop = false;
            this.groupBox7.Text = "Diffusion Rates";
            // 
            // tableLayoutPanel2
            // 
            this.tableLayoutPanel2.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.tableLayoutPanel2.ColumnCount = 3;
            this.tableLayoutPanel2.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 68.12749F));
            this.tableLayoutPanel2.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 31.87251F));
            this.tableLayoutPanel2.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 99F));
            this.tableLayoutPanel2.Controls.Add(this.label26, 0, 3);
            this.tableLayoutPanel2.Controls.Add(this.txt_rate_live, 1, 0);
            this.tableLayoutPanel2.Controls.Add(this.txt_rate_bound, 1, 2);
            this.tableLayoutPanel2.Controls.Add(this.label7, 0, 1);
            this.tableLayoutPanel2.Controls.Add(this.label8, 0, 2);
            this.tableLayoutPanel2.Controls.Add(this.txt_rate_dead, 1, 1);
            this.tableLayoutPanel2.Controls.Add(this.label25, 0, 0);
            this.tableLayoutPanel2.Controls.Add(this.txt_rate_extra, 1, 3);
            this.tableLayoutPanel2.Controls.Add(this.label30, 2, 0);
            this.tableLayoutPanel2.Controls.Add(this.label31, 2, 1);
            this.tableLayoutPanel2.Controls.Add(this.label32, 2, 2);
            this.tableLayoutPanel2.Controls.Add(this.label33, 2, 3);
            this.tableLayoutPanel2.Location = new System.Drawing.Point(30, 21);
            this.tableLayoutPanel2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel2.Name = "tableLayoutPanel2";
            this.tableLayoutPanel2.RowCount = 4;
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 25F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 25F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 25F));
            this.tableLayoutPanel2.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 25F));
            this.tableLayoutPanel2.Size = new System.Drawing.Size(308, 114);
            this.tableLayoutPanel2.TabIndex = 12;
            // 
            // label26
            // 
            this.label26.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label26.AutoSize = true;
            this.label26.Location = new System.Drawing.Point(8, 90);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(131, 17);
            this.label26.TabIndex = 33;
            this.label26.Text = "Extracellular [1/sec]";
            // 
            // txt_rate_live
            // 
            this.txt_rate_live.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_rate_live.Location = new System.Drawing.Point(145, 3);
            this.txt_rate_live.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_rate_live.Name = "txt_rate_live";
            this.txt_rate_live.Size = new System.Drawing.Size(57, 22);
            this.txt_rate_live.TabIndex = 32;
            // 
            // txt_rate_bound
            // 
            this.txt_rate_bound.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_rate_bound.Location = new System.Drawing.Point(145, 59);
            this.txt_rate_bound.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_rate_bound.Name = "txt_rate_bound";
            this.txt_rate_bound.Size = new System.Drawing.Size(57, 22);
            this.txt_rate_bound.TabIndex = 27;
            // 
            // label7
            // 
            this.label7.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(16, 33);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(123, 17);
            this.label7.TabIndex = 20;
            this.label7.Text = "Dead Axon [1/sec]";
            // 
            // label8
            // 
            this.label8.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(24, 61);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(115, 17);
            this.label8.TabIndex = 26;
            this.label8.Text = "Boundary [1/sec]";
            // 
            // txt_rate_dead
            // 
            this.txt_rate_dead.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_rate_dead.Location = new System.Drawing.Point(145, 31);
            this.txt_rate_dead.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_rate_dead.Name = "txt_rate_dead";
            this.txt_rate_dead.Size = new System.Drawing.Size(57, 22);
            this.txt_rate_dead.TabIndex = 21;
            // 
            // label25
            // 
            this.label25.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label25.AutoSize = true;
            this.label25.Location = new System.Drawing.Point(24, 5);
            this.label25.Name = "label25";
            this.label25.Size = new System.Drawing.Size(115, 17);
            this.label25.TabIndex = 29;
            this.label25.Text = "Live Axon [1/sec]";
            // 
            // txt_rate_extra
            // 
            this.txt_rate_extra.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_rate_extra.Location = new System.Drawing.Point(145, 88);
            this.txt_rate_extra.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_rate_extra.Name = "txt_rate_extra";
            this.txt_rate_extra.Size = new System.Drawing.Size(57, 22);
            this.txt_rate_extra.TabIndex = 34;
            // 
            // label30
            // 
            this.label30.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label30.AutoSize = true;
            this.label30.Location = new System.Drawing.Point(267, 5);
            this.label30.Name = "label30";
            this.label30.Size = new System.Drawing.Size(38, 17);
            this.label30.TabIndex = 35;
            this.label30.Text = "0→1";
            // 
            // label31
            // 
            this.label31.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label31.AutoSize = true;
            this.label31.Location = new System.Drawing.Point(267, 33);
            this.label31.Name = "label31";
            this.label31.Size = new System.Drawing.Size(38, 17);
            this.label31.TabIndex = 36;
            this.label31.Text = "0→1";
            // 
            // label32
            // 
            this.label32.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label32.AutoSize = true;
            this.label32.Location = new System.Drawing.Point(267, 61);
            this.label32.Name = "label32";
            this.label32.Size = new System.Drawing.Size(38, 17);
            this.label32.TabIndex = 37;
            this.label32.Text = "0→1";
            // 
            // label33
            // 
            this.label33.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label33.AutoSize = true;
            this.label33.Location = new System.Drawing.Point(267, 90);
            this.label33.Name = "label33";
            this.label33.Size = new System.Drawing.Size(38, 17);
            this.label33.TabIndex = 38;
            this.label33.Text = "0→1";
            // 
            // groupBox6
            // 
            this.groupBox6.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox6.Controls.Add(this.tableLayoutPanel15);
            this.groupBox6.Location = new System.Drawing.Point(0, 320);
            this.groupBox6.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox6.Name = "groupBox6";
            this.groupBox6.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox6.Size = new System.Drawing.Size(367, 85);
            this.groupBox6.TabIndex = 32;
            this.groupBox6.TabStop = false;
            this.groupBox6.Text = "Detoxification Rates";
            // 
            // tableLayoutPanel15
            // 
            this.tableLayoutPanel15.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.tableLayoutPanel15.ColumnCount = 3;
            this.tableLayoutPanel15.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 67.53247F));
            this.tableLayoutPanel15.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 32.46753F));
            this.tableLayoutPanel15.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 101F));
            this.tableLayoutPanel15.Controls.Add(this.txt_detox_extra, 1, 1);
            this.tableLayoutPanel15.Controls.Add(this.label27, 0, 0);
            this.tableLayoutPanel15.Controls.Add(this.label28, 0, 1);
            this.tableLayoutPanel15.Controls.Add(this.txt_detox_intra, 1, 0);
            this.tableLayoutPanel15.Controls.Add(this.label34, 2, 0);
            this.tableLayoutPanel15.Controls.Add(this.label35, 2, 1);
            this.tableLayoutPanel15.Location = new System.Drawing.Point(35, 23);
            this.tableLayoutPanel15.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel15.Name = "tableLayoutPanel15";
            this.tableLayoutPanel15.RowCount = 2;
            this.tableLayoutPanel15.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel15.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel15.Size = new System.Drawing.Size(311, 51);
            this.tableLayoutPanel15.TabIndex = 12;
            // 
            // txt_detox_extra
            // 
            this.txt_detox_extra.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_detox_extra.Location = new System.Drawing.Point(144, 27);
            this.txt_detox_extra.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_detox_extra.Name = "txt_detox_extra";
            this.txt_detox_extra.Size = new System.Drawing.Size(54, 22);
            this.txt_detox_extra.TabIndex = 27;
            // 
            // label27
            // 
            this.label27.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label27.AutoSize = true;
            this.label27.Location = new System.Drawing.Point(16, 4);
            this.label27.Name = "label27";
            this.label27.Size = new System.Drawing.Size(122, 17);
            this.label27.TabIndex = 20;
            this.label27.Text = "Detox Intra [1/sec]";
            // 
            // label28
            // 
            this.label28.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label28.AutoSize = true;
            this.label28.Location = new System.Drawing.Point(12, 29);
            this.label28.Name = "label28";
            this.label28.Size = new System.Drawing.Size(126, 17);
            this.label28.TabIndex = 26;
            this.label28.Text = "Detox Extra [1/sec]";
            // 
            // txt_detox_intra
            // 
            this.txt_detox_intra.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.txt_detox_intra.Location = new System.Drawing.Point(144, 2);
            this.txt_detox_intra.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_detox_intra.Name = "txt_detox_intra";
            this.txt_detox_intra.Size = new System.Drawing.Size(54, 22);
            this.txt_detox_intra.TabIndex = 21;
            // 
            // label34
            // 
            this.label34.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label34.AutoSize = true;
            this.label34.Location = new System.Drawing.Point(270, 4);
            this.label34.Name = "label34";
            this.label34.Size = new System.Drawing.Size(38, 17);
            this.label34.TabIndex = 36;
            this.label34.Text = "0→1";
            // 
            // label35
            // 
            this.label35.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label35.AutoSize = true;
            this.label35.Location = new System.Drawing.Point(270, 29);
            this.label35.Name = "label35";
            this.label35.Size = new System.Drawing.Size(38, 17);
            this.label35.TabIndex = 37;
            this.label35.Text = "0→1";
            // 
            // txt_resolution
            // 
            this.txt_resolution.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.txt_resolution.Location = new System.Drawing.Point(135, 22);
            this.txt_resolution.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_resolution.Name = "txt_resolution";
            this.txt_resolution.Size = new System.Drawing.Size(68, 22);
            this.txt_resolution.TabIndex = 34;
            // 
            // label9
            // 
            this.label9.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(43, 25);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(87, 17);
            this.label9.TabIndex = 33;
            this.label9.Text = "Res (pix/um)";
            // 
            // label12
            // 
            this.label12.Anchor = System.Windows.Forms.AnchorStyles.Top;
            this.label12.AutoSize = true;
            this.label12.ForeColor = System.Drawing.Color.Maroon;
            this.label12.Location = new System.Drawing.Point(213, 25);
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
            this.lbl_image_siz.Location = new System.Drawing.Point(301, 25);
            this.lbl_image_siz.Name = "lbl_image_siz";
            this.lbl_image_siz.Size = new System.Drawing.Size(13, 17);
            this.lbl_image_siz.TabIndex = 30;
            this.lbl_image_siz.Text = "-";
            // 
            // btn_preprocess
            // 
            this.btn_preprocess.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.btn_preprocess.Location = new System.Drawing.Point(256, 416);
            this.btn_preprocess.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_preprocess.Name = "btn_preprocess";
            this.btn_preprocess.Size = new System.Drawing.Size(89, 36);
            this.btn_preprocess.TabIndex = 29;
            this.btn_preprocess.Text = "Preprocess";
            this.btn_preprocess.UseVisualStyleBackColor = true;
            // 
            // btn_load_setts
            // 
            this.btn_load_setts.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.btn_load_setts.Location = new System.Drawing.Point(145, 416);
            this.btn_load_setts.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_load_setts.Name = "btn_load_setts";
            this.btn_load_setts.Size = new System.Drawing.Size(105, 36);
            this.btn_load_setts.TabIndex = 29;
            this.btn_load_setts.Text = "Load Settings";
            this.btn_load_setts.UseVisualStyleBackColor = true;
            // 
            // btn_save_setts
            // 
            this.btn_save_setts.Anchor = System.Windows.Forms.AnchorStyles.Bottom;
            this.btn_save_setts.Location = new System.Drawing.Point(35, 416);
            this.btn_save_setts.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_save_setts.Name = "btn_save_setts";
            this.btn_save_setts.Size = new System.Drawing.Size(105, 36);
            this.btn_save_setts.TabIndex = 20;
            this.btn_save_setts.Text = "Save Settings";
            this.btn_save_setts.UseVisualStyleBackColor = true;
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.tableLayoutPanel4);
            this.groupBox3.Location = new System.Drawing.Point(16, 12);
            this.groupBox3.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox3.Size = new System.Drawing.Size(405, 66);
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
            this.tableLayoutPanel4.Controls.Add(this.btn_save_prog, 2, 0);
            this.tableLayoutPanel4.Controls.Add(this.btn_start, 0, 0);
            this.tableLayoutPanel4.Controls.Add(this.btn_reset, 1, 0);
            this.tableLayoutPanel4.Location = new System.Drawing.Point(12, 25);
            this.tableLayoutPanel4.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel4.Name = "tableLayoutPanel4";
            this.tableLayoutPanel4.RowCount = 1;
            this.tableLayoutPanel4.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel4.Size = new System.Drawing.Size(395, 36);
            this.tableLayoutPanel4.TabIndex = 24;
            // 
            // btn_save_prog
            // 
            this.btn_save_prog.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.btn_save_prog.Location = new System.Drawing.Point(260, 4);
            this.btn_save_prog.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_save_prog.Name = "btn_save_prog";
            this.btn_save_prog.Size = new System.Drawing.Size(124, 27);
            this.btn_save_prog.TabIndex = 43;
            this.btn_save_prog.Text = "Save Progress";
            this.btn_save_prog.UseVisualStyleBackColor = true;
            // 
            // btn_sweep
            // 
            this.btn_sweep.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.btn_sweep.Location = new System.Drawing.Point(12, 4);
            this.btn_sweep.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
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
            this.cmb_sw_sel1.Location = new System.Drawing.Point(183, 5);
            this.cmb_sw_sel1.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.cmb_sw_sel1.Name = "cmb_sw_sel1";
            this.cmb_sw_sel1.Size = new System.Drawing.Size(79, 24);
            this.cmb_sw_sel1.TabIndex = 34;
            // 
            // chk_save_sw_prog
            // 
            this.chk_save_sw_prog.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.chk_save_sw_prog.Appearance = System.Windows.Forms.Appearance.Button;
            this.chk_save_sw_prog.AutoSize = true;
            this.chk_save_sw_prog.ForeColor = System.Drawing.SystemColors.HotTrack;
            this.chk_save_sw_prog.Location = new System.Drawing.Point(18, 39);
            this.chk_save_sw_prog.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.chk_save_sw_prog.Name = "chk_save_sw_prog";
            this.chk_save_sw_prog.Size = new System.Drawing.Size(96, 27);
            this.chk_save_sw_prog.TabIndex = 37;
            this.chk_save_sw_prog.Text = "Save Sweep";
            this.chk_save_sw_prog.UseVisualStyleBackColor = true;
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.txt_rec_inerval);
            this.groupBox4.Controls.Add(this.chk_show_axons);
            this.groupBox4.Controls.Add(this.label17);
            this.groupBox4.Controls.Add(this.chk_show_tox);
            this.groupBox4.Controls.Add(this.chk_rec_avi);
            this.groupBox4.Location = new System.Drawing.Point(428, 12);
            this.groupBox4.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox4.Size = new System.Drawing.Size(309, 98);
            this.groupBox4.TabIndex = 28;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Display";
            // 
            // txt_rec_inerval
            // 
            this.txt_rec_inerval.Location = new System.Drawing.Point(100, 61);
            this.txt_rec_inerval.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_rec_inerval.Name = "txt_rec_inerval";
            this.txt_rec_inerval.Size = new System.Drawing.Size(52, 22);
            this.txt_rec_inerval.TabIndex = 45;
            this.txt_rec_inerval.Text = "0.5";
            // 
            // chk_show_axons
            // 
            this.chk_show_axons.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.chk_show_axons.AutoSize = true;
            this.chk_show_axons.Checked = true;
            this.chk_show_axons.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chk_show_axons.Location = new System.Drawing.Point(196, 19);
            this.chk_show_axons.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.chk_show_axons.Name = "chk_show_axons";
            this.chk_show_axons.Size = new System.Drawing.Size(106, 21);
            this.chk_show_axons.TabIndex = 26;
            this.chk_show_axons.Text = "Show Axons";
            this.chk_show_axons.UseVisualStyleBackColor = true;
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(12, 57);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(86, 34);
            this.label17.TabIndex = 44;
            this.label17.Text = "Period [Real\r\nTime Unit]";
            // 
            // chk_show_tox
            // 
            this.chk_show_tox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.chk_show_tox.AutoSize = true;
            this.chk_show_tox.Checked = true;
            this.chk_show_tox.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chk_show_tox.Location = new System.Drawing.Point(196, 52);
            this.chk_show_tox.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.chk_show_tox.Name = "chk_show_tox";
            this.chk_show_tox.Size = new System.Drawing.Size(102, 21);
            this.chk_show_tox.TabIndex = 27;
            this.chk_show_tox.Text = "Show Toxin";
            this.chk_show_tox.UseVisualStyleBackColor = true;
            // 
            // chk_rec_avi
            // 
            this.chk_rec_avi.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Left | System.Windows.Forms.AnchorStyles.Right)));
            this.chk_rec_avi.Appearance = System.Windows.Forms.Appearance.Button;
            this.chk_rec_avi.AutoSize = true;
            this.chk_rec_avi.ForeColor = System.Drawing.SystemColors.HotTrack;
            this.chk_rec_avi.Location = new System.Drawing.Point(30, 25);
            this.chk_rec_avi.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.chk_rec_avi.Name = "chk_rec_avi";
            this.chk_rec_avi.Size = new System.Drawing.Size(64, 27);
            this.chk_rec_avi.TabIndex = 25;
            this.chk_rec_avi.Text = "Record";
            this.chk_rec_avi.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.chk_rec_avi.UseVisualStyleBackColor = true;
            // 
            // tableLayoutPanel9
            // 
            this.tableLayoutPanel9.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.tableLayoutPanel9.ColumnCount = 4;
            this.tableLayoutPanel9.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 22.88488F));
            this.tableLayoutPanel9.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 24.27184F));
            this.tableLayoutPanel9.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 25.9362F));
            this.tableLayoutPanel9.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 27.18447F));
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel14, 0, 1);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel8, 0, 0);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel10, 0, 0);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel5, 0, 0);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel7, 0, 1);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel12, 3, 0);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel6, 1, 1);
            this.tableLayoutPanel9.Controls.Add(this.tableLayoutPanel13, 3, 1);
            this.tableLayoutPanel9.Location = new System.Drawing.Point(16, 731);
            this.tableLayoutPanel9.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel9.Name = "tableLayoutPanel9";
            this.tableLayoutPanel9.RowCount = 2;
            this.tableLayoutPanel9.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel9.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel9.Size = new System.Drawing.Size(721, 49);
            this.tableLayoutPanel9.TabIndex = 29;
            // 
            // tableLayoutPanel14
            // 
            this.tableLayoutPanel14.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel14.ColumnCount = 2;
            this.tableLayoutPanel14.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 54.16667F));
            this.tableLayoutPanel14.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 45.83333F));
            this.tableLayoutPanel14.Controls.Add(this.lbl_sim_time, 1, 0);
            this.tableLayoutPanel14.Controls.Add(this.label19, 0, 0);
            this.tableLayoutPanel14.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel14.Location = new System.Drawing.Point(167, 26);
            this.tableLayoutPanel14.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel14.Name = "tableLayoutPanel14";
            this.tableLayoutPanel14.RowCount = 1;
            this.tableLayoutPanel14.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel14.Size = new System.Drawing.Size(168, 18);
            this.tableLayoutPanel14.TabIndex = 40;
            // 
            // lbl_sim_time
            // 
            this.lbl_sim_time.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_sim_time.AutoSize = true;
            this.lbl_sim_time.Location = new System.Drawing.Point(94, 0);
            this.lbl_sim_time.Name = "lbl_sim_time";
            this.lbl_sim_time.Size = new System.Drawing.Size(16, 17);
            this.lbl_sim_time.TabIndex = 3;
            this.lbl_sim_time.Text = "0";
            // 
            // label19
            // 
            this.label19.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(22, 0);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(66, 17);
            this.label19.TabIndex = 7;
            this.label19.Text = "Sim Time";
            // 
            // tableLayoutPanel8
            // 
            this.tableLayoutPanel8.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel8.ColumnCount = 2;
            this.tableLayoutPanel8.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 54.16667F));
            this.tableLayoutPanel8.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 45.83333F));
            this.tableLayoutPanel8.Controls.Add(this.label14, 0, 0);
            this.tableLayoutPanel8.Controls.Add(this.lbl_real_time, 1, 0);
            this.tableLayoutPanel8.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel8.Location = new System.Drawing.Point(167, 2);
            this.tableLayoutPanel8.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel8.Name = "tableLayoutPanel8";
            this.tableLayoutPanel8.RowCount = 1;
            this.tableLayoutPanel8.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel8.Size = new System.Drawing.Size(168, 18);
            this.tableLayoutPanel8.TabIndex = 36;
            // 
            // label14
            // 
            this.label14.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(21, 0);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(67, 17);
            this.label14.TabIndex = 16;
            this.label14.Text = "Real time";
            // 
            // lbl_real_time
            // 
            this.lbl_real_time.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_real_time.AutoSize = true;
            this.lbl_real_time.Location = new System.Drawing.Point(94, 0);
            this.lbl_real_time.Name = "lbl_real_time";
            this.lbl_real_time.Size = new System.Drawing.Size(16, 17);
            this.lbl_real_time.TabIndex = 15;
            this.lbl_real_time.Text = "0";
            // 
            // tableLayoutPanel10
            // 
            this.tableLayoutPanel10.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel10.ColumnCount = 2;
            this.tableLayoutPanel10.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 56.61765F));
            this.tableLayoutPanel10.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 43.38235F));
            this.tableLayoutPanel10.Controls.Add(this.lbl_itr, 1, 0);
            this.tableLayoutPanel10.Controls.Add(this.label1, 0, 0);
            this.tableLayoutPanel10.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel10.Location = new System.Drawing.Point(3, 2);
            this.tableLayoutPanel10.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel10.Name = "tableLayoutPanel10";
            this.tableLayoutPanel10.RowCount = 1;
            this.tableLayoutPanel10.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel10.Size = new System.Drawing.Size(158, 18);
            this.tableLayoutPanel10.TabIndex = 35;
            // 
            // lbl_itr
            // 
            this.lbl_itr.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_itr.AutoSize = true;
            this.lbl_itr.Location = new System.Drawing.Point(92, 0);
            this.lbl_itr.Name = "lbl_itr";
            this.lbl_itr.Size = new System.Drawing.Size(16, 17);
            this.lbl_itr.TabIndex = 3;
            this.lbl_itr.Text = "0";
            // 
            // label1
            // 
            this.label1.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(23, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(63, 17);
            this.label1.TabIndex = 7;
            this.label1.Text = "Iteration:";
            // 
            // tableLayoutPanel5
            // 
            this.tableLayoutPanel5.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel5.ColumnCount = 2;
            this.tableLayoutPanel5.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel5.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel5.Controls.Add(this.label22, 0, 0);
            this.tableLayoutPanel5.Controls.Add(this.lbl_alive_axons_perc, 1, 0);
            this.tableLayoutPanel5.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel5.Location = new System.Drawing.Point(341, 2);
            this.tableLayoutPanel5.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel5.Name = "tableLayoutPanel5";
            this.tableLayoutPanel5.RowCount = 1;
            this.tableLayoutPanel5.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel5.Size = new System.Drawing.Size(180, 18);
            this.tableLayoutPanel5.TabIndex = 32;
            // 
            // label22
            // 
            this.label22.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label22.AutoSize = true;
            this.label22.Location = new System.Drawing.Point(14, 0);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(73, 18);
            this.label22.TabIndex = 8;
            this.label22.Text = "live neurs (%):";
            // 
            // lbl_alive_axons_perc
            // 
            this.lbl_alive_axons_perc.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_alive_axons_perc.AutoSize = true;
            this.lbl_alive_axons_perc.Location = new System.Drawing.Point(93, 0);
            this.lbl_alive_axons_perc.Name = "lbl_alive_axons_perc";
            this.lbl_alive_axons_perc.Size = new System.Drawing.Size(16, 17);
            this.lbl_alive_axons_perc.TabIndex = 1;
            this.lbl_alive_axons_perc.Text = "0";
            // 
            // tableLayoutPanel7
            // 
            this.tableLayoutPanel7.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tableLayoutPanel7.ColumnCount = 2;
            this.tableLayoutPanel7.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 56.61765F));
            this.tableLayoutPanel7.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 43.38235F));
            this.tableLayoutPanel7.Controls.Add(this.label3, 0, 0);
            this.tableLayoutPanel7.Controls.Add(this.lbl_itr_s, 1, 0);
            this.tableLayoutPanel7.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel7.Location = new System.Drawing.Point(3, 26);
            this.tableLayoutPanel7.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel7.Name = "tableLayoutPanel7";
            this.tableLayoutPanel7.RowCount = 1;
            this.tableLayoutPanel7.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel7.Size = new System.Drawing.Size(158, 18);
            this.tableLayoutPanel7.TabIndex = 33;
            // 
            // label3
            // 
            this.label3.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(5, 0);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(81, 17);
            this.label3.TabIndex = 16;
            this.label3.Text = "iterations/s:";
            // 
            // lbl_itr_s
            // 
            this.lbl_itr_s.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_itr_s.AutoSize = true;
            this.lbl_itr_s.Location = new System.Drawing.Point(92, 0);
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
            this.tableLayoutPanel12.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 54.50237F));
            this.tableLayoutPanel12.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 45.49763F));
            this.tableLayoutPanel12.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel12.Location = new System.Drawing.Point(527, 2);
            this.tableLayoutPanel12.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel12.Name = "tableLayoutPanel12";
            this.tableLayoutPanel12.RowCount = 1;
            this.tableLayoutPanel12.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel12.Size = new System.Drawing.Size(191, 18);
            this.tableLayoutPanel12.TabIndex = 38;
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
            this.tableLayoutPanel6.Location = new System.Drawing.Point(341, 26);
            this.tableLayoutPanel6.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel6.Name = "tableLayoutPanel6";
            this.tableLayoutPanel6.RowCount = 1;
            this.tableLayoutPanel6.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel6.Size = new System.Drawing.Size(180, 18);
            this.tableLayoutPanel6.TabIndex = 34;
            // 
            // label2
            // 
            this.label2.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(16, 0);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(71, 17);
            this.label2.TabIndex = 8;
            this.label2.Text = "sum toxin:";
            // 
            // lbl_tox
            // 
            this.lbl_tox.Anchor = System.Windows.Forms.AnchorStyles.Left;
            this.lbl_tox.AutoSize = true;
            this.lbl_tox.Location = new System.Drawing.Point(93, 0);
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
            this.tableLayoutPanel13.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 54.9763F));
            this.tableLayoutPanel13.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 45.0237F));
            this.tableLayoutPanel13.Controls.Add(this.lbl_rem_time, 1, 0);
            this.tableLayoutPanel13.Controls.Add(this.label18, 0, 0);
            this.tableLayoutPanel13.ForeColor = System.Drawing.Color.Maroon;
            this.tableLayoutPanel13.Location = new System.Drawing.Point(527, 26);
            this.tableLayoutPanel13.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel13.Name = "tableLayoutPanel13";
            this.tableLayoutPanel13.RowCount = 1;
            this.tableLayoutPanel13.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanel13.Size = new System.Drawing.Size(191, 18);
            this.tableLayoutPanel13.TabIndex = 39;
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
            // label18
            // 
            this.label18.Anchor = System.Windows.Forms.AnchorStyles.Right;
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(27, 0);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(75, 18);
            this.label18.TabIndex = 7;
            this.label18.Text = "Sim Remaining";
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
            this.statusStrip1.Padding = new System.Windows.Forms.Padding(1, 0, 13, 0);
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
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(759, 20);
            this.toolStripStatusLabel1.Spring = true;
            // 
            // toolStripStatusLabel2
            // 
            this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
            this.toolStripStatusLabel2.Size = new System.Drawing.Size(69, 20);
            this.toolStripStatusLabel2.Text = "               ";
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.tableLayoutPanel11);
            this.groupBox5.Location = new System.Drawing.Point(16, 85);
            this.groupBox5.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox5.Size = new System.Drawing.Size(405, 105);
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
            this.tableLayoutPanel11.Location = new System.Drawing.Point(12, 25);
            this.tableLayoutPanel11.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.tableLayoutPanel11.Name = "tableLayoutPanel11";
            this.tableLayoutPanel11.RowCount = 2;
            this.tableLayoutPanel11.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel11.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 33.33333F));
            this.tableLayoutPanel11.Size = new System.Drawing.Size(395, 70);
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
            this.cmb_sw_sel2.Location = new System.Drawing.Point(183, 40);
            this.cmb_sw_sel2.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.cmb_sw_sel2.Name = "cmb_sw_sel2";
            this.cmb_sw_sel2.Size = new System.Drawing.Size(79, 24);
            this.cmb_sw_sel2.TabIndex = 38;
            // 
            // txt_delay_ms
            // 
            this.txt_delay_ms.Location = new System.Drawing.Point(528, 111);
            this.txt_delay_ms.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_delay_ms.Name = "txt_delay_ms";
            this.txt_delay_ms.Size = new System.Drawing.Size(52, 22);
            this.txt_delay_ms.TabIndex = 35;
            this.txt_delay_ms.Text = "0";
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(449, 113);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(72, 17);
            this.label13.TabIndex = 34;
            this.label13.Text = "Delay(ms)";
            // 
            // btn_snapshot
            // 
            this.btn_snapshot.Location = new System.Drawing.Point(627, 108);
            this.btn_snapshot.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_snapshot.Name = "btn_snapshot";
            this.btn_snapshot.Size = new System.Drawing.Size(105, 27);
            this.btn_snapshot.TabIndex = 36;
            this.btn_snapshot.Text = "Snapshot";
            this.btn_snapshot.UseVisualStyleBackColor = true;
            // 
            // txt_stop_itr
            // 
            this.txt_stop_itr.Location = new System.Drawing.Point(528, 163);
            this.txt_stop_itr.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_stop_itr.Name = "txt_stop_itr";
            this.txt_stop_itr.Size = new System.Drawing.Size(52, 22);
            this.txt_stop_itr.TabIndex = 38;
            this.txt_stop_itr.Text = "0";
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(454, 165);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(67, 17);
            this.label16.TabIndex = 37;
            this.label16.Text = "Stop@ Itr";
            // 
            // txt_stop_time
            // 
            this.txt_stop_time.Location = new System.Drawing.Point(678, 163);
            this.txt_stop_time.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_stop_time.Name = "txt_stop_time";
            this.txt_stop_time.Size = new System.Drawing.Size(52, 22);
            this.txt_stop_time.TabIndex = 41;
            this.txt_stop_time.Text = "0";
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.Location = new System.Drawing.Point(451, 139);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(71, 17);
            this.label24.TabIndex = 39;
            this.label24.Text = "Block size";
            // 
            // txt_block_siz
            // 
            this.txt_block_siz.Location = new System.Drawing.Point(528, 137);
            this.txt_block_siz.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_block_siz.Name = "txt_block_siz";
            this.txt_block_siz.Size = new System.Drawing.Size(52, 22);
            this.txt_block_siz.TabIndex = 40;
            this.txt_block_siz.Text = "0";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(596, 165);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(81, 17);
            this.label5.TabIndex = 42;
            this.label5.Text = "Stop@ time";
            // 
            // groupBox8
            // 
            this.groupBox8.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBox8.Controls.Add(this.btn_clr);
            this.groupBox8.Controls.Add(this.txt_status);
            this.groupBox8.Location = new System.Drawing.Point(743, 647);
            this.groupBox8.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox8.Name = "groupBox8";
            this.groupBox8.Padding = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.groupBox8.Size = new System.Drawing.Size(373, 133);
            this.groupBox8.TabIndex = 38;
            this.groupBox8.TabStop = false;
            this.groupBox8.Text = "Output";
            // 
            // btn_clr
            // 
            this.btn_clr.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.btn_clr.Location = new System.Drawing.Point(323, 19);
            this.btn_clr.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.btn_clr.Name = "btn_clr";
            this.btn_clr.Size = new System.Drawing.Size(43, 27);
            this.btn_clr.TabIndex = 34;
            this.btn_clr.Text = "clr";
            this.btn_clr.UseVisualStyleBackColor = true;
            // 
            // txt_status
            // 
            this.txt_status.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.txt_status.Location = new System.Drawing.Point(8, 19);
            this.txt_status.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.txt_status.Multiline = true;
            this.txt_status.Name = "txt_status";
            this.txt_status.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.txt_status.Size = new System.Drawing.Size(357, 110);
            this.txt_status.TabIndex = 33;
            // 
            // Main_Form
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1137, 807);
            this.Controls.Add(this.groupBox8);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.txt_stop_time);
            this.Controls.Add(this.txt_block_siz);
            this.Controls.Add(this.label24);
            this.Controls.Add(this.txt_stop_itr);
            this.Controls.Add(this.label16);
            this.Controls.Add(this.btn_snapshot);
            this.Controls.Add(this.txt_delay_ms);
            this.Controls.Add(this.label13);
            this.Controls.Add(this.groupBox5);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.tableLayoutPanel9);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.picB);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Margin = new System.Windows.Forms.Padding(3, 2, 3, 2);
            this.Name = "Main_Form";
            this.Text = "LHON";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Main_Form_FormClosing);
            this.Load += new System.EventHandler(this.Main_Form_Load);
            ((System.ComponentModel.ISupportInitialize)(this.picB)).EndInit();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox7.ResumeLayout(false);
            this.tableLayoutPanel2.ResumeLayout(false);
            this.tableLayoutPanel2.PerformLayout();
            this.groupBox6.ResumeLayout(false);
            this.tableLayoutPanel15.ResumeLayout(false);
            this.tableLayoutPanel15.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.tableLayoutPanel4.ResumeLayout(false);
            this.groupBox4.ResumeLayout(false);
            this.groupBox4.PerformLayout();
            this.tableLayoutPanel9.ResumeLayout(false);
            this.tableLayoutPanel14.ResumeLayout(false);
            this.tableLayoutPanel14.PerformLayout();
            this.tableLayoutPanel8.ResumeLayout(false);
            this.tableLayoutPanel8.PerformLayout();
            this.tableLayoutPanel10.ResumeLayout(false);
            this.tableLayoutPanel10.PerformLayout();
            this.tableLayoutPanel5.ResumeLayout(false);
            this.tableLayoutPanel5.PerformLayout();
            this.tableLayoutPanel7.ResumeLayout(false);
            this.tableLayoutPanel7.PerformLayout();
            this.tableLayoutPanel6.ResumeLayout(false);
            this.tableLayoutPanel6.PerformLayout();
            this.tableLayoutPanel13.ResumeLayout(false);
            this.tableLayoutPanel13.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.groupBox5.ResumeLayout(false);
            this.tableLayoutPanel11.ResumeLayout(false);
            this.tableLayoutPanel11.PerformLayout();
            this.groupBox8.ResumeLayout(false);
            this.groupBox8.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        
        private PictureBoxWithInterpolationMode picB;
        private CueTextBox txt_sw_range1;
        private CueTextBox txt_sw_range2;

        private System.Windows.Forms.Button btn_start;
        private Button btn_reset;
        private Button btn_save_model;
        private Button btn_load_model;
        private Button btn_redraw;
        private Label label4;
        private Label lbl_num_axons;
        private GroupBox groupBox1;
        private GroupBox groupBox2;
        private GroupBox groupBox3;
        private TableLayoutPanel tableLayoutPanel4;
        private GroupBox groupBox4;
        private Button btn_load_setts;
        private Button btn_save_setts;
        private Button btn_preprocess;
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
        private ComboBox cmb_sw_sel1;
        private Label label12;
        private Label lbl_image_siz;
        private CheckBox chk_save_sw_prog;
        private GroupBox groupBox5;
        private TableLayoutPanel tableLayoutPanel11;
        private CheckBox chk_strict_rad;
        private TextBox txt_delay_ms;
        private Label label13;
        private TableLayoutPanel tableLayoutPanel8;
        private TableLayoutPanel tableLayoutPanel10;
        private Label label14;
        private Label lbl_real_time;
        private Label label1;
        private Label lbl_itr;
        private TableLayoutPanel tableLayoutPanel12;
        private Label lbl_rem_time;
        private TableLayoutPanel tableLayoutPanel6;
        private Label label2;
        private Label lbl_tox;
        private TableLayoutPanel tableLayoutPanel13;
        private Label label19;
        private Label lbl_sim_time;
        private Button btn_snapshot;
        private Label label21;
        private Label label20;
        private ComboBox cmb_sw_sel2;
        private TableLayoutPanel tableLayoutPanel14;
        private TextBox txt_stop_itr;
        private Label label16;
        private GroupBox groupBox6;
        private TableLayoutPanel tableLayoutPanel15;
        private TextBox txt_detox_extra;
        private Label label27;
        private Label label28;
        private TextBox txt_detox_intra;
        private TextBox txt_resolution;
        private Label label9;
        private GroupBox groupBox7;
        private TableLayoutPanel tableLayoutPanel2;
        private TextBox txt_rate_live;
        private TextBox txt_rate_bound;
        private Label label7;
        private Label label8;
        private TextBox txt_rate_dead;
        private Label label25;
        private Label label26;
        private TextBox txt_rate_extra;
        private Label label30;
        private Label label31;
        private Label label32;
        private Label label33;
        private Label label34;
        private Label label35;
        private Label label36;
        private TextBox txt_death_tox_threshold;
        private TextBox txt_tox_prod_rate;
        private Label label29;
        private TextBox txt_stop_time;
        private Label label24;
        private TextBox txt_block_siz;
        private Label label5;
        private TableLayoutPanel tableLayoutPanel5;
        private Label label22;
        private Label lbl_alive_axons_perc;
        private Label label18;
        private Label lbl_nerve_siz;
        private TextBox txt_insult_tox;
        private Label label6;
        private GroupBox groupBox8;
        private Button btn_clr;
        private TextBox txt_status;
        private TextBox txt_nerve_scale;
        private Label label11;
        private Button btn_save_prog;
        private TextBox txt_on_death_tox;
        private Label label10;
        private Label label15;
        private TextBox txt_clearance;
        private TextBox txt_rec_inerval;
        private CheckBox chk_show_axons;
        private Label label17;
        private CheckBox chk_show_tox;
        private CheckBox chk_rec_avi;
    }
}

