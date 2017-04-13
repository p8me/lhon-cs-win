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

namespace LHON_Form
{
    public partial class Main_Form
    {
        // =======================================

        private void btn_sweep_Click(object sender, EventArgs e)
        {

            if (sweep_is_running)
            {
                stop_sweep_req = true;
                stop_sim(sim_stat_enum.Paused);
                btn_sweep.Text = "S&weep";
                append_stat_ln("Sweeping Terminated by User!");
            }
            else sweep();
        }

        enum param_select
        {
            Repeat,
            Nerve_Rad,
            Clearance,
            Resolution,
            Tolerance,
            Axon_Rate,
            Insult_Rad,
            Insult_Peri
        };

        async void sweep()
        {
            if (sim_stat == sim_stat_enum.Running) return;

            try
            {
                int delay_ms = 2000;

                int sweep_repetitions1, sweep_repetitions2 = 0;
                float start1 = 0, end1 = 0;
                float start2 = 0, end2 = 0;
                int selection1 = cmb_sw_sel1.SelectedIndex;
                int selection2 = cmb_sw_sel2.SelectedIndex;

                // Get first dimension of sweep
                try
                {
                    string[] values = txt_sw_range1.Text.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    sweep_repetitions1 = int.Parse(values[0]);
                    if (selection1 > 0)
                    {
                        start1 = float.Parse(values[1]);
                        if (sweep_repetitions1 > 1) end1 = float.Parse(values[2]);
                    }
                }
                catch
                {
                    MessageBox.Show("Not enough or bad input for Sweep command!");
                    return;
                }
                // Get Second dimension
                try
                {
                    string[] values = txt_sw_range2.Text.Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                    sweep_repetitions2 = int.Parse(values[0]);
                    if (selection2 > 0)
                    {
                        start2 = float.Parse(values[1]);
                        if (sweep_repetitions2 > 1) end2 = float.Parse(values[2]);
                    }
                }
                catch { }

                btn_sweep.Text = "S&top";
                sweep_is_running = true;

                string parameter_name1 = cmb_sw_sel1.Items[selection1].ToString();
                string parameter_name2 = cmb_sw_sel2.Items[selection2].ToString();

                int failures = 0;
                string dir_name = null;

                for (int i1 = 0; i1 < sweep_repetitions1; i1++)
                {
                    float val1 = sweep_upd_param((param_select)selection1, start1, end1, i1, sweep_repetitions1);
                    float val2 = 0;
                    append_stat_ln(parameter_name1 + " : " + val1.ToString());

                    bool regenerate_model = selection1 < (int)param_select.Resolution;
                    int i2 = 0;
                    do
                    {
                        if (sweep_repetitions2 > 0)
                        {
                            val2 = sweep_upd_param((param_select)selection2, start2, end2, i2, sweep_repetitions2);
                            append_stat_ln(parameter_name2 + " : " + val2.ToString());
                        }
                        update_mdl_and_setts_ui();

                        if (regenerate_model || (sweep_repetitions2 > 0 && selection2 < (int)param_select.Resolution))
                        {
                            regenerate_model = false;
                            // Regenerate Model
                            new_model_worker.RunWorkerAsync();
                            await Task.Delay(delay_ms);
                            while (sim_stat == sim_stat_enum.Running)
                            {
                                await Task.Delay(delay_ms);
                                if (stop_sweep_req) { stop_sweep_req = false; sweep_is_running = false; return; }
                            }
                        }
                        else
                            preprocess_model();

                        start_sim();
                        //Debug.WriteLine("simulation should've been started" + sim_stat);
                        while (sim_stat != sim_stat_enum.Successful && sim_stat != sim_stat_enum.Failed)
                        {
                            await Task.Delay(delay_ms);
                            if (stop_sweep_req) { stop_sweep_req = false; sweep_is_running = false; return; }
                        }
                        await Task.Delay(delay_ms / 5);

                        if (sim_stat == sim_stat_enum.Failed) failures++;
                        else // Successful
                        if (chk_save_sw_prog.Checked)
                        {
                            if (dir_name == null)
                            {
                                string par_nam = "(" + parameter_name1 + ")";
                                if (sweep_repetitions2 > 0)
                                    par_nam += "(" + parameter_name2 + ")";
                                dir_name = string.Format(ProjectOutputDir + "Progression\\{0} {1}", DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss"), par_nam);
                                Directory.CreateDirectory(dir_name);
                            }
                            string par_val;
                            if ((param_select)selection1 == param_select.Repeat)
                                par_val = val1.ToString("(00)");
                            else
                                par_val = val1.ToString("(0.00)");
                            if (sweep_repetitions2 > 0)
                                if ((param_select)selection2 == param_select.Repeat)
                                    par_val += val2.ToString("(00)");
                                else
                                    par_val += val2.ToString("(0.00)");
                            Save_Progress(string.Format("{0}\\{1}.prgim", dir_name, par_val));
                        }
                        i2++;
                    }
                    while (i2 < sweep_repetitions2);
                }

                append_stat_ln(string.Format("Sweeping finished after {0} repetitions and {1} failure(s).", (sweep_repetitions2 > 0 ? sweep_repetitions2 * sweep_repetitions1 : sweep_repetitions1).ToString(), failures > 0 ? failures.ToString() : "no"));
                sweep_is_running = false;
                btn_sweep.Text = "S&weep";
            }
            catch (Exception e)
            {
                MessageBox.Show(e.ToString());
            }
        }

        float sweep_upd_param(param_select selection, float start, float end, int i, int sweep_repetitions)
        {
            float sig = end < start ? -1 : 1;
            float step_siz;
            if (selection == 0 || sweep_repetitions == 1) step_siz = 1;
            else step_siz = Math.Abs(end - start) / (float)(sweep_repetitions - 1);
            float val = start + sig * step_siz * i;

            switch (selection)
            {
                case param_select.Nerve_Rad:
                    mdl.nerve_scale_ratio = val;
                    break;
                case param_select.Clearance:
                    mdl.clearance = val;
                    break;
                case param_select.Resolution:
                    setts.resolution = val;
                    break;
                case param_select.Insult_Rad:
                    init_insult[0] = mdl.nerve_scale_ratio * (val * 2 - 1);
                    init_insult[1] = 0;
                    break;
                case param_select.Insult_Peri:
                    init_insult[0] = -mdl.nerve_scale_ratio * (float)Math.Cos(val * Math.PI / 180);
                    init_insult[1] = -mdl.nerve_scale_ratio * (float)Math.Sin(val * Math.PI / 180);
                    break;
            }
            return val;
        }
    }
}
