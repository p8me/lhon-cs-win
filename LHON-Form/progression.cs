
 /*

 NO_PROGRESSION
                    gpu.Set(progress_dev);
                    gpu.Launch(blocks_per_grid, threads_per_block)).gpu_areal_progress(tox_dev, locked_pix_dev, progress_dev, areal_progress_lim);
                    gpu.CopyFromDevice(progress_dev, progress_dat);
                    gpu.CopyFromDevice(num_alive_axons_dev, num_live_neur);

                    areal_progress = progress_dat[2] / progress_dat[0];
                    chron_progress = (float)iteration / last_itr;
                    
                    if (progress_dat[1] == 0)
                    {
                        stop_sim(sim_stat_enum.Failed);
                        append_stat_ln(string.Format("Simulation failed after {0:0.0} secs.", tt_sim.read() / 1000F));
                        tt_sim.pause();
                    }

                    if (areal_progress > next_areal_progress_snapshot || areal_progress >= last_areal_prog)
                    {
                        areal_progress_chron_val[areal_progression_image_stack_cnt] = chron_progress;
                        if (areal_progress >= last_areal_prog) last_areal_prog = Single.PositiveInfinity;
                        next_areal_progress_snapshot += progress_step;
                        Take_Progress_Snapshot(areal_progression_image_stack, areal_progression_image_stack_cnt++);
                    }

                    if (chron_progress > next_chron_progress_snapshot)
                    {
                        chron_progress_areal_val[chron_progression_image_stack_cnt] = areal_progress;
                        next_chron_progress_snapshot += progress_step;
                        Take_Progress_Snapshot(chron_progression_image_stack, chron_progression_image_stack_cnt++);
                    }
                    
                    // Automatic End of Simulation
                    if (iteration > last_itr + 5)
                    {
                        stop_sim(sim_stat_enum.Successful);
                        append_stat_ln(string.Format("Simulation Successful after {0:0.0} secs.", tt_sim.read() / 1000F));
                        tt_sim.pause();

                        if (!sweep_is_running && chk_save_prog.Checked)
                            Save_Progress(ProjectOutputDir + @"Progression\" + DateTime.Now.ToString("yyyy-MM-dd @HH-mm-ss") + ".prgim");
                    }
                    */
