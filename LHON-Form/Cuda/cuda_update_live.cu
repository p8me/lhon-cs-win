
extern "C" __global__  void cuda_update_live(int n_axons, float* tox, float* rate, float* detox, float* tox_prod, float k_rate_dead_axon, float k_detox_extra, float death_tox_lim,
	unsigned int * axons_cent_pix, unsigned int* axons_inside_pix, unsigned int* axons_inside_pix_idx, unsigned int* axon_surr_rate, unsigned int* axon_surr_rate_idx,
	bool* axon_is_alive, unsigned char* axon_mask, int* num_alive_axons, int* death_itr, int iteration)
{
	int n = threadIdx.x + blockIdx.x * blockDim.x;

	if (n < n_axons)
	{
		if (axon_is_alive[n] && tox[axons_cent_pix[n]] > death_tox_lim)
		{ 	// Kill the axon
			for (int p = axons_inside_pix_idx[n]; p < axons_inside_pix_idx[n + 1]; p++)
			{
				int idx = axons_inside_pix[p];
				int idx4 = 4 * idx;
				rate[idx4] = k_rate_dead_axon;
				rate[idx4 + 1] = k_rate_dead_axon;
				rate[idx4 + 2] = k_rate_dead_axon;
				rate[idx4 + 3] = k_rate_dead_axon;

				detox[idx] = k_detox_extra;
				tox_prod[idx] = 0;
				axon_mask[idx] = 2; // dead
			}

			for (int p = axon_surr_rate_idx[n]; p < axon_surr_rate_idx[n + 1]; p++)
				rate[axon_surr_rate[p]] = k_rate_dead_axon;

			axon_is_alive[n] = false;
			death_itr[n] = iteration;
			atomicAdd(&num_alive_axons[0], -1);
		}
	}
}
