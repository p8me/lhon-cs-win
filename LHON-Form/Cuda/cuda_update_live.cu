
extern "C" __global__  void cuda_update_live(int im_size, int n_neurs, float* tox, float* rate, float* detox,
	bool* live_neur, int* num_live_neur, float* tox_touch_neur, float* neur_tol, int* axons_bound_touch_pix, int max_set_size_bound_touch,
	int* axons_bound_touch_npix, unsigned short* axons_inside_pix, int* axons_inside_pix_idx, unsigned char* locked_pix, int* death_itr, int itr)
{
	int t_id = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int n = t_id; n < n_neurs; n += stride)
	{
		if (live_neur[n])
		{
			if (tox_touch_neur[n] > neur_tol[n])
			{ 	// Kill the axon
				for (int p = axons_inside_pix_idx[n]; p < axons_inside_pix_idx[n + 1]; p++)
				{
					int idx_x = p * 2;
					int idx = axons_inside_pix[idx_x] * im_size + axons_inside_pix[idx_x + 1];
					locked_pix[idx]--;
				}

				for (int p = 0; p < axons_bound_touch_npix[n]; p++)
				{
					int idx_x = (n * max_set_size_bound_touch + p) * 2;
					int idx = axons_bound_touch_pix[idx_x] * im_size + axons_bound_touch_pix[idx_x + 1];
					detox[idx] = 0;
				}

				live_neur[n] = false;
				death_itr[n] = itr;
				atomicAdd(&num_live_neur[0], -1);
			}
		}
	}
}
