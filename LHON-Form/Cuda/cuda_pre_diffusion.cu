﻿//
//  cuda_pre_diffusion.cu
//  LHON-Form
//
//  Created by Pooya Merat in 2016.
//  

extern "C" __global__  void cuda_pre_diffusion(int* pix_idx, int pix_idx_num, unsigned short im_size,
	float* tox_new, float* tox, float* rate, float* detox, float* tox_prod)
{
	int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if (idx < pix_idx_num)
	{
		int xy = pix_idx[idx];

		int xy0 = xy + im_size;
		int xy1 = xy - im_size;
		int xy2 = xy + 1;
		int xy3 = xy - 1;
		int xy4 = xy * 4;

		float t = tox[xy];

		tox_new[xy] = t +
			(tox[xy0] - t) * rate[xy4] +
			(tox[xy1] - t) * rate[xy4 + 1] +
			(tox[xy2] - t) * rate[xy4 + 2] +
			(tox[xy3] - t) * rate[xy4 + 3] +
			tox_prod[xy];

		tox_new[xy] *= detox[xy];
	}
}
