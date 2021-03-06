﻿//
//  cuda_tox_sum.cu
//  LHON-Form
//
//  Created by Pooya Merat in 2016.
//

extern "C" __global__  void cuda_tox_sum(int* pix_idx, int pix_idx_num, float* tox, float* tox_sum)
{
	int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if (idx < pix_idx_num)
	{
		int xy = pix_idx[idx];
		atomicAdd(tox_sum, tox[xy]);
	}
}
