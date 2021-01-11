//
//  cuda_diffusion.cu
//  LHON-Form
//
//  Created by Pooya Merat in 2016.
//

extern "C" __global__  void cuda_diffusion(int* pix_idx, int pix_idx_num, float* tox_new, float* tox)
{
	int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if (idx < pix_idx_num)
	{
		int xy = pix_idx[idx];
		tox[xy] = tox_new[xy];
	}
}
