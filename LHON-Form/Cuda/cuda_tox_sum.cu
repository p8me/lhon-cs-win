
extern "C" __global__  void cuda_tox_sum(int* pix_idx, int pix_idx_num, float* tox, float* tox_sum)
{

	int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
	if (idx < pix_idx_num)
	{
		int xy = pix_idx[blockIdx.x * blockDim.x + threadIdx.x];
		atomicAdd(tox_sum, tox[xy]);
	}
}
