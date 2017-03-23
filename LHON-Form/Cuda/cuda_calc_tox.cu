
extern "C" __global__  void cuda_calc_tox(int im_size, float* tox, float* rate, float* detox, unsigned char* locked_pix, float* diff)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x_y = x * im_size + y;

	if (!locked_pix[x_y])
	{
		if (rate[x_y] > 0 && diff[x_y] > 0)
		{
			tox[x_y] += diff[x_y];
			if (tox[x_y] > 0 && detox[x_y] > 0) tox[x_y] -= detox[x_y];
		}
	}
}
