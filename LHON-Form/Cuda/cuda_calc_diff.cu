
extern "C" __global__  void cuda_calc_diff(int im_size, float* tox, float* rate, unsigned char* locked_pix, float* diff)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x_y = x * im_size + y;

	if (!locked_pix[x_y])
	{
		int x_y_1 = x_y + im_size;
		int x_y_2 = x_y - im_size;
		int x_y_3 = x_y + 1;
		int x_y_4 = x_y - 1;

		float num_surr = 0;
		float tox_gives = tox[x_y] / 4;

		if (!locked_pix[x_y_1]) {num_surr++; atomicAdd(&diff[x_y_1], tox_gives);}
		if (!locked_pix[x_y_2]) {num_surr++; atomicAdd(&diff[x_y_2], tox_gives);}
		if (!locked_pix[x_y_3]) {num_surr++; atomicAdd(&diff[x_y_3], tox_gives);}
		if (!locked_pix[x_y_4]) {num_surr++; atomicAdd(&diff[x_y_4], tox_gives);}

		tox[x_y] *= (1 - num_surr / 4);
	}
}
