
extern "C" __global__  void cuda_diffusion(unsigned int im_size, float* tox, float* rate, float* detox, float* tox_prod)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x_y = x * im_size + y;

	int x_y_1 = x_y + im_size;
	int x_y_2 = x_y - im_size;
	int x_y_3 = x_y + 1;
	int x_y_4 = x_y - 1;
	int x_y4 = 4 * x_y;

	float t = tox[x_y];

	tox[x_y] +=
		(tox[x_y_1] - t) * rate[x_y4] +
		(tox[x_y_2] - t) * rate[x_y4 + 1] +
		(tox[x_y_3] - t) * rate[x_y4 + 2] +
		(tox[x_y_4] - t) * rate[x_y4 + 3];
	
	tox[x_y] += tox_prod[x_y];

	tox[x_y] *= detox[x_y];

	// 12 float operations
	// 14 int operations (including indices)
	// 10 array addressing
}
