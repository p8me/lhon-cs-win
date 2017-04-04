
extern "C" __global__  void cuda_calc_diff(int im_size, float* tox, float* rate, unsigned char* locked_pix, float* diff)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x_y = x * im_size + y;

	if (!locked_pix[x_y] && tox[x_y] > 0)
	{
		int x_y_1 = x_y + im_size;
		int x_y_2 = x_y - im_size;
		int x_y_3 = x_y + 1;
		int x_y_4 = x_y - 1;

		float rate_sum = 0;

		float rate_4 = rate[x_y] / 4;
		float tox_gives = tox[x_y] * rate_4;

		if (!locked_pix[x_y_1]) { rate_sum += rate[x_y_1]; atomicAdd(&diff[x_y_1], tox_gives * rate[x_y_1]); }
		if (!locked_pix[x_y_2]) { rate_sum += rate[x_y_2]; atomicAdd(&diff[x_y_2], tox_gives * rate[x_y_2]); }
		if (!locked_pix[x_y_3]) { rate_sum += rate[x_y_3]; atomicAdd(&diff[x_y_3], tox_gives * rate[x_y_3]); }
		if (!locked_pix[x_y_4]) { rate_sum += rate[x_y_4]; atomicAdd(&diff[x_y_4], tox_gives * rate[x_y_4]); }

		float tox_giving_away_portion = rate_4 * rate_sum;

		tox[x_y] *= (1 - tox_giving_away_portion);
	}
}