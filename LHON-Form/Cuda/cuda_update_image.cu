
extern "C" __global__  void cuda_update_image(unsigned short im_size, unsigned short bmp_im_size, float bmp_image_compression_ratio,
	unsigned char* bmp, float* tox, unsigned char* axon_mask, float death_tox_lim, bool* show_opts)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < bmp_im_size && y < bmp_im_size) {

		int xy4_bmp = (x * bmp_im_size + y) * 4;
		int xy = (int)((float)x * bmp_image_compression_ratio) * im_size + (int)((float)y * bmp_image_compression_ratio);

		unsigned char r = 0, g = 0, b = 0;
		float tmp = tox[xy] / death_tox_lim;
		if (tmp > 1) tmp = 1;
		unsigned char v = (unsigned char)(tmp * 255);

		r = v;
		//g = 255 - v;

		// live
		//if (axon_mask[xy] >= 1) { b = 255; }
		// dead
		//else if (axon_mask[xy] == 2) { b = 0; g = 0; }

		bmp[xy4_bmp] = b;
		bmp[xy4_bmp + 1] = g;
		bmp[xy4_bmp + 2] = r;
	}
}
