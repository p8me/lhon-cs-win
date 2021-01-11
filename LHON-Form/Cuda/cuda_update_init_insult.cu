//
//  cuda_update_init_insult.cu
//  LHON-Form
//
//  Created by Pooya Merat in 2016.
//

extern "C" __global__  void cuda_update_init_insult(unsigned short bmp_im_size, int insult_x, int insult_y, int insult_r2,
	unsigned char* init_insult_mask)
{
	int x_bmp = blockIdx.x * blockDim.x + threadIdx.x;
	int y_bmp = blockIdx.y * blockDim.y + threadIdx.y;

	if (x_bmp < bmp_im_size && y_bmp < bmp_im_size) {

		int xy_bmp = x_bmp * bmp_im_size + y_bmp;

		int dx = x_bmp - insult_x;
		int dy = y_bmp - insult_y;
		int dis2 = dx * dx + dy * dy;

		int dis2perim2 = insult_r2 - dis2;

		bool inside = dis2perim2 < bmp_im_size/2 && dis2perim2 > 0;
		init_insult_mask[xy_bmp] = inside;
	}
}
