//
//  cuda_update_image.cu
//  LHON-Form
//
//  Created by Pooya Merat in 2016.
//

extern "C" __global__  void cuda_update_image(unsigned short im_size, unsigned short bmp_im_size, float bmp_image_compression_ratio,
	unsigned char* bmp, float* tox, unsigned char* axon_mask, unsigned char* init_insult_mask, float death_tox_thres, bool* show_opts)
{
	int x_bmp = blockIdx.x * blockDim.x + threadIdx.x;
	int y_bmp = blockIdx.y * blockDim.y + threadIdx.y;

	if (x_bmp < bmp_im_size && y_bmp > 0) {

		int xy_bmp = x_bmp * bmp_im_size + y_bmp;
		int xy4_bmp = xy_bmp * 4;
		int xy = (int)((float)y_bmp * bmp_image_compression_ratio) * im_size + (int)((float)(bmp_im_size - x_bmp) * bmp_image_compression_ratio);

		unsigned char red = 0, green = 0, blue = 0;
		float tmp = tox[xy] / death_tox_thres;
		if (tmp > 1) tmp = 1;
		unsigned char normalized_toxin = (unsigned char)(tmp * 255); // 0 - 255
		
		if (init_insult_mask[xy_bmp]) {blue = green = 127; red = 0;}
		else
		{
			if (show_opts[0]){
				if (axon_mask[xy] == 1) { green = 100; } // live
				if (axon_mask[xy] == 2) { blue = 255; green = 0; } // dead
				// else: the pixel doesn't belongs to any axon
			}
			else {blue = green = 0;}
		
			if (show_opts[1]){
				red = normalized_toxin;
				// green = 255 - normalized_toxin;
			}
			else {red = 0;}
		}

		bmp[xy4_bmp] = blue;
		bmp[xy4_bmp + 1] = green;
		bmp[xy4_bmp + 2] = red;
	}
}

/*

// Jet colormap: https://www.mathworks.com/help/matlab/ref/jet.html

if (normalized_toxin < 64) { r = 0; g = 4 * v; b = 255; }
else if (normalized_toxin < 128) { r = 0; b = 255 + 4 * (64 - v); g = 255; }
else if (normalized_toxin < 192) { r = 4 * (v - 128); b = 0; g = 255; }
else { g = 255 + 4 * (192 - normalized_toxin); b = 0; r = 255; }

*/
