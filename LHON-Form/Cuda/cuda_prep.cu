//
//  cuda_prep.cu
//  LHON-Form
//
//  Created by Pooya Merat in 2016.
//

extern "C" __global__  void cuda_prep(unsigned short im_size, unsigned char* pix_out_of_nerve, float* rate)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < im_size && y < im_size)
	{
		int xy = x * im_size + y;
		int xy4 = xy * 4;

		if (pix_out_of_nerve[xy]) {
			rate[xy4] = 0;
			rate[xy4 + 1] = 0;
			rate[xy4 + 2] = 0;
			rate[xy4 + 3] = 0;
		}
		else {
			if (pix_out_of_nerve[xy + im_size]) rate[xy4] = 0;
			if (pix_out_of_nerve[xy - im_size]) rate[xy4 + 1] = 0;
			if (pix_out_of_nerve[xy + 1])		rate[xy4 + 2] = 0;
			if (pix_out_of_nerve[xy - 1])		rate[xy4 + 3] = 0;
		}
	}
}

// Set nerve boundary rates to 0
//for (int y = 0; y < im_size; y++)
//    for (int x = 0; x < im_size; x++)
//    {
//        int[,] neighbors = new int[,] { { x + 1, y }, { x - 1, y }, { x, y + 1 }, { x, y - 1 } };
//        for (uint k = 0; k < 4; k++)
//            if (pix_out_of_nerve[x, y] || pix_out_of_nerve[neighbors[k, 0], neighbors[k, 1]])
//                rate[x, y, k] = 0;
//    }
