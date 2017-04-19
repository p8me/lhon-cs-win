
extern "C" __global__  void cuda_prep0(unsigned short im_size, int nerve_cent_pix, int nerve_r_pix_2, int vein_r_pix_2, float k_rate_extra, float k_detox_extra,
	unsigned char* pix_out_of_nerve, float* rate, float* detox)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < im_size && y < im_size) {

		int xy = x * im_size + y;
		int xy4 = xy * 4;

		int dx = x - nerve_cent_pix;
		int dy = y - nerve_cent_pix;
		int dis2 = dx * dx + dy * dy;

		bool outside = nerve_r_pix_2 - dis2 < 0 || vein_r_pix_2 - dis2 > 0;
		pix_out_of_nerve[xy] = outside ? 1 : 0;

		if (!outside)
		{
			rate[xy4] = k_rate_extra;
			rate[xy4 + 1] = k_rate_extra;
			rate[xy4 + 2] = k_rate_extra;
			rate[xy4 + 3] = k_rate_extra;

			detox[xy] = k_detox_extra;
		}
	}
}

//bool[,] pix_out_of_nerve = new bool[im_size, im_size];
//for (int y = 0; y < im_size; y++)
//    for (int x = 0; x < im_size; x++)
//    {
//        int dx = x - nerve_cent_pix;
//        int dy = y - nerve_cent_pix;
//        int dis2 = dx * dx + dy * dy;

//        bool outside = nerve_r_pix_2 - dis2 < 0 || vein_r_pix_2 - dis2 > 0;
//        pix_out_of_nerve[x, y] = outside;
//        if (!outside)
//        {
//            pix_idx[pix_idx_num++] = x * im_size + y;
//            for (uint k = 0; k < 4; k++)
//                rate[x, y, k] = k_rate_extra;
//            detox[x, y] = k_detox_extra;
//        }
//    }

