/*

 Copyright (c) 2021 Computer Systems Lab - University of Thessaly

 This code is licensed under the MIT License.

*/
#define _HW_

// HW implementation of the Integrate kernel
// enabled optimizations: Int_Inter, Int_Pipe, Int_Inroll

#include <kernels.hpp>
#include <string.h>

#define BRAM_SZ 256*256

extern "C"{
void integrateKernel(uint4 *size, short2 *vol_data_out_1, short2 *vol_data_out_2, short2 *vol_data_1, short2 *vol_data_2, float4 *dim, float *depth,
		int depthSize_x,int depthSize_y ,
		float *invTrack, float* K, float mu,
		float maxweight,int z_offset, int end) {


	#pragma HLS INTERFACE s_axilite port=return bundle=control

	#pragma HLS INTERFACE m_axi port=depth offset=slave bundle=depth
	#pragma HLS INTERFACE s_axilite port=depth bundle=control

	#pragma HLS INTERFACE m_axi  port=vol_data_1 offset=slave bundle=vol_data_1
	#pragma HLS INTERFACE s_axilite port=vol_data_1 bundle=control

	#pragma HLS INTERFACE m_axi  port=vol_data_2 offset=slave bundle=vol_data_2
	#pragma HLS INTERFACE s_axilite port=vol_data_2 bundle=control

	#pragma HLS INTERFACE m_axi  port=vol_data_out_1 offset=slave bundle=vol_data_out_1
	#pragma HLS INTERFACE s_axilite port=vol_data_out_1 bundle=control

	#pragma HLS INTERFACE m_axi  port=vol_data_out_2 offset=slave bundle=vol_data_out_2
	#pragma HLS INTERFACE s_axilite port=vol_data_out_2 bundle=control

	#pragma HLS INTERFACE m_axi  port=size offset=slave bundle=size
	#pragma HLS INTERFACE s_axilite port=size bundle=control

	#pragma HLS INTERFACE m_axi  port=dim offset=slave bundle=dim
	#pragma HLS INTERFACE s_axilite port=dim bundle=control

	#pragma HLS INTERFACE m_axi  port=invTrack offset=slave bundle=invTrack
	#pragma HLS INTERFACE s_axilite port=invTrack bundle=control

	#pragma HLS INTERFACE m_axi  port=K offset=slave bundle=K
	#pragma HLS INTERFACE s_axilite port=K bundle=control

	#pragma HLS INTERFACE s_axilite port=depthSize_x bundle=control
	#pragma HLS INTERFACE s_axilite port=depthSize_y bundle=control
	#pragma HLS INTERFACE s_axilite port=mu bundle=control
	#pragma HLS INTERFACE s_axilite port=maxweight bundle=control
	#pragma HLS INTERFACE s_axilite port=z_offset bundle=control
	#pragma HLS INTERFACE s_axilite port=end bundle=control

	#pragma HLS DATA_PACK variable=vol_data_1
	#pragma HLS DATA_PACK variable=vol_data_2
	#pragma HLS DATA_PACK variable=vol_data_out_1
	#pragma HLS DATA_PACK variable=vol_data_out_2
	#pragma HLS DATA_PACK variable=size
	#pragma HLS DATA_PACK variable=dim

	unsigned int y,x,z;
	Matrix4 invTrack_local;
	Matrix4 K_local;

	uint3 volSz;
	float3 volDim;

	volSz.x = size->x;
	volSz.y = size->y;
	volSz.z = size->z;

	volDim.x = dim->x;
	volDim.y = dim->y;
	volDim.z = dim->z;

	uint idx = 0;
	float3 pos_pix;
	float3 pos;
	float3 cameraX;
	float3 pos_2;
	float3 cameraX_2;
	uint T_idx;

	float depth_hls[76800];

	memcpy(depth_hls, depth, 76800*sizeof(float));

	float temp1[16];
	float temp2[16];
	#pragma HLS ARRAY_PARTITION variable=temp1 complete
	#pragma HLS ARRAY_PARTITION variable=temp2 complete

	memcpy(temp1, invTrack, 16*sizeof(float));
	memcpy(temp2, K, 16*sizeof(float));

	COPY_LOOP: for (int i = 0; i < 4; i ++) {
		#pragma HLS PIPELINE II=1
		invTrack_local.data[i].x = temp1[i*4];
		invTrack_local.data[i].y = temp1[i*4 + 1];
		invTrack_local.data[i].z = temp1[i*4 + 2];
		invTrack_local.data[i].w = temp1[i*4 + 3];
		K_local.data[i].x = temp2[i*4];
		K_local.data[i].y = temp2[i*4 + 1];
		K_local.data[i].z = temp2[i*4 + 2];
		K_local.data[i].w = temp2[i*4 + 3];
	}

	float3 constants;
	constants.x = volDim.x / volSz.x;
	constants.y = volDim.y / volSz.y;
	constants.z = volDim.z / volSz.z;

	Z_LOOP:for (z = 0; z < end; z++) {

		T_idx = z * volSz.x * volSz.y;

		uint in_idx = 0;
		Y_LOOP:for (y = 0; y < 256; y++){


			X_LOOP:for (x = 0; x < 256; x+=2, in_idx+=2) {

				#pragma HLS PIPELINE II=1

				pos_pix = make_float3((x + 0.5f) * constants.x,
						(y + 0.5f) * constants.y, (z + z_offset + 0.5f) * constants.z);
				pos = invTrack_local * pos_pix;
				
				pos_pix.x += constants.x;
				pos_2 = invTrack_local * pos_pix;

				if (pos.z >= 0.0001f) // some near plane constraint
				{
					cameraX = K_local * pos;

					const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
							cameraX.y / cameraX.z + 0.5f);

					if (pixel.x >= 0 && pixel.x <= depthSize_x - 1 && pixel.y >= 0
							&& pixel.y <= depthSize_y - 1)
					{

						const uint2 px = make_uint2(pixel.x, pixel.y);

						idx = px.x + px.y * depthSize_x;

						if (depth_hls[idx] == 0)
							continue;
						const float diff =
								(depth_hls[idx] - cameraX.z)
										* std::sqrt(
												1 + sq(pos.x / pos.z)
														+ sq(pos.y / pos.z));



						if (diff > -mu) {
							const float sdf = fminf(1.f, diff / mu);
							int index = in_idx + T_idx;

							float2 data = make_float2(vol_data_1[index].x* 0.00003051944088f, vol_data_1[index].y);

							data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f,1.f);
							data.y = fminf(data.y + 1, maxweight);


							vol_data_out_1[index] = make_short2(data.x * 32766.0f,data.y);
						}
					}
				}

				if (pos_2.z >= 0.0001f) // some near plane constraint
				{
					cameraX_2 = K_local * pos_2;
					const float2 pixel = make_float2(cameraX_2.x / cameraX_2.z + 0.5f,
							cameraX_2.y / cameraX_2.z + 0.5f);

					if (pixel.x >= 0 && pixel.x <= depthSize_x - 1 && pixel.y >= 0
							&& pixel.y <= depthSize_y - 1)
					{

						const uint2 px = make_uint2(pixel.x, pixel.y);

						idx = px.x + px.y * depthSize_x;

						if (depth_hls[idx] == 0)
							continue;
						const float diff =
								(depth_hls[idx] - cameraX_2.z)
										* std::sqrt(
												1 + sq(pos_2.x / pos_2.z)
														+ sq(pos_2.y / pos_2.z));



						if (diff > -mu) {
							const float sdf = fminf(1.f, diff / mu);
							int index = in_idx + 1 + T_idx;

							float2 data = make_float2(vol_data_2[index].x* 0.00003051944088f, vol_data_2[index].y);

							data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f,1.f);
							data.y = fminf(data.y + 1, maxweight);


							vol_data_out_2[index] = make_short2(data.x * 32766.0f,data.y);
						}
					}
				}

			}
		}
	}


}
}
