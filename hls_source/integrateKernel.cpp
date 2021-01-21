/*

 Copyright (c) 2021 Computer Systems Lab - University of Thessaly

 This code is licensed under the MIT License.

*/

#define _HW_

// unoptimized HW implementation of the Integrate kernel


#include <kernels.hpp>
#include <string.h>

extern "C"{
void integrateKernel(uint4 *size, short2 *vol_data_out, short2 *vol_data, float4 *dim, float *depth,
		int depthSize_x,int depthSize_y ,
		float *invTrack, float* K, float mu,
		float maxweight,int start, int end) {


	#pragma HLS INTERFACE s_axilite port=return bundle=control

	#pragma HLS INTERFACE m_axi port=depth offset=slave bundle=depth
	#pragma HLS INTERFACE s_axilite port=depth bundle=control

	#pragma HLS INTERFACE m_axi  port=vol_data offset=slave bundle=vol_data
	#pragma HLS INTERFACE s_axilite port=vol_data bundle=control

	#pragma HLS INTERFACE m_axi  port=vol_data_out offset=slave bundle=vol_data_out
	#pragma HLS INTERFACE s_axilite port=vol_data_out bundle=control

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
	#pragma HLS INTERFACE s_axilite port=start bundle=control
	#pragma HLS INTERFACE s_axilite port=end bundle=control

	#pragma HLS DATA_PACK variable=vol_data
	#pragma HLS DATA_PACK variable=vol_data_out
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


	COPY_LOOP: for (int i = 0; i < 4; i ++) {
		//#pragma HLS PIPELINE II=1
		invTrack_local.data[i].x = invTrack[i*4];
		invTrack_local.data[i].y = invTrack[i*4 + 1];
		invTrack_local.data[i].z = invTrack[i*4 + 2];
		invTrack_local.data[i].w = invTrack[i*4 + 3];
		K_local.data[i].x = K[i*4];
		K_local.data[i].y = K[i*4 + 1];
		K_local.data[i].z = K[i*4 + 2];
		K_local.data[i].w = K[i*4 + 3];
	}

	const float3 delta = rotate(invTrack_local,
			make_float3(0, 0, volDim.z / volSz.z));

	const float3 cameraDelta = rotate(K_local, delta);

	Y_LOOP:for (y = 0; y < volSz.y; y++) {
		#pragma hls pipeline off
		X_LOOP:for (unsigned int x = 0; x < volSz.x; x++) {
			#pragma hls pipeline off

			uint3 pix = make_uint3(x, y, 0);
			float3 pos_pix = make_float3((pix.x + 0.5f) * volDim.x / volSz.x,
				(pix.y + 0.5f) * volDim.y / volSz.y, (pix.z + 0.5f) * volDim.z / volSz.z);
			
			float3 pos = invTrack_local * pos_pix;
			float3 cameraX = K_local * pos;
			
			Z_LOOP:for (pix.z = 0; pix.z < volSz.z;
					++pix.z, pos += delta, cameraX += cameraDelta) {
				#pragma hls pipeline off

				if (pos.z < 0.0001f) // some near plane constraint
					continue;
				const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
						cameraX.y / cameraX.z + 0.5f);

				
				if (pixel.x < 0 || pixel.x > depthSize_x - 1 || pixel.y < 0
						|| pixel.y > depthSize_y - 1)
					continue;
				const uint2 px = make_uint2(pixel.x, pixel.y);
				if (depth[px.x + px.y * depthSize_x] == 0)
					continue;
				const float diff =
						(depth[px.x + px.y * depthSize_x] - cameraX.z)
								* std::sqrt(
										1 + sq(pos.x / pos.z)
												+ sq(pos.y / pos.z));

				if (diff > -mu) {
					const float sdf = fminf(1.f, diff / mu);
					float2 data = make_float2(vol_data[pix.x + pix.y * volSz.x + pix.z * volSz.x * volSz.y].x* 0.00003051944088f,vol_data[pix.x + pix.y * volSz.x + pix.z * volSz.x * volSz.y].y);
					data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f, 1.f);
					data.y = fminf(data.y + 1, maxweight);

					vol_data_out[pix.x + pix.y * volSz.x + pix.z * volSz.x * volSz.y] = make_short2(
				data.x * 32766.0f, data.y);

				}
			}
		}
	}
}
}
