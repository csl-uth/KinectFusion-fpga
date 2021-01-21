/*

 Copyright (c) 2021 Computer Systems Lab - University of Thessaly

 This code is licensed under the MIT License.

*/

#define _HW_

// unoptimized HW implementation of the Tracking kernel

#include <kernels.hpp>
#include <string.h>


extern "C"{
void trackKernel(int* result, float* output, float4* inVertex,
		float4* inNormal, int inSize_x, int inSize_y, float4* refVertex,
		float4* refNormal, int refSize_x, int refSize_y, float* Ttrack_f,
		float* view_f, float dist_threshold,
		float normal_threshold) {

	#pragma HLS INTERFACE s_axilite port=return bundle=control

	#pragma HLS INTERFACE m_axi port=result offset=slave bundle=result
	#pragma HLS INTERFACE s_axilite port=result bundle=control

	#pragma HLS INTERFACE m_axi  port=output offset=slave bundle=output
	#pragma HLS INTERFACE s_axilite port=output bundle=control

	#pragma HLS INTERFACE m_axi  port=inVertex offset=slave bundle=inVertex
	#pragma HLS INTERFACE s_axilite port=inVertex bundle=control

	#pragma HLS INTERFACE m_axi  port=inNormal offset=slave bundle=inNormal
	#pragma HLS INTERFACE s_axilite port=inNormal bundle=control

	#pragma HLS INTERFACE m_axi  port=refVertex offset=slave bundle=refVertex
	#pragma HLS INTERFACE s_axilite port=refVertex bundle=control

	#pragma HLS INTERFACE m_axi  port=refNormal offset=slave bundle=refNormal
	#pragma HLS INTERFACE s_axilite port=refNormal bundle=control

	#pragma HLS INTERFACE m_axi  port=Ttrack_f offset=slave bundle=Ttrack_f
	#pragma HLS INTERFACE s_axilite port=Ttrack_f bundle=control

	#pragma HLS INTERFACE m_axi  port=view_f offset=slave bundle=view_f
	#pragma HLS INTERFACE s_axilite port=view_f bundle=control

	#pragma HLS INTERFACE s_axilite port=inSize_x bundle=control
	#pragma HLS INTERFACE s_axilite port=inSize_y bundle=control
	#pragma HLS INTERFACE s_axilite port=refSize_x bundle=control
	#pragma HLS INTERFACE s_axilite port=refSize_y bundle=control
	#pragma HLS INTERFACE s_axilite port=dist_threshold bundle=control
	#pragma HLS INTERFACE s_axilite port=normal_threshold bundle=control

	#pragma HLS DATA_PACK variable=inVertex
	#pragma HLS DATA_PACK variable=inNormal
	#pragma HLS DATA_PACK variable=refVertex
	#pragma HLS DATA_PACK variable=refNormal



	Matrix4 Ttrack;
	Matrix4 view;

	

	float temp1[16];
	memcpy(temp1, Ttrack_f, 16*sizeof(float));
	float temp2[16];
	memcpy(temp2, view_f, 16*sizeof(float));
	#pragma HLS ARRAY_PARTITION variable=temp1 complete
	#pragma HLS ARRAY_PARTITION variable=temp2 complete

	COPY_LOOP: for (int i = 0; i < 4; i ++) {
		#pragma HLS PIPELINE II=1
		Ttrack.data[i].x = temp1[i*4];
		Ttrack.data[i].y = temp1[i*4 + 1];
		Ttrack.data[i].z = temp1[i*4 + 2];
		Ttrack.data[i].w = temp1[i*4 + 3];
		view.data[i].x = temp2[i*4];
		view.data[i].y = temp2[i*4 + 1];
		view.data[i].z = temp2[i*4 + 2];
		view.data[i].w = temp2[i*4 + 3];
	}

	uint2 pixel = make_uint2(0, 0);
	unsigned int pixely, pixelx;

	for (pixely = 0; pixely < inSize_y; pixely++) {
		#pragma HLS PIPELINE off
		for (pixelx = 0; pixelx < inSize_x; pixelx++) {
			#pragma HLS PIPELINE off
			pixel.x = pixelx;
			pixel.y = pixely;

			//TrackData & row = output[pixel.x + pixel.y * refSize_x];
			int idx = pixel.x + pixel.y * refSize_x;

			if (inNormal[pixel.x + pixel.y * inSize_x].x == KFUSION_INVALID) {
				result[idx] = -1;
				continue;
			}

			float3 projectedVertex = Ttrack
					* make_float3(inVertex[pixel.x + pixel.y * inSize_x]);
			float3 projectedPos = view * projectedVertex;
			float2 projPixel = make_float2(
					projectedPos.x / projectedPos.z + 0.5f,
					projectedPos.y / projectedPos.z + 0.5f);
			if (projPixel.x < 0 || projPixel.x > refSize_x - 1
					|| projPixel.y < 0 || projPixel.y > refSize_y - 1) {
				result[idx] = -2;
				continue;
			}

			uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
			float3 referenceNormal;
			referenceNormal.x = refNormal[refPixel.x + refPixel.y * refSize_x].x;
			referenceNormal.y = refNormal[refPixel.x + refPixel.y * refSize_x].y;
			referenceNormal.z = refNormal[refPixel.x + refPixel.y * refSize_x].z;

			if (referenceNormal.x == KFUSION_INVALID) {
				result[idx] = -3;
				continue;
			}

			float3 diff = make_float3(refVertex[refPixel.x + refPixel.y * refSize_x])
					- projectedVertex;
			float3 projectedNormal = rotate(Ttrack,
					inNormal[pixel.x + pixel.y * inSize_x]);

			if (length(diff) > dist_threshold) {
				result[idx] = -4;
				continue;
			}
			if (dot(projectedNormal, referenceNormal) < normal_threshold) {
				result[idx] = -5;
				continue;
			}
			result[idx] = 1;

			idx *= 7;
			output[idx] = dot(referenceNormal, diff);

			float3 cross_res = cross(projectedVertex, referenceNormal);

			output[idx + 1] = referenceNormal.x;
			output[idx + 2] = referenceNormal.y;
			output[idx + 3] = referenceNormal.z;
			output[idx + 4] = cross_res.x;
			output[idx + 5] = cross_res.y;
			output[idx + 6] = cross_res.z;
		}
	}
}
}
