/*

 Copyright (c) 2021 Computer Systems Lab - University of Thessaly

 This code is licensed under the MIT License.

*/
#define _HW_

// HW implementation of the Tracking kernel
// enabled optimizations: Tr_Pipe

// A difference between the Conf1 Tracking kernel and this kernel is that we use a larger BRAM buffers.

#include <kernels.hpp>
#include <string.h>

#define BLCK_PIXEL_Y 8

#define MAX_X 320
#define FLT4_PER_ELEM 2

#define BUFFER_LINE MAX_X*FLT4_PER_ELEM
#define BUFFER_SZ BUFFER_LINE*BLCK_PIXEL_Y

extern "C"{
void trackKernel(float4* outputdata, float4* inVertex,
		float4* inNormal, int inSize_x, int inSize_y, float4* refVertex,
		float4* refNormal, int refSize_x, int refSize_y, float* Ttrack_f,
		float* view_f, float dist_threshold,
		float normal_threshold) {

	#pragma HLS INTERFACE s_axilite port=return bundle=control

	#pragma HLS INTERFACE m_axi  port=outputdata offset=slave bundle=outputdata
	#pragma HLS INTERFACE s_axilite port=outputdata bundle=control

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


	#pragma HLS DATA_PACK variable=outputdata
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

	float4 output[BUFFER_SZ];
	#pragma HLS ARRAY_PARTITION variable=output cyclic factor=2

	unsigned int pixely, pixelx, y;


	BLCK_Y: for (y = 0; y < inSize_y; y += BLCK_PIXEL_Y) {
		Y_LOOP:for (pixely = 0; pixely < BLCK_PIXEL_Y; pixely++) {
			X_LOOP:for (pixelx = 0; pixelx < inSize_x; pixelx++) {

				#pragma HLS PIPELINE II=1
				int bufIdx = FLT4_PER_ELEM*pixelx + pixely*BUFFER_LINE;

				int cur_level_idx = pixelx + (pixely+y) * inSize_x;

				float4 normalFrame = inNormal[cur_level_idx];
				if (normalFrame.x == KFUSION_INVALID) {
					output[bufIdx].x = -1;
					continue;
				}

				float3 projectedVertex = Ttrack
						* make_float3(inVertex[cur_level_idx]);
				float3 projectedPos = view * projectedVertex;
				float2 projPixel = make_float2(
						projectedPos.x / projectedPos.z + 0.5f,
						projectedPos.y / projectedPos.z + 0.5f);
				if (projPixel.x < 0 || projPixel.x > refSize_x - 1
						|| projPixel.y < 0 || projPixel.y > refSize_y - 1) {
					output[bufIdx].x = -2;
					continue;
				}

				uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
				float3 referenceNormal;
				int refId = refPixel.x + refPixel.y * refSize_x;
				referenceNormal.x = refNormal[refId].x;
				referenceNormal.y = refNormal[refId].y;
				referenceNormal.z = refNormal[refId].z;

				if (referenceNormal.x == KFUSION_INVALID) {
					output[bufIdx].x = -3;
					continue;
				}

				float3 diff = make_float3(refVertex[refId])
						- projectedVertex;
				float3 projectedNormal = rotate(Ttrack,
								normalFrame);

				if (length(diff) > dist_threshold) {
					output[bufIdx].x = -4;
					continue;
				}
				if (dot(projectedNormal, referenceNormal) < normal_threshold) {
					output[bufIdx].x = -5;
					continue;
				}

				float3 cross_res = cross(projectedVertex, referenceNormal);
				output[bufIdx] = make_float4(1, dot(referenceNormal, diff), referenceNormal.x, referenceNormal.y);
				output[bufIdx + 1] = make_float4(referenceNormal.z, cross_res.x, cross_res.y, cross_res.z);

			}
		}

		for (int i = 0; i < BLCK_PIXEL_Y; i++)
			memcpy(outputdata + BUFFER_LINE*(y+i), output + i*BUFFER_LINE, FLT4_PER_ELEM*inSize_x*sizeof(float4));
		
	}

}
}
