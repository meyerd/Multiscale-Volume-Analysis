/**
 *
 * Master Thesis: GPU-based Multiscale Analysis of Volume Data
 *
 * Copyright (C) 2011 Dominik Meyer <meyerd@mytum.de>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 */

#include "modulusMaxima.h"

#include "angle_kernel.h"
#include "modulus_kernel.h"
#include "maxima_kernel.h"
#include "threshold_kernel.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <float.h>
#include <algorithm>
#include <cassert>

#include "timing.h"
#include "cutil_replacement.h"

using namespace std;
using namespace Volume;

namespace Cuda { namespace ModulusMaxima {
	// default threshold value, if no automatic thresholding selected
	float threshold = DEFAULT_THRESHOLD;

	/**
	  Computes the angle of the wavelet transform coefficients. Then searches for the maxima
	  along this gradient direction.

	  @param input VolumeContainer of length 3 with the highpass wavelet coefficients in x,y and z direction
	  */
	MultilevelVolumeContainer<float, 3>* calculateModulusAngles(MultilevelVolumeContainer<float, 3>& input) {
		if(input.x.sSize != input.y.sSize || input.y.sSize != input.z.sSize)
			throw CudaModulusMaximaError("Input erroneous input sizes.");
		if(input.x.m_lLevels != input.y.m_lLevels || input.y.m_lLevels != input.z.m_lLevels)
			throw CudaModulusMaximaError("Input erroneous levels.");
		VolumeSize sSize = input.x.sSize;
		long lLevels = input.x.m_lLevels;
		MultilevelVolumeContainer<float, 3>* output = new MultilevelVolumeContainer<float, 3>(sSize, lLevels);

		// compute angle values and find modulus maximas along the gradient direction

		// Allocate device memory for one level
		cudaPitchedPtr devSrcXPtr = {0};
		cudaPitchedPtr devSrcYPtr = {0};
		cudaPitchedPtr devSrcZPtr = {0};
		cudaPitchedPtr devDstPtr = {0};
		OUT_INFO("modulusMaxima: Allocating device memory >=%li bytes ...\n", sizeof(float)*sSize.x*sSize.y*sSize.z*4);
		cutilSafeCall(cudaMalloc3D(&devSrcXPtr, make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));
		cutilSafeCall(cudaMalloc3D(&devSrcYPtr, make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));
		cutilSafeCall(cudaMalloc3D(&devSrcZPtr, make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));
		cutilSafeCall(cudaMalloc3D(&devDstPtr, make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));

		dim3 threads_angles(anglesBlockSize.x, anglesBlockSize.y);
		dim3 grid_angles((sSize.x + anglesBlockSize.x - 1) / anglesBlockSize.x,
			(sSize.y * sSize.z + anglesBlockSize.y - 1) / anglesBlockSize.y);
		dim3 threads_modulus(modulusBlockSize.x, modulusBlockSize.y);
		dim3 grid_modulus((sSize.x + modulusBlockSize.x - 1) / modulusBlockSize.x,
			(sSize.y * sSize.z + modulusBlockSize.y - 1) / modulusBlockSize.y);
		dim3 threads_maxima(maximaBlockSize.x, maximaBlockSize.y);
		dim3 grid_maxima((sSize.x + maximaBlockSize.x - 1) / maximaBlockSize.x,
			(sSize.y * sSize.z + maximaBlockSize.y - 1) / maximaBlockSize.y);
		dim3 threads_threshold(thresholdBlockSize.x, thresholdBlockSize.y);
		dim3 grid_threshold((sSize.x + thresholdBlockSize.x - 1) / thresholdBlockSize.x,
			(sSize.y * sSize.z + thresholdBlockSize.y - 1) / thresholdBlockSize.y);

		dim3 threads_threshold_mean_pass1(thresholdMeanBlockSize.x, thresholdMeanBlockSize.y);
		dim3 grid_threshold_mean_pass1((sSize.x + thresholdMeanBlockSize.x - 1) / thresholdMeanBlockSize.x,
			(sSize.z + thresholdMeanBlockSize.y - 1) / thresholdMeanBlockSize.y);
		dim3 threads_threshold_mean_pass2(thresholdMeanBlockSize.x, 1);
		dim3 grid_threshold_mean_pass2((sSize.x + thresholdMeanBlockSize.x - 1) / thresholdMeanBlockSize.x, 1);
		dim3 threads_threshold_mean_pass3(1, 1);
		dim3 grid_threshold_mean_pass3(1, 1);

		cudaMemcpy3DParms devToHostCopyParams = {0};

		CTimer ctAllLevelsTime;
		CTimer ctOneLevelTime;
		CTimer ctOneStepTime;

		ctAllLevelsTime.Reset();

		for(long l = 0; l < lLevels; l++) {
			ctOneLevelTime.Reset();
			// Copy data to device
			OUT_INFO("modulusMaxima: Copying volume data level %li to device ...\n", l);
			cudaMemcpy3DParms hostToDevCopyParams = {0};
			hostToDevCopyParams.dstPtr = devSrcXPtr;
			hostToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.srcPtr = make_cudaPitchedPtr(input.x.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.kind = cudaMemcpyHostToDevice;
			cutilSafeCall(cudaMemcpy3D(&hostToDevCopyParams));
			hostToDevCopyParams.dstPtr = devSrcYPtr;
			hostToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.srcPtr = make_cudaPitchedPtr(input.y.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.kind = cudaMemcpyHostToDevice;
			cutilSafeCall(cudaMemcpy3D(&hostToDevCopyParams));
			hostToDevCopyParams.dstPtr = devSrcZPtr;
			hostToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.srcPtr = make_cudaPitchedPtr(input.z.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.kind = cudaMemcpyHostToDevice;
			cutilSafeCall(cudaMemcpy3D(&hostToDevCopyParams));

			ctOneStepTime.Reset();
			OUT_INFO("modulusMaxima: Running angles kernel xy ... ");
			calculateAnglesXZ<float><<<grid_angles, threads_angles>>>(devSrcXPtr, devSrcZPtr, devDstPtr, sSize);
			cudaThreadSynchronize();
			cutilCheckMsg("modulusMaxima: Kernel execution failed");
			devToHostCopyParams.dstPtr = make_cudaPitchedPtr(output->x.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.srcPtr = devDstPtr;
			devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
			OUT_INFO("copy to cpu ... ");
			cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));
			OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());

			ctOneStepTime.Reset();
			OUT_INFO("modulusMaxima: Running angles kernel z ... ");
			calculateAnglesY<float><<<grid_angles, threads_angles>>>(devSrcXPtr, devSrcYPtr, devSrcZPtr, devDstPtr, sSize);
			cudaThreadSynchronize();
			cutilCheckMsg("modulusMaxima: Kernel execution failed");
			devToHostCopyParams.dstPtr = make_cudaPitchedPtr(output->y.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.srcPtr = devDstPtr;
			devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
			OUT_INFO("copy to cpu ... ");
			cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));
			OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());

			ctOneStepTime.Reset();
			OUT_INFO("modulusMaxima: Running modulus kernel ... ");
			calculateModulus<float><<<grid_modulus, threads_modulus>>>(devSrcXPtr, devSrcYPtr, devSrcZPtr, devDstPtr, sSize);
			cudaThreadSynchronize();
			cutilCheckMsg("modulusMaxima: Kernel execution failed");

			//// copy back to cpu (DEBUG only)
			/*devToHostCopyParams.dstPtr = make_cudaPitchedPtr(output->z.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.srcPtr = devDstPtr;
			devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
			OUT_INFO("copy to cpu ... ");
			cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));*/

			OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());

			// data should be: devSrcXPtr: anglesxz, devSrcYPtr: anglesy, devDstPtr: modulus, results -> devSrcZPtr
			hostToDevCopyParams.dstPtr = devSrcXPtr;
			hostToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.srcPtr = make_cudaPitchedPtr(output->x.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.kind = cudaMemcpyHostToDevice;
			cutilSafeCall(cudaMemcpy3D(&hostToDevCopyParams));
			hostToDevCopyParams.dstPtr = devSrcYPtr;
			hostToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.srcPtr = make_cudaPitchedPtr(output->y.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			hostToDevCopyParams.kind = cudaMemcpyHostToDevice;
			cutilSafeCall(cudaMemcpy3D(&hostToDevCopyParams));

#if DO_MAXIMA == 1
			ctOneStepTime.Reset();
			OUT_INFO("modulusMaxima: Running maxima kernel ... ");
			findModulusMaxima<float><<<grid_maxima, threads_maxima>>>(devSrcXPtr, devSrcYPtr, devDstPtr, devSrcZPtr, sSize);
			cudaThreadSynchronize();
			cutilCheckMsg("modulusMaxima: Kernel execution failed");
			OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());
#else
			cudaMemcpy3DParms devToDevCopyParams = {0};
			devToDevCopyParams.dstPtr = devSrcZPtr;
			devToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToDevCopyParams.srcPtr = devDstPtr;
			devToDevCopyParams.kind = cudaMemcpyDeviceToDevice;
			cutilSafeCall(cudaMemcpy3D(&devToDevCopyParams));
#endif

			#if AUTO_THRESHOLD == 2
			float* median_tmp = new float[sSize.x * sSize.y * sSize.z];
			devToHostCopyParams.dstPtr = make_cudaPitchedPtr(median_tmp, sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.srcPtr = devSrcZPtr;
			devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
			cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));
			
			sort(&median_tmp[0], &median_tmp[(sSize.z - 1) * sSize.x * sSize.y + (sSize.y - 1) * sSize.x + sSize.x - 1]);

			float fMed = median_tmp[(sSize.x * sSize.y * sSize.z / 2)];
			float fMad = 0.0f;
			
			#pragma omp parallel for
			for(long z = 0; z < sSize.z; ++z) {
				for(long y = 0; y < sSize.y; ++y) {
					for(long x = 0; x < sSize.x; ++x) {
						median_tmp[z * sSize.x * sSize.y + y * sSize.x + x] -= fMed;
					}
				}
			}

			sort(&median_tmp[0], &median_tmp[(sSize.z - 1) * sSize.x * sSize.y + (sSize.y - 1) * sSize.x + sSize.x - 1]);
			fMad = median_tmp[(sSize.x * sSize.y * sSize.z / 2)];

			//fMad = fMed / 0.6745f;
			OUT_INFO("modulusMaxima: median: %f, mad: %f\n", fMed, fMad);

			SAFE_DELETE(median_tmp);
			#endif

			#if THRESHOLDING == 1
				#if AUTO_THRESHOLD == 1
				float fMean = 0.0f;
				float fVarSq = 0.0f;

				ctOneStepTime.Reset();
				OUT_INFO("modulusMaxima: calculating mean ... pass 1 ");
				findMeanPass1<float><<<grid_threshold_mean_pass1, threads_threshold_mean_pass1>>>(devSrcZPtr, devSrcXPtr, sSize);
				cudaThreadSynchronize();
				cutilCheckMsg("modulusMaxima: mean pass 1 kernel execution failed");
				OUT_INFO("2 ");
				findMeanPass2<float><<<grid_threshold_mean_pass2, threads_threshold_mean_pass2>>>(devSrcXPtr, devSrcYPtr, sSize);
				cudaThreadSynchronize();
				cutilCheckMsg("modulusMaxima: mean pass 2 kernel execution failed");
				OUT_INFO("3 ... ");
				findMeanPass3<float><<<grid_threshold_mean_pass3, threads_threshold_mean_pass3>>>(devSrcYPtr, devSrcXPtr, sSize);
				cudaThreadSynchronize();
				cutilCheckMsg("modulusMaxima: mean pass 3 kernel execution failed");
				OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());

				// result is in upper left corner
				devToHostCopyParams.dstPtr = make_cudaPitchedPtr(&fMean, sizeof(float), 1, 1);
				devToHostCopyParams.extent = make_cudaExtent(sizeof(float), 1, 1);
				devToHostCopyParams.srcPtr = devSrcXPtr;
				devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
				cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));

				OUT_INFO("modulusMaxima: mean is %f.\n", fMean);

				ctOneStepTime.Reset();
				OUT_INFO("modulusMaxima: calculating variance ... pass 1 ");
				findVariancePass1<float><<<grid_threshold_mean_pass1, threads_threshold_mean_pass1>>>(devSrcZPtr, devSrcXPtr, sSize, fMean);
				cudaThreadSynchronize();
				cutilCheckMsg("modulusMaxima: variance pass 1 kernel execution failed");
				OUT_INFO("2 ");
				findVariancePass2<float><<<grid_threshold_mean_pass2, threads_threshold_mean_pass2>>>(devSrcXPtr, devSrcYPtr, sSize);
				cudaThreadSynchronize();
				cutilCheckMsg("modulusMaxima: variance pass 2 kernel execution failed");
				OUT_INFO("3 ... ");
				findVariancePass3<float><<<grid_threshold_mean_pass3, threads_threshold_mean_pass3>>>(devSrcYPtr, devSrcXPtr, sSize);
				cudaThreadSynchronize();
				cutilCheckMsg("modulusMaxima: variance pass 3 kernel execution failed");
				OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());

				// result is in upper left corner
				devToHostCopyParams.dstPtr = make_cudaPitchedPtr(&fVarSq, sizeof(float), 1, 1);
				devToHostCopyParams.extent = make_cudaExtent(sizeof(float), 1, 1);
				devToHostCopyParams.srcPtr = devSrcXPtr;
				devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
				cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));

				OUT_INFO("modulusMaxima: variance is %f (%f).\n", sqrt(fVarSq), fVarSq);
				threshold = sqrt(fVarSq * 2.0f * log((float)(sSize.x * sSize.y * sSize.z)));
				#elif AUTO_THRESHOLD == 2
					threshold = fMad * sqrt(log((float)(sSize.x * sSize.y * sSize.z)));
				#endif
				ctOneStepTime.Reset();
				OUT_INFO("modulusMaxima: thresholding ");
				#if SOFT_THRESHOLDING == 1
					OUT_INFO("(SOFT)");
				#elif SOFT_THRESHOLDING == 2	
					OUT_INFO("(COMBINED)");
				#else
					OUT_INFO("(HARD)");
				#endif
				OUT_INFO(" (%.3f) ... ", threshold);
				#if SOFT_THRESHOLDING == 1
				thresholdModulus<float><<<grid_threshold, threads_threshold>>>(devSrcZPtr, devDstPtr, sSize, threshold, 1);
				#elif SOFT_THRESHOLDING == 2
				thresholdModulus<float><<<grid_threshold, threads_threshold>>>(devSrcZPtr, devDstPtr, sSize, threshold, 1);
				thresholdModulus<float><<<grid_threshold, threads_threshold>>>(devDstPtr, devSrcZPtr, sSize, threshold, 0);
				#else
				thresholdModulus<float><<<grid_threshold, threads_threshold>>>(devSrcZPtr, devDstPtr, sSize, threshold, 0);
				#endif
				cudaThreadSynchronize();
				cutilCheckMsg("modulusMaxima: Kernel execution failed");
			#endif

			devToHostCopyParams.dstPtr = make_cudaPitchedPtr(output->z.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			#if THRESHOLDING == 1
				#if SOFT_THRESHOLDING == 2
				devToHostCopyParams.srcPtr = devSrcZPtr;
				#else
				devToHostCopyParams.srcPtr = devDstPtr;
				#endif
			#else
				devToHostCopyParams.srcPtr = devSrcZPtr;
			#endif
			devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
			OUT_INFO("copy to cpu ... ");
			cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));
			OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());

			OUT_INFO("modulusMaxima: Level %li done [%.5fs].\n", l, (float)ctOneLevelTime.Query());
		}
		DEBUG_OUT("modulusMaxima: Total time (incl. copy) %.5fs.\n", (float)ctAllLevelsTime.Query());

		cutilSafeCall(cudaFree(devSrcXPtr.ptr));
		cutilSafeCall(cudaFree(devSrcYPtr.ptr));
		cutilSafeCall(cudaFree(devSrcZPtr.ptr));
		cutilSafeCall(cudaFree(devDstPtr.ptr));

		output->x.m_bIsOk = true;
		output->y.m_bIsOk = true;
		output->z.m_bIsOk = true;
		return output;
	};
};};