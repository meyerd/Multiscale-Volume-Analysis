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

#include "global.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "dwtForward.h"
#include "2ddwt_kernel.h"
#include "copy_kernel.h"
#include "threshold_kernel.h"

#include <string>
#include <stdexcept>
#include <cmath>

#include "timing.h"
#include "cutil_replacement.h"

using namespace Volume;
using namespace std;

namespace Cuda { namespace DWT {

	float pre_dwt_threshold = 63.0f;
	#define THRESHOLDING_PRE_DWT 0

	class CudaDWTError : public std::runtime_error {
	public:
		CudaDWTError(const string& sWhat) : runtime_error(sWhat) {};
	};
	class CudaDWTSizeError : public CudaDWTError {
	public:
		CudaDWTSizeError(const string& sWhat) : CudaDWTError(sWhat) {};
	};
	
	const float dwtLambda[] = {1.50f, 1.12f, 1.03f, 1.01f};
	const int dwtLambdaLength = 4;
	float getLambda(long level) {
		if(level >= dwtLambdaLength) {
			return 1.0f;
		} else {
			return dwtLambda[level];
		}
	}
	
	void dwtCheckSizes(const Volume::BasicVolume<float>& input, long lLevels) {
		VolumeSize sSize = input.sSize;
		if(sSize.x % (DWT_X_STEPS * dwtXBlockSize.x) != 0)
			throw CudaDWTSizeError("Volume width not supported.");
		if(sSize.y % (dwtXBlockSize.y) != 0)
			throw CudaDWTSizeError("Volume height not supported.");
		if(sSize.x % (dwtXBlockSize.x) != 0)
			throw CudaDWTSizeError("Volume width not supported.");
		if(sSize.y % (DWT_X_STEPS * dwtXBlockSize.y) != 0)
			throw CudaDWTSizeError("Volume height not supported.");
		if(sSize.z % (DWT_X_STEPS * dwtXBlockSize.z) != 0)
			throw CudaDWTSizeError("Volume depth not supported.");
		if(sSize.z % (dwtXBlockSize.z) != 0)
			throw CudaDWTSizeError("Volume depth not supported.");

		if(sSize.x % (DWT_Y_STEPS * dwtXBlockSize.x) != 0)
			throw CudaDWTSizeError("Volume width not supported.");
		if(sSize.y % (dwtXBlockSize.y) != 0)
			throw CudaDWTSizeError("Volume height not supported.");
		if(sSize.x % (dwtXBlockSize.x) != 0)
			throw CudaDWTSizeError("Volume width not supported.");
		if(sSize.y % (DWT_Y_STEPS * dwtXBlockSize.y) != 0)
			throw CudaDWTSizeError("Volume height not supported.");
		if(sSize.z % (DWT_Y_STEPS * dwtXBlockSize.z) != 0)
			throw CudaDWTSizeError("Volume depth not supported.");
		if(sSize.z % (dwtXBlockSize.z) != 0)
			throw CudaDWTSizeError("Volume depth not supported.");

		if(sSize.x % (DWT_Z_STEPS * dwtXBlockSize.x) != 0)
			throw CudaDWTSizeError("Volume width not supported.");
		if(sSize.y % (dwtXBlockSize.y) != 0)
			throw CudaDWTSizeError("Volume height not supported.");
		if(sSize.x % (dwtXBlockSize.x) != 0)
			throw CudaDWTSizeError("Volume width not supported.");
		if(sSize.y % (DWT_Z_STEPS * dwtXBlockSize.y) != 0)
			throw CudaDWTSizeError("Volume height not supported.");
		if(sSize.z % (DWT_Z_STEPS * dwtXBlockSize.z) != 0)
			throw CudaDWTSizeError("Volume depth not supported.");
		if(sSize.z % (dwtXBlockSize.z) != 0)
			throw CudaDWTSizeError("Volume depth not supported.");
	};

	Volume::MultilevelVolumeContainer<float, 3>* dwtForward(const Volume::BasicVolume<float>& input, long lLevels, Volume::MultilevelVolume<float>* lowpass) {
		Volume::MultilevelVolumeContainer<float, 3>* output = new Volume::MultilevelVolumeContainer<float, 3>(input.sSize, lLevels);

		VolumeSize sSize = input.sSize;

		if(lowpass) {
			if(lowpass->sSize != sSize)
				throw CudaDWTSizeError("Lowpass size does not fit input size");
			if(lowpass->m_lLevels != lLevels)
				throw CudaDWTSizeError("Lowpass levels does not fit desired levels");
		}

		// Allocate device memory for one level
		cudaPitchedPtr devSrcPtr = {0};
		cudaPitchedPtr devDstPtr = {0};
		OUT_INFO("dwtForward: Allocating device memory >=%li bytes ...\n", sizeof(float)*sSize.x*sSize.y*sSize.z*2);
		cutilSafeCall(cudaMalloc3D(&devSrcPtr, make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));
		cutilSafeCall(cudaMalloc3D(&devDstPtr, make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));

		// Copy data to device
		cudaMemcpy3DParms hostToDevCopyParams = {0};
		hostToDevCopyParams.dstPtr = devSrcPtr;
		hostToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
		hostToDevCopyParams.srcPtr = make_cudaPitchedPtr(input.m_pData, sizeof(float)*sSize.x, sSize.y, sSize.z);
		hostToDevCopyParams.kind = cudaMemcpyHostToDevice;
		OUT_INFO("dwtForward: Copying volume data to device ...\n");
		cutilSafeCall(cudaMemcpy3D(&hostToDevCopyParams));

		dim3 threads_threshold(ModulusMaxima::thresholdBlockSize.x, ModulusMaxima::thresholdBlockSize.y);
		dim3 grid_threshold((sSize.x + ModulusMaxima::thresholdBlockSize.x - 1) / ModulusMaxima::thresholdBlockSize.x,
			(sSize.y * sSize.z + ModulusMaxima::thresholdBlockSize.y - 1) / ModulusMaxima::thresholdBlockSize.y);

		cudaMemcpy3DParms devToHostCopyParams = {0};
		cudaMemcpy3DParms devToDevCopyParams = {0};

#if THRESHOLDING_PRE_DWT == 1
		ModulusMaxima::thresholdModulus<float><<<grid_threshold, threads_threshold>>>(devSrcPtr, devDstPtr, sSize, pre_dwt_threshold, 0);
		cudaThreadSynchronize();
		cutilCheckMsg("dwtForward: Threshold kernel execution failed");
		// Copy highpass data back to cpu memory
		devToDevCopyParams.dstPtr = devSrcPtr;
		devToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
		devToDevCopyParams.srcPtr = devDstPtr;
		devToDevCopyParams.kind = cudaMemcpyDeviceToDevice;
		cutilSafeCall(cudaMemcpy3D(&devToDevCopyParams));
#endif

		dim3 threads_x(dwtXBlockSize.x, dwtXBlockSize.y);
		dim3 grid_x((sSize.x + (DWT_X_STEPS * dwtXBlockSize.x) - 1) / (DWT_X_STEPS * dwtXBlockSize.x),
			(sSize.y * sSize.z + dwtXBlockSize.y - 1) / dwtXBlockSize.y);
		dim3 threads_y(dwtYBlockSize.x, dwtYBlockSize.y);
		dim3 grid_y((sSize.x + dwtYBlockSize.x - 1) / dwtYBlockSize.x,
			(sSize.y * sSize.z + (DWT_Y_STEPS * dwtYBlockSize.y) - 1) / (DWT_Y_STEPS * dwtYBlockSize.y));
		dim3 threads_z(dwtZBlockSize.x, dwtZBlockSize.y);
		dim3 grid_z((sSize.x * sSize.y + dwtZBlockSize.x - 1) / dwtZBlockSize.x,
			(sSize.z + (DWT_Z_STEPS * dwtZBlockSize.y) - 1) / (DWT_Z_STEPS * dwtZBlockSize.y));

		CTimer ctAllLevelsTime;
		CTimer ctOneLevelTime;
		CTimer ctOneStepTime;

		ctAllLevelsTime.Reset();

		for(long l = 0; l < lLevels; l++) {
			ctOneLevelTime.Reset();
			OUT_INFO("dwtForward: Calculating level %li ...\n", l);

			float one_over_lambda = 1.0f / getLambda(l);
			long zeroes_padding = (long)max((pow(2.0, l) - 1.0), 0.0);
			long leftshift = (long)max(pow(2.0, l-1), 0.0);

			// --- Highpasses ---

			ctOneStepTime.Reset();
			// highpass row
			OUT_INFO("dwtForward: Running highpass kernel x ... ");
			filterDWTX<float, dwt_highpass_kernel_length, dwt_highpass_kernel_radius><<<grid_x, threads_x>>>
				(devSrcPtr, devDstPtr, sSize, zeroes_padding, one_over_lambda, leftshift, 1);


			cudaThreadSynchronize();
			cutilCheckMsg("dwtForward: Kernel execution failed");
			// Copy highpass data back to cpu memory
			devToHostCopyParams.dstPtr = make_cudaPitchedPtr(output->x.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.srcPtr = devDstPtr;
			devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
			OUT_INFO("copy to cpu ... ");
			cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));
			OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());

			ctOneStepTime.Reset();
			OUT_INFO("dwtForward: Running highpass kernel y ... ");
			filterDWTY<float, dwt_highpass_kernel_length, dwt_highpass_kernel_radius><<<grid_y, threads_y>>>
				(devSrcPtr, devDstPtr, sSize, zeroes_padding, one_over_lambda, leftshift, 1);

			cudaThreadSynchronize();
			cutilCheckMsg("dwtForward: Kernel execution failed");
			// Copy highpass data back to cpu memory
			devToHostCopyParams.dstPtr = make_cudaPitchedPtr(output->y.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.srcPtr = devDstPtr;
			devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
			OUT_INFO("copy to cpu ... ");
			cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));
			OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());

			ctOneStepTime.Reset();
			OUT_INFO("dwtForward: Running highpass kernel z ... ");
			filterDWTZ<float, dwt_highpass_kernel_length, dwt_highpass_kernel_radius><<<grid_z, threads_z>>>
				(devSrcPtr, devDstPtr, sSize, zeroes_padding, one_over_lambda, leftshift, 1);

			cudaThreadSynchronize();
			cutilCheckMsg("dwtForward: Kernel execution failed");
			// Copy highpass data back to cpu memory
			devToHostCopyParams.dstPtr = make_cudaPitchedPtr(output->z.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToHostCopyParams.srcPtr = devDstPtr;
			devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
			OUT_INFO("copy to cpu ... ");
			cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));
			OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());


			// --- Lowpasses ---
			ctOneStepTime.Reset();
			// highpass row
			OUT_INFO("dwtForward: Running lowpass kernel x ... ");
			filterDWTX<float, dwt_lowpass_kernel_length, dwt_lowpass_kernel_radius><<<grid_x, threads_x>>>
				(devSrcPtr, devDstPtr, sSize, zeroes_padding, 1.0f, leftshift, 0);
			cudaThreadSynchronize();

			OUT_INFO("y ... ");
			filterDWTY<float, dwt_lowpass_kernel_length, dwt_lowpass_kernel_radius><<<grid_y, threads_y>>>
				(devDstPtr, devSrcPtr, sSize, zeroes_padding, 1.0f, leftshift, 0);
			cudaThreadSynchronize();

			OUT_INFO("z ... ");
			filterDWTZ<float, dwt_lowpass_kernel_length, dwt_lowpass_kernel_radius><<<grid_z, threads_z>>>
				(devSrcPtr, devDstPtr, sSize, zeroes_padding, 1.0f, leftshift, 0);
			cudaThreadSynchronize();

			if(lowpass)  {
				// copy back the lowpass data to cpu if pointer was given
				devToHostCopyParams.dstPtr = make_cudaPitchedPtr(lowpass->getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
				devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
				devToHostCopyParams.srcPtr = devDstPtr;
				devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
				OUT_INFO("copy to cpu ... ");
				cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));
			}

			// TODO: Eleminate this memcpy through more intelligent usage of the two device buffers
			devToDevCopyParams.dstPtr = devSrcPtr;
			devToDevCopyParams.srcPtr = devDstPtr;
			devToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
			devToDevCopyParams.kind = cudaMemcpyDeviceToDevice;
			cutilSafeCall(cudaMemcpy3D(&devToDevCopyParams));
			cutilCheckMsg("dwtForward: Kernel execution failed");
			
			OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());
			OUT_INFO("dwtForward: Level %li done [%.5fs].\n", l, (float)ctOneLevelTime.Query());
		}
		DEBUG_OUT("dwtForward: Total transformation time (incl. copy) %.5fs.\n", (float)ctAllLevelsTime.Query());

		cutilSafeCall(cudaFree(devSrcPtr.ptr));
		cutilSafeCall(cudaFree(devDstPtr.ptr));

		if(lowpass)
			lowpass->m_bIsOk = true;
		output->x.m_bIsOk = true;
		output->y.m_bIsOk = true;
		output->z.m_bIsOk = true;

		return output;
	};
};};
