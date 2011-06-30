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

#include "calculateLipschitz.h"

#include "lipschitz_kernel.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "timing.h"
#include "cutil_replacement.h"

using namespace std;
using namespace Volume;

namespace Cuda { namespace Lipschitz {

/**
  Calculates the lipschitz/hölder exponent for each voxel.

  @param input The angles in the xz plane and the angle in y direction, the modulus maxima
 */
Volume::MultilevelVolumeContainer<float, 3>* calculateLipschitz(Volume::MultilevelVolumeContainer<float, 3>& input) {
	if(input.x.sSize != input.y.sSize || input.y.sSize != input.z.sSize)
		throw CudaLipschitzError("Input erroneous input sizes.");
	if(input.x.m_lLevels != input.y.m_lLevels || input.y.m_lLevels != input.z.m_lLevels)
		throw CudaLipschitzError("Input erroneous levels.");
	VolumeSize sSize = input.x.sSize;
	long lLevels = input.x.m_lLevels;
	MultilevelVolumeContainer<float, 3>* output = new MultilevelVolumeContainer<float, 3>(sSize, lLevels);

	// Allocate device memory for one level
	size_t devPtrStoragePitch = sizeof(cudaPitchedPtr);
	cudaPitchedPtr* devPtrModulusStorage = NULL;
	cudaPitchedPtr* devPtrAnglesXZStorage = NULL;
	cudaPitchedPtr* devPtrAnglesYStorage = NULL;
	cutilSafeCall(cudaMallocPitch<cudaPitchedPtr>(&devPtrModulusStorage, &devPtrStoragePitch, sizeof(cudaPitchedPtr), lLevels));
	cutilSafeCall(cudaMallocPitch<cudaPitchedPtr>(&devPtrAnglesXZStorage, &devPtrStoragePitch, sizeof(cudaPitchedPtr), lLevels));
	cutilSafeCall(cudaMallocPitch<cudaPitchedPtr>(&devPtrAnglesYStorage, &devPtrStoragePitch, sizeof(cudaPitchedPtr), lLevels));
	//cutilSafeCall(cudaMalloc<cudaPitchedPtr>(&devPtrStorage, sizeof(cudaPitchedPtr)*lLevels));
	cudaPitchedPtr* devSrcModulusPtr = new cudaPitchedPtr[lLevels];
	cudaPitchedPtr* devSrcAnglesXZPtr = new cudaPitchedPtr[lLevels];
	cudaPitchedPtr* devSrcAnglesYPtr = new cudaPitchedPtr[lLevels];
	cudaPitchedPtr devDstPtr = {0};

	OUT_INFO("calculateLipschitz: Allocating device memory >=%li bytes ...\n", sizeof(float)*sSize.x*sSize.y*sSize.z*(lLevels+1)*3);
	for(long l = 0; l < lLevels; ++l) {
		cutilSafeCall(cudaMalloc3D(&(devSrcModulusPtr[l]), make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));
		cutilSafeCall(cudaMalloc3D(&(devSrcAnglesXZPtr[l]), make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));
		cutilSafeCall(cudaMalloc3D(&(devSrcAnglesYPtr[l]), make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));
	}
	cutilSafeCall(cudaMalloc3D(&devDstPtr, make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z)));
	cutilSafeCall(cudaMemcpy2D(devPtrModulusStorage, devPtrStoragePitch, devSrcModulusPtr, sizeof(cudaPitchedPtr), 
		sizeof(cudaPitchedPtr), lLevels, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy2D(devPtrAnglesXZStorage, devPtrStoragePitch, devSrcAnglesXZPtr, sizeof(cudaPitchedPtr), 
		sizeof(cudaPitchedPtr), lLevels, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy2D(devPtrAnglesYStorage, devPtrStoragePitch, devSrcAnglesYPtr, sizeof(cudaPitchedPtr), 
		sizeof(cudaPitchedPtr), lLevels, cudaMemcpyHostToDevice));
	//cutilSafeCall(cudaMemcpy(devPtrStorage, devSrcModulusPtr, sizeof(cudaPitchedPtr), cudaMemcpyHostToDevice));

	dim3 threads_lipschitz(lipschitzBlockSize.x, lipschitzBlockSize.y);
	dim3 grid_lipschitz((sSize.x + lipschitzBlockSize.x - 1) / lipschitzBlockSize.x,
			(sSize.y * sSize.z + lipschitzBlockSize.y - 1) / lipschitzBlockSize.y);

	cudaMemcpy3DParms devToHostCopyParams = {0};
	cudaMemcpy3DParms hostToDevCopyParams = {0};

	// Copy data to device
	OUT_INFO("calculateLipschitz: Copying input data to device ...\n");
	for(long l = 0; l < lLevels; l++) {
		// Modulus
		hostToDevCopyParams.dstPtr = devSrcModulusPtr[l];
		hostToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
		hostToDevCopyParams.srcPtr = make_cudaPitchedPtr(input.z.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
		hostToDevCopyParams.kind = cudaMemcpyHostToDevice;
		cutilSafeCall(cudaMemcpy3D(&hostToDevCopyParams));
		// Angles XZ
		hostToDevCopyParams.dstPtr = devSrcAnglesXZPtr[l];
		hostToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
		hostToDevCopyParams.srcPtr = make_cudaPitchedPtr(input.x.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
		hostToDevCopyParams.kind = cudaMemcpyHostToDevice;
		cutilSafeCall(cudaMemcpy3D(&hostToDevCopyParams));
		// Angles Y
		hostToDevCopyParams.dstPtr = devSrcAnglesYPtr[l];
		hostToDevCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
		hostToDevCopyParams.srcPtr = make_cudaPitchedPtr(input.y.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
		hostToDevCopyParams.kind = cudaMemcpyHostToDevice;
		cutilSafeCall(cudaMemcpy3D(&hostToDevCopyParams));
	}

	CTimer ctAllLevelsTime;
	CTimer ctOneLevelTime;
	CTimer ctOneStepTime;

	ctAllLevelsTime.Reset();

	for(long l = 0; l < lLevels; l++) {
		ctOneLevelTime.Reset();
		ctOneStepTime.Reset();
		OUT_INFO("calculateLipschitz: Running lipschitz kernel ... ");
		calculateLipschitz<float, 5><<<grid_lipschitz, threads_lipschitz>>>(devPtrModulusStorage, devPtrAnglesXZStorage,
			devPtrAnglesYStorage, devPtrStoragePitch / sizeof(cudaPitchedPtr), devDstPtr, sSize, l, lLevels);
		cudaThreadSynchronize();
		cutilCheckMsg("calculateLipschitz: Kernel execution failed");
		devToHostCopyParams.dstPtr = make_cudaPitchedPtr(output->x.getLevel(l), sizeof(float)*sSize.x, sSize.y, sSize.z);
		devToHostCopyParams.extent = make_cudaExtent(sizeof(float)*sSize.x, sSize.y, sSize.z);
		devToHostCopyParams.srcPtr = devDstPtr;
		devToHostCopyParams.kind = cudaMemcpyDeviceToHost;
		OUT_INFO("copy to cpu ... ");
		cutilSafeCall(cudaMemcpy3D(&devToHostCopyParams));
		OUT_INFO("done [%.5fs].\n", (float)ctOneStepTime.Query());

		OUT_INFO("calculateLipschitz: Level %li done [%.5fs].\n", l, (float)ctOneLevelTime.Query());
	}
	DEBUG_OUT("calculateLipschitz: Total time (incl. copy) %.5fs.\n", (float)ctAllLevelsTime.Query());

	cutilSafeCall(cudaFree(devPtrModulusStorage));
	cutilSafeCall(cudaFree(devPtrAnglesXZStorage));
	cutilSafeCall(cudaFree(devPtrAnglesYStorage));
	for(long l = 0; l < lLevels; ++l) {
		cutilSafeCall(cudaFree(devSrcModulusPtr[l].ptr));
		cutilSafeCall(cudaFree(devSrcAnglesXZPtr[l].ptr));
		cutilSafeCall(cudaFree(devSrcAnglesYPtr[l].ptr));
	}
	SAFE_DELETE_ARRAY(devSrcModulusPtr);
	SAFE_DELETE_ARRAY(devSrcAnglesXZPtr);
	SAFE_DELETE_ARRAY(devSrcAnglesYPtr);
	cutilSafeCall(cudaFree(devDstPtr.ptr));
	output->x.m_bIsOk = true;
	return output;
}
};};