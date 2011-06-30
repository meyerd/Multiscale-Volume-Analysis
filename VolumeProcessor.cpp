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

#include "VolumeProcessor.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "cutil_replacement.h"

#include "dwtForward.h"
#include "modulusMaxima.h"
#include "calculateLipschitz.h"

VolumeProcessor::VolumeProcessor(void) : m_bCudaInitialized(false) {
	m_bCudaInitialized = initializeCuda();
}

VolumeProcessor::VolumeProcessor(int iDeviceId) : m_bCudaInitialized(false) {
	m_bCudaInitialized = initializeCuda(false, iDeviceId);
}

VolumeProcessor::~VolumeProcessor(void) {
	if(m_bCudaInitialized)
		cudaThreadExit();
}

bool VolumeProcessor::initializeCuda(bool bChooseFastest, int iDeviceId) {
	if(!m_bCudaInitialized) {
		int iMyDeviceID = iDeviceId;
		if(bChooseFastest) {
			iMyDeviceID = cutGetMaxGflopsDeviceId();
		}
		cudaSetDevice(iMyDeviceID);
		cutilSafeCall(cudaGetDevice(&(iMyDeviceID)));

		cudaDeviceProp sDeviceProperties;

		cutilSafeCall(cudaGetDeviceProperties(&(sDeviceProperties), iMyDeviceID));

		OUT_INFO("VolumeProcessor: Device %d: %s with compute %d.%d capability\n", iMyDeviceID,
			sDeviceProperties.name, sDeviceProperties.major, sDeviceProperties.minor);
		OUT_INFO("VolumeProcessor: Major revision number:         %d\n",  sDeviceProperties.major);
		OUT_INFO("VolumeProcessor: Minor revision number:         %d\n",  sDeviceProperties.minor);
		OUT_INFO("VolumeProcessor: Name:                          %s\n",  sDeviceProperties.name);
		OUT_INFO("VolumeProcessor: Total global memory:           %li\n",  sDeviceProperties.totalGlobalMem);
		OUT_INFO("VolumeProcessor: Total shared memory per block: %li\n",  sDeviceProperties.sharedMemPerBlock);
		OUT_INFO("VolumeProcessor: Total registers per block:     %d\n",  sDeviceProperties.regsPerBlock);
		OUT_INFO("VolumeProcessor: Warp size:                     %d\n",  sDeviceProperties.warpSize);
		OUT_INFO("VolumeProcessor: Maximum memory pitch:          %li\n",  sDeviceProperties.memPitch);
		OUT_INFO("VolumeProcessor: Maximum threads per block:     %d\n",  sDeviceProperties.maxThreadsPerBlock);
		for (int i = 0; i < 3; ++i)
			OUT_INFO("VolumeProcessor: Maximum dimension %d of block:  %d\n", i, sDeviceProperties.maxThreadsDim[i]);
		for (int i = 0; i < 3; ++i)
			OUT_INFO("VolumeProcessor: Maximum dimension %d of grid:   %d\n", i, sDeviceProperties.maxGridSize[i]);
		OUT_INFO("VolumeProcessor: Clock rate:                    %d\n",  sDeviceProperties.clockRate);
		OUT_INFO("VolumeProcessor: Total constant memory:         %li\n",  sDeviceProperties.totalConstMem);
		OUT_INFO("VolumeProcessor: Texture alignment:             %li\n",  sDeviceProperties.textureAlignment);
		OUT_INFO("VolumeProcessor: Concurrent copy and execution: %s\n",  (sDeviceProperties.deviceOverlap ? "Yes" : "No"));
		OUT_INFO("VolumeProcessor: Number of multiprocessors:     %d\n",  sDeviceProperties.multiProcessorCount);
		OUT_INFO("VolumeProcessor: Kernel execution timeout:      %s\n",  (sDeviceProperties.kernelExecTimeoutEnabled ? "Yes" : "No"));

	#ifdef WIN32
		if(cuInit(0) != CUDA_SUCCESS)
			throw VolumeProcessorError("cuInit failed.");
	#endif
	}

	return true;
}

Volume::MultilevelVolumeContainer<float, 3>* VolumeProcessor::dwtForward(const Volume::BasicVolume<float>& input, long lLevels, Volume::MultilevelVolume<float>* lowpass) {
	Cuda::DWT::dwtCheckSizes(input, lLevels);
	return Cuda::DWT::dwtForward(input, lLevels, lowpass);
}

Volume::MultilevelVolumeContainer<float, 3>* VolumeProcessor::modulusMaximaAngles(Volume::MultilevelVolumeContainer<float, 3>& input) {
	return Cuda::ModulusMaxima::calculateModulusAngles(input);
}

Volume::MultilevelVolumeContainer<float, 3>* VolumeProcessor::calculateLipschitz(Volume::MultilevelVolumeContainer<float, 3>& input) {
	return Cuda::Lipschitz::calculateLipschitz(input);
	//return CPU::Lipschitz::calculateLipschitz(input);
}