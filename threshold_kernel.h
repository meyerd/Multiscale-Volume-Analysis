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

#ifndef __THRESHOLD_KERNEL_H__
#define __THRESHOLD_KERNEL_H__

#include "global.h"

#include <math_constants.h>
#include <cuda.h>

#include "GeneralVolume.h"

namespace Cuda { namespace ModulusMaxima {

/* TODO: this does not work on linux, especially the allocation of the __shared__ mem ... maybe problem with nvcc compiler */
const Volume::ConstVolumeSize<16, 16, 1> thresholdBlockSize;
// TODO: get rid of these defines (only to fix linux compile error)
#define thresholdBlockSizex	16
#define thresholdBlockSizey	16
#define thresholdBlockSizez	1

/* TODO: this does not work on linux, especially the allocation of the __shared__ mem ... maybe problem with nvcc compiler */
const Volume::ConstVolumeSize<16, 16, 1> thresholdMeanBlockSize;
// TODO: get rid of these defines (only to fix linux compile error)
#define thresholdMeanBlockSizex	16
#define thresholdMeanBlockSizey	16
#define thresholdMeanBlockSizez	1

template<typename T>
__global__ void thresholdModulus(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size, float threshold, int type) 
{
	const long inputRowPitch = src.pitch / sizeof(T);
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dst.pitch / sizeof(T);
	const long outputSlicePitch = size.y * outputRowPitch;

	const long blocksPerSliceY = size.y / thresholdBlockSize.y;

	const long basex = (long)blockIdx.x * thresholdBlockSize.x + (long)threadIdx.x;
	const long basey = ((long)blockIdx.y % blocksPerSliceY) * thresholdBlockSize.y + (long)threadIdx.y;
	const long basez = (long)blockIdx.y / blocksPerSliceY;

	const long inputOffset = basez * inputSlicePitch + basey * inputRowPitch + basex;
	const long outputOffset = basez * outputSlicePitch + basey * outputRowPitch + basex;

	T tmp = *(((T*)src.ptr) + inputOffset);

	if(type == 1) {
		if(tmp >= threshold) {
			*(((T*)dst.ptr) + outputOffset) = tmp - threshold;
		} else if(tmp <= threshold) {
			*(((T*)dst.ptr) + outputOffset) = tmp + threshold;
		} else /* abs(tmp) <= threshold */ {
			*(((T*)dst.ptr) + outputOffset) = 0.0f;
		}
	} else {
		*(((T*)dst.ptr) + outputOffset) = (abs(tmp) >= threshold ? tmp : 0.0f);
	}
};

template<typename T>
__global__ void findMeanPass1(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size) {
	// loop over the y dimension, the thread grid is distributed over the x/z plane
	// the y-threadIdx and blockIdx parameter is actually the z dimension in the first pass

	const long inputRowPitch = src.pitch / sizeof(T);
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dst.pitch / sizeof(T);

	const long basex = (long)blockIdx.x * thresholdMeanBlockSize.x + (long)threadIdx.x;
	const long basez = (long)blockIdx.y * thresholdMeanBlockSize.y + (long)threadIdx.y;

	T mean = 0.0f;

	// loop over y dimension
	for(long basey = 0; basey < size.y; basey++) {
		mean += (1.0f / (float)size.y) * *(((T*)src.ptr) + basez * inputSlicePitch + basey * inputRowPitch + basex);
	}

	*(((T*)dst.ptr) + basez * outputRowPitch + basex) = mean;
};

template<typename T>
__global__ void findMeanPass2(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size) {
	// loop over the the previously generated x/z plane in z direction
	const long inputRowPitch = src.pitch / sizeof(T);
	
	const long basex = (long)blockIdx.x * thresholdMeanBlockSize.x + (long)threadIdx.x;
	
	T mean = 0.0f;
	// loop over z dimension
	for(long basey = 0; basey < size.z; basey++) {
		mean += (1.0f / (float)size.z) * *(((T*)src.ptr) + basey * inputRowPitch + basex);
	}

	*(((T*)dst.ptr) + basex) = mean;
};
template<typename T>
__global__ void findMeanPass3(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size) {
	// loop over the previously generated vector (x direction)

	T mean = 0.0f;
	for(long basex = 0; basex < size.x; basex++) {
		mean += (1.0f / (float)(size.x)) * *(((T*)src.ptr) + basex);
	}

	*(((T*)dst.ptr)) = mean;
};

template<typename T>
__global__ void findVariancePass1(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size, T mean) {
	// loop over the y dimension, the thread grid is distributed over the x/z plane
	// the y-threadIdx and blockIdx parameter is actually the z dimension in the first pass

	const long inputRowPitch = src.pitch / sizeof(T);
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dst.pitch / sizeof(T);

	const long basex = (long)blockIdx.x * thresholdMeanBlockSize.x + (long)threadIdx.x;
	const long basez = (long)blockIdx.y * thresholdMeanBlockSize.y + (long)threadIdx.y;

	T varsq = 0.0f;
	
	// loop over y dimension
	for(long basey = 0; basey < size.y; basey++) {
		varsq += (1.0f / (float)size.y) * square(*(((T*)src.ptr) + basez * inputSlicePitch + basey * inputRowPitch + basex) - mean);
	}

	*(((T*)dst.ptr) + basez * outputRowPitch + basex) = varsq;
};

template<typename T>
__global__ void findVariancePass2(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size) {
	// loop over the the previously generated x/z plane in z direction
	const long inputRowPitch = src.pitch / sizeof(T);

	const long basex = (long)blockIdx.x * thresholdMeanBlockSize.x + (long)threadIdx.x;
	
	T varsq = 0.0f;
	
	// loop over z dimension
	for(long basey = 0; basey < size.z; basey++) {
		varsq += (1.0f / (float)size.z) * *(((T*)src.ptr) + basey * inputRowPitch + basex);
	}

	*(((T*)dst.ptr) + basex) = varsq;
};
template<typename T>
__global__ void findVariancePass3(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size) {
	// loop over the previously generated vector (x direction)

	T varsq = 0.0f;

	for(long basex = 0; basex < size.x; basex++) {
		varsq += (1.0f / (float)(size.x)) * *(((T*)src.ptr) + basex);
	}

	*(((T*)dst.ptr)) = varsq;
};

};};

#endif /* __THRESHOLD_KERNEL_H__ */