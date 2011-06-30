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

#ifndef __2DDWT_KERNEL_H__
#define __2DDWT_KERNEL_H__

#include "global.h"

#include <cuda.h>

#include "GeneralVolume.h"

namespace Cuda { namespace DWT {

/* TODO: this does not work on linux, especially the allocation of the __shared__ mem ... maybe problem with nvcc compiler */
const Volume::ConstVolumeSize<32, 4, 1> dwtXBlockSize;
// TODO: get rid of these defines (only to fix linux compile error)
#define dwtXBlockSizex		32
#define dwtXBlockSizey		4
#define dwtXBlockSizez		1 
const long DWT_X_STEPS = 4;
const long DWT_X_HALO = 2;

const Volume::ConstVolumeSize<8, 8, 1> dwtYBlockSize;
// TODO: get rid of these defines (only to fix linux compile error)
#define dwtYBlockSizex	8
#define dwtYBlockSizey	8
#define dwtYBlockSizez	1
const long DWT_Y_STEPS = 2;
const long DWT_Y_HALO = 5;

const Volume::ConstVolumeSize<8, 8, 1> dwtZBlockSize;
// TODO: get rid of these defines (only to fix linux compile error)
#define dwtZBlockSizex	8
#define dwtZBlockSizey	8
#define dwtZBlockSizez	1
const long DWT_Z_STEPS = 2;
const long DWT_Z_HALO = 5;

// kernels are assumed to be odd length centered

/* spline kernel */
__constant__ float dwt_highpass_kernel[] = {0.0f, -2.0f, 2.0f};
const int dwt_highpass_kernel_length = 3; 
__constant__ float dwt_lowpass_kernel[] = {0.0f, 0.125f, 0.375f, 0.375f, 0.125f};
const int dwt_lowpass_kernel_length = 5;
/* daubechies */
/* float cuda_helper_2ddwt_mallat_h[] = {-0.125f, 0.25f, 0.75f, 0.25f, -0.125f};
int cuda_helper_2ddwt_mallat_h_length = 5;
float cuda_helper_2ddwt_mallat_g[] = {-0.25f, 0.5f, -0.25f};
int cuda_helper_2ddwt_mallat_g_length = 3; */

const int dwt_highpass_kernel_radius = dwt_highpass_kernel_length / 2;
const int dwt_lowpass_kernel_radius = dwt_lowpass_kernel_length / 2;

/* How to obtain filter for specific level?
	 insert 2^level - 1 zeroes between filter coefficients
 */

/* stolen from marc ;) */
__device__ __inline__ long mirrorLeft(long index) {
	return abs(index);
}
__device__ __inline__ long mirrorRight(long index, long size) {
	return (size-1) - abs((size-1) - index);
/*	if(index >= size)
		return size * 2 - index - 1;
	return index; */
}

__device__ __inline__ long wrapRight(long index, long size) {
	if(index >= size) {
		return index - size;
	}
	return index;
}

template<typename T, int kernel_length, int kernel_radius>
__global__ void filterDWTX(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size, int zeroes, 
	float one_over_lambda, long leftshift, int lh)
{
	float* kernel = dwt_lowpass_kernel;
	if(lh)
		kernel = dwt_highpass_kernel;

	const long inputRowPitch = src.pitch / sizeof(T);
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dst.pitch / sizeof(T);
	const long outputSlicePitch = size.y * outputRowPitch;

	const long blocksPerSliceY = size.y / dwtXBlockSize.y;

	// shared memory: MALLAT_ROW_STEPS + 2 "halo" tiles x extension 
	//   (one thread processes MALLAT_ROW_STEPS pixels)
	__shared__ T s[dwtXBlockSizey][dwtXBlockSizex * (DWT_X_STEPS + 2 * DWT_X_HALO)];

	const long basex = ((long)blockIdx.x * DWT_X_STEPS - DWT_X_HALO) * dwtXBlockSize.x + (long)threadIdx.x;
	const long basey = ((long)blockIdx.y % blocksPerSliceY) * dwtXBlockSize.y + (long)threadIdx.y;
	const long basez = (long)blockIdx.y / blocksPerSliceY;

	// offset into correct row
	const long inputOffset = basez * inputSlicePitch + basey * inputRowPitch;
	const long outputOffset = basez * outputSlicePitch + basey * outputRowPitch;

	// left "halo"
	#pragma unroll
	for(int i = 0; i < DWT_X_HALO; i++) {
		const long xindex = basex + i * dwtXBlockSize.x;
		T tmp = *(((T*)src.ptr) + (mirrorLeft(xindex) + inputOffset));
		s[threadIdx.y][threadIdx.x + i * dwtXBlockSize.x] = tmp;
	}

	// main data
	#pragma unroll
	for(int i = DWT_X_HALO; i < DWT_X_STEPS + DWT_X_HALO; i++) {
		const long xindex = basex + i * dwtXBlockSize.x;
		T tmp = *(((T*)src.ptr) + (xindex + inputOffset));
		s[threadIdx.y][threadIdx.x + i * dwtXBlockSize.x] = tmp;
	}

	// right "halo"
	#pragma unroll
	for(int i = DWT_X_STEPS + DWT_X_HALO; i < DWT_X_HALO + DWT_X_STEPS + DWT_X_HALO; i++) {
		const long xindex = basex + i * dwtXBlockSize.x;
		T tmp = *(((T*)src.ptr) + (mirrorRight(xindex, size.x) + inputOffset));
		s[threadIdx.y][threadIdx.x + i * dwtXBlockSize.x] = tmp;
	}

	__syncthreads();

	#pragma unroll
	for(int i = DWT_X_HALO; i < DWT_X_STEPS + DWT_X_HALO; i++) {
		float sum = 0.0f;

		#pragma unroll
		for(int k = -kernel_radius; k <= kernel_radius; k++) {
			sum += kernel[k + kernel_radius] *
				s[threadIdx.y][threadIdx.x + i * dwtXBlockSize.x + k + k * zeroes - leftshift];
		}

		*(((T*)dst.ptr) + (basex + outputOffset + (i * dwtXBlockSize.x))) = one_over_lambda * sum;
	} 
};

template<typename T, int kernel_length, int kernel_radius>
__global__ void filterDWTY(cudaPitchedPtr src,
	cudaPitchedPtr dst, const Volume::VolumeSize size, int zeroes, 
	float one_over_lambda, long leftshift, int lh)
{
	float* kernel = dwt_lowpass_kernel;
	if(lh)
		kernel = dwt_highpass_kernel;

	const long inputRowPitch = src.pitch / sizeof(T);
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dst.pitch / sizeof(T);
	const long outputSlicePitch = size.y * outputRowPitch;

	const long blocksPerSliceY = size.y / dwtYBlockSize.y;

	__shared__ T s[dwtYBlockSizex][(DWT_Y_STEPS + 2 * DWT_Y_HALO) * dwtYBlockSizey + 1];

	const long basex = (long)blockIdx.x * dwtYBlockSize.x + (long)threadIdx.x;
	const long basey = (((long)blockIdx.y * DWT_Y_STEPS) % blocksPerSliceY) * dwtYBlockSize.y + (long)threadIdx.y - DWT_Y_HALO * dwtYBlockSize.y;
	const long basez = ((long)blockIdx.y * DWT_Y_STEPS) / blocksPerSliceY;

	// offset into correct column
	const long inputOffset = basez * inputSlicePitch + basex;
	const long outputOffset = basez * outputSlicePitch + basex;

	// left "halo" blocks
	#pragma unroll
	for(int i = 0; i < DWT_Y_HALO; i++) {
		const long yindex = basey + i * dwtYBlockSize.y;
		T tmp = *(((T*)src.ptr) + (mirrorLeft(yindex) * inputRowPitch + inputOffset));
		s[threadIdx.x][threadIdx.y + i * dwtYBlockSize.y] =	tmp;
	}

	// main data
	#pragma unroll
	for(int i = DWT_Y_HALO; i < DWT_Y_STEPS + DWT_Y_HALO; i++) {
		const long yindex = basey + i * dwtYBlockSize.y;
		T tmp = *(((T*)src.ptr) + (yindex * inputRowPitch + inputOffset));
		s[threadIdx.x][threadIdx.y + i * dwtYBlockSize.y] = tmp;
	}

	// right "halo" blocks
	#pragma unroll
	for(int i = DWT_Y_STEPS + DWT_Y_HALO; i < DWT_Y_HALO + DWT_Y_STEPS + DWT_Y_HALO; i++) {
		const long yindex = basey + i * dwtYBlockSize.y;
		T tmp = *(((T*)src.ptr) + (mirrorRight(yindex, size.y) * inputRowPitch + inputOffset));
		s[threadIdx.x][threadIdx.y + i * dwtYBlockSize.y] = tmp;
	}

	__syncthreads();

	#pragma unroll
	for(int i = DWT_Y_HALO; i < DWT_Y_STEPS + DWT_Y_HALO; i++) {
		float sum = 0;

		#pragma unroll
		for(int k = -kernel_radius; k <= kernel_radius; k++) {
			sum += kernel[k + kernel_radius] *
				s[threadIdx.x][threadIdx.y + i * dwtYBlockSize.y + k + k * zeroes - leftshift];
		}

		*(((T*)dst.ptr) + ((basey + i * dwtYBlockSize.y) * outputRowPitch + outputOffset)) = one_over_lambda * sum;
	}
};

template<typename T, int kernel_length, int kernel_radius>
__global__ void filterDWTZ(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size, int zeroes, 
	float one_over_lambda, long leftshift, int lh)
{
	float* kernel = dwt_lowpass_kernel;
	if(lh)
		kernel = dwt_highpass_kernel;

	const long inputRowPitch = src.pitch / sizeof(T);
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dst.pitch / sizeof(T);
	const long outputSlicePitch = size.y * outputRowPitch;

	const long blocksPerRowX = size.x / dwtZBlockSize.x;

	__shared__ T s[dwtZBlockSizex][(DWT_Y_STEPS + 2 * DWT_Y_HALO) * dwtZBlockSizey + 1];

	const long basex = ((long)blockIdx.x % blocksPerRowX) * dwtZBlockSize.x + (long)threadIdx.x;
	const long basey = (long)blockIdx.x / blocksPerRowX;
	const long basez = ((long)blockIdx.y * DWT_Y_STEPS - DWT_Y_HALO) * dwtZBlockSize.y + (long)threadIdx.y;

	// offset into correct row and column
	const long inputOffset = basey * inputRowPitch + basex;
	const long outputOffset = basey * outputRowPitch + basex;

	// left "halo" blocks
	#pragma unroll
	for(int i = 0; i < DWT_Y_HALO; i++) {
		const long zindex = basez + i * dwtZBlockSize.y;
		T tmp = *(((T*)src.ptr) + (mirrorLeft(zindex) * inputSlicePitch + inputOffset));
		s[threadIdx.x][threadIdx.y + i * dwtZBlockSize.y] =	tmp;
	}

	// main data
	#pragma unroll
	for(int i = DWT_Y_HALO; i < DWT_Y_STEPS + DWT_Y_HALO; i++) {
		const long zindex = basez + i * dwtZBlockSize.y;
		T tmp = *(((T*)src.ptr) + (zindex * inputSlicePitch + inputOffset));
		s[threadIdx.x][threadIdx.y + i * dwtZBlockSize.y] = tmp;
	}

	// right "halo" blocks
	#pragma unroll
	for(int i = DWT_Y_STEPS + DWT_Y_HALO; i < DWT_Y_HALO + DWT_Y_STEPS + DWT_Y_HALO; i++) {
		const long zindex = basez + i * dwtZBlockSize.y;
		T tmp = *(((T*)src.ptr) + (mirrorRight(zindex, size.z) * inputSlicePitch + inputOffset));
		s[threadIdx.x][threadIdx.y + i * dwtZBlockSize.y] = tmp;
	}

	__syncthreads();

	#pragma unroll
	for(int i = DWT_Y_HALO; i < DWT_Y_STEPS + DWT_Y_HALO; i++) {
		float sum = 0;

		#pragma unroll
		for(int k = -kernel_radius; k <= kernel_radius; k++) {
			sum += kernel[k + kernel_radius] *
				s[threadIdx.x][threadIdx.y + i * dwtZBlockSize.y + k + k * zeroes - leftshift];
		}

		*(((T*)dst.ptr) + ((basez + i * dwtZBlockSize.y) * outputSlicePitch + outputOffset)) = one_over_lambda * sum;
	}
};
};};


#endif /* __2DDWT_KERNEL_H__ */
