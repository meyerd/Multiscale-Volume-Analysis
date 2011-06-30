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

#ifndef __MAXIMA_KERNEL_H__
#define __MAXIMA_KERNEL_H__

#include "global.h"

#include <math_constants.h>
#include <cuda.h>

#include "GeneralVolume.h"

namespace Cuda { namespace ModulusMaxima {

/* TODO: this does not work on linux, especially the allocation of the __shared__ mem ... maybe problem with nvcc compiler */
const Volume::ConstVolumeSize<16, 16, 1> maximaBlockSize;
// TODO: get rid of these defines (only to fix linux compile error)
#define maximaBlockSizex	16
#define maximaBlockSizey	16
#define maximaBlockSizez	1 

#define myPI (CUDART_PI_F)

#define PI_0_8 (0.0f)
#define PI_1_8 (myPI/8.0f)
#define PI_3_8 (3.0f*myPI/8.0f)
#define PI_5_8 (5.0f*myPI/8.0f)
#define PI_7_8 (7.0f*myPI/8.0f)
#define PI_8_8 (myPI)

#define PI_0_4 (0.0f)
#define PI_1_4 (myPI/4.0f)
#define PI_2_4 (2.0f*myPI/4.0f)
#define PI_3_4 (3.0f*myPI/4.0f)
#define PI_4_4 (myPI)

#define mPI_0_8 (-0.0f)
#define mPI_1_8 (-myPI/8.0f)
#define mPI_3_8 (-3.0f*myPI/8.0f)
#define mPI_5_8 (-5.0f*myPI/8.0f)
#define mPI_7_8 (-7.0f*myPI/8.0f)
#define mPI_8_8 (-myPI)

#define mPI_0_4 (-0.0f)
#define mPI_1_4 (-myPI/4.0f)
#define mPI_2_4 (-2.0f*myPI/4.0f)
#define mPI_3_4 (-3.0f*myPI/4.0f)
#define mPI_4_4 (-myPI)

// TODO: is this as fast as a define would be??
__device__ __inline__ long clampLeft(const long& x) {
	return max((long long)x, (long long)0);
};

__device__ __inline__ long clampRight(const long& x, const long& width) {
	return min((long long)width, (long long)x);
};

__device__ __inline__ long clampBoth(const long& x, const long& width) {
	return min(max((long long)x, (long long)0), (long long)width);
};

template<typename T>
__global__ void findModulusMaxima(cudaPitchedPtr anglesxz, cudaPitchedPtr anglesy, cudaPitchedPtr modulus, cudaPitchedPtr dst, 
	const Volume::VolumeSize size) 
{
	const long inputRowPitch = anglesxz.pitch / sizeof(T);
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dst.pitch / sizeof(T);
	const long outputSlicePitch = size.y * outputRowPitch;

	const long blocksPerSliceY = size.y / maximaBlockSize.y;

	__shared__ T axz[maximaBlockSizey][maximaBlockSizex];
	__shared__ T ay[maximaBlockSizey][maximaBlockSizex];
	// 3 z-slices, +2 in x and y direction for 1 voxel border
	__shared__ T m[3][maximaBlockSizey+2][maximaBlockSizex+2];

	const long basex = (long)blockIdx.x * maximaBlockSize.x + (long)threadIdx.x;
	const long basey = ((long)blockIdx.y % blocksPerSliceY) * maximaBlockSize.y + (long)threadIdx.y;
	const long basez = (long)blockIdx.y / blocksPerSliceY;

	const long inputOffset = basez * inputSlicePitch + basey * inputRowPitch + basex;
	const long outputOffset = basez * outputSlicePitch + basey * outputRowPitch + basex;

	// fill shared memory angles
	axz[threadIdx.y][threadIdx.x] = *(((T*)anglesxz.ptr) + inputOffset);
	ay[threadIdx.y][threadIdx.x] = *(((T*)anglesy.ptr) + inputOffset);
	// fill shared memory modulus (incl. border)
	// for each z slice
	#pragma unroll
	for(long z = -1; z <= 1; z++) {
		// top left corner
		if(threadIdx.x == 0 && threadIdx.y == 0) {
			const long index = clampBoth(basez + z, size.z-1) * inputSlicePitch + clampLeft(basey - 1) * inputRowPitch + clampLeft(basex - 1);
			m[z+1][threadIdx.y][threadIdx.x] = *(((T*)modulus.ptr) + index);
		}
		// top border
		if(threadIdx.y == 0) {
			const long index = clampBoth(basez + z, size.z-1) * inputSlicePitch + clampLeft(basey - 1) * inputRowPitch + basex;
			m[z+1][threadIdx.y][threadIdx.x+1] = *(((T*)modulus.ptr) + index);
		}
		// top right corner
		if(threadIdx.x == (maximaBlockSize.x - 1) && threadIdx.y == 0) {
			const long index = clampBoth(basez + z, size.z-1) * inputSlicePitch + clampLeft(basey - 1) * inputRowPitch + clampRight(basex + 1, size.x-1);
			m[z+1][threadIdx.y][threadIdx.x+2] = *(((T*)modulus.ptr) + index);
		}
		// left border
		if(threadIdx.x == 0) {
			const long index = clampBoth(basez + z, size.z-1) * inputSlicePitch + basey * inputRowPitch + clampLeft(basex - 1);
			m[z+1][threadIdx.y+1][threadIdx.x] = *(((T*)modulus.ptr) + index);
		}
		// center
		{
			const long index = clampBoth(basez + z, size.z-1) * inputSlicePitch + basey * inputRowPitch + basex;
			m[z+1][threadIdx.y+1][threadIdx.x+1] = *(((T*)modulus.ptr) + index);
		}
		// right border
		if(threadIdx.x == (maximaBlockSize.x - 1)) {
			const long index = clampBoth(basez + z, size.z-1) * inputSlicePitch + basey * inputRowPitch + clampRight(basex + 1, size.x-1);
			m[z+1][threadIdx.y+1][threadIdx.x+2] = *(((T*)modulus.ptr) + index);
		}
		// bottom left corner
		if(threadIdx.x == 0 && threadIdx.y == (maximaBlockSize.y - 1)) {
			const long index = clampBoth(basez + z, size.z-1) * inputSlicePitch + clampRight(basey + 1, size.y-1) * inputRowPitch + clampLeft(basex - 1);
			m[z+1][threadIdx.y+2][threadIdx.x] = *(((T*)modulus.ptr) + index);
		}
		// bottom border
		if(threadIdx.y == (maximaBlockSize.y - 1)) {
			const long index = clampBoth(basez + z, size.z-1) * inputSlicePitch + clampRight(basey + 1, size.y-1) * inputRowPitch + basex;
			m[z+1][threadIdx.y+2][threadIdx.x+1] = *(((T*)modulus.ptr) + index);
		}
		// bottom right corner
		if(threadIdx.x == (maximaBlockSize.x - 1) && threadIdx.y == (maximaBlockSize.y - 1)) {
			const long index = clampBoth(basez + z, size.z-1) * inputSlicePitch + clampRight(basey + 1, size.y-1) * inputRowPitch + clampRight(basex + 1, size.x-1);
			m[z+1][threadIdx.y+2][threadIdx.x+2] = *(((T*)modulus.ptr) + index);
		}
	}

	__syncthreads();

	int3 offs_left = make_int3(0, 0, 0);
	int3 offs_right = make_int3(0, 0, 0);

	float anglexz = axz[threadIdx.y][threadIdx.x];
	float angley = ay[threadIdx.y][threadIdx.x];
	if(anglexz >= PI_7_8) {
		offs_left.x = 1;
		offs_right.x = -1;
		offs_left.z = 0;
		offs_right.z = 0;
	} else if(anglexz >= PI_5_8 && anglexz < PI_7_8) {
		offs_left.x = 1;
		offs_right.x = -1;
		offs_left.z = -1;
		offs_right.z = 1;
	} else if(anglexz >= PI_3_8 && anglexz < PI_5_8) {
		offs_left.x = 0;
		offs_right.x = 0;
		offs_left.z = -1;
		offs_right.z = 1;
	} else if(anglexz >= PI_1_8 && anglexz < PI_3_8) {
		offs_left.x = -1;
		offs_right.x = 1;
		offs_left.z = -1;
		offs_right.z = 1;
	} else if(anglexz >= PI_0_8 && anglexz < PI_1_8) {
		offs_left.x = -1;
		offs_right.x = 1;
		offs_left.z = 0;
		offs_right.z = 0;
	} else if(anglexz <= PI_0_8 && anglexz > mPI_1_8) {
		offs_left.x = -1;
		offs_right.x = 1;
		offs_left.z = 0;
		offs_right.z = 0;
	} else if(anglexz <= mPI_1_8 && anglexz > mPI_3_8) {
		offs_left.x = -1;
		offs_right.x = 1;
		offs_left.z = 1;
		offs_right.z = -1;
	} else if(anglexz <= mPI_3_8 && anglexz > mPI_5_8) {
		offs_left.x = 0;
		offs_right.x = 0;
		offs_left.z = 1;
		offs_right.z = -1;
	} else if(anglexz <= mPI_5_8 && anglexz > mPI_7_8) {
		offs_left.x = 1;
		offs_right.x = -1;
		offs_left.z = 1;
		offs_right.z = -1;
	} else /* (anglexz <= mPI_7_8) */ {
		offs_left.x = 1;
		offs_right.x = -1;
		offs_left.z = 0;
		offs_right.z = 0;
	}
	if(angley >= PI_7_8) {
		offs_left.y = 0;
		offs_right.y = 0;
	} else if(angley >= PI_5_8 && angley < PI_7_8) {
		offs_left.y = -1;
		offs_right.y = 1;
	} else if(angley >= PI_3_8 && angley < PI_5_8) {
		offs_left.y = -1;
		offs_right.y = 1;
		offs_left.x = 0;
		offs_right.x = 0;
		offs_left.z = 0;
		offs_right.z = 0;
	} else if(angley >= PI_1_8 && angley < PI_3_8) {
		offs_left.y = -1;
		offs_right.y = 1;
	} else if(angley >= PI_0_8 && angley < PI_1_8) {
		offs_left.y = 0;
		offs_right.y = 0;
	} else if(angley <= PI_0_8 && angley > mPI_1_8) {
		offs_left.y = 0;
		offs_right.y = 0;
	} else if(angley <= mPI_1_8 && angley > mPI_3_8) {
		offs_left.y = 1;
		offs_right.y = -1;
	} else if(angley <= mPI_3_8 && angley > mPI_5_8) {
		offs_left.y = 1;
		offs_right.y = -1;
		offs_left.x = 0;
		offs_right.x = 0;
		offs_left.z = 0;
		offs_right.z = 0;
	} else if(angley <= mPI_5_8 && angley > mPI_7_8) {
		offs_left.y = 1;
		offs_right.y = -1;
	} else /* (angley <= mPI_7_8) */ {
		offs_left.y = 0;
		offs_right.y = 0;
	}

	float at = m[1][threadIdx.y + 1][threadIdx.x + 1];
	float left = m[offs_left.z + 1][threadIdx.y + 1 + offs_left.y][threadIdx.x + 1 + offs_left.x];
	float right = m[offs_right.z + 1][threadIdx.y + 1 + offs_right.y][threadIdx.x + 1 + offs_right.x];

	if(((at >= left) && (at >= right)) && ((at > left) || (at > right))) {
		*(((T*)dst.ptr) + outputOffset) = at;
	} else {
		*(((T*)dst.ptr) + outputOffset) = 0.0f;
	}

	//*(((T*)dst.ptr) + outputOffset) = at;
};
};};

#endif /* __MAXIMA_KERNEL_H__ */
