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

#ifndef __MODULUS_KERNEL_H__
#define __MODULUS_KERNEL_H__

#include "global.h"

#include <cuda.h>

#include "GeneralVolume.h"

namespace Cuda { namespace ModulusMaxima {

/* TODO: this does not work on linux, especially the allocation of the __shared__ mem ... maybe problem with nvcc compiler */
const Volume::ConstVolumeSize<16, 16, 1> modulusBlockSize;
// TODO: get rid of these defines (only to fix linux compile error)
#define modulusBlockSizex		16
#define modulusBlockSizey		16
#define modulusBlockSizez		1 

template<typename T>
__global__ void calculateModulus(cudaPitchedPtr srcx, cudaPitchedPtr srcy, cudaPitchedPtr srcz, cudaPitchedPtr dst, 
	const Volume::VolumeSize size) 
{
	const long inputRowPitch = srcx.pitch / sizeof(T);
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dst.pitch / sizeof(T);
	const long outputSlicePitch = size.y * outputRowPitch;

	const long blocksPerSliceY = size.y / modulusBlockSize.y;


	const long basex = (long)blockIdx.x * modulusBlockSize.x + (long)threadIdx.x;
	const long basey = ((long)blockIdx.y % blocksPerSliceY) * modulusBlockSize.y + (long)threadIdx.y;
	const long basez = (long)blockIdx.y / blocksPerSliceY;

	const long inputOffset = basez * inputSlicePitch + basey * inputRowPitch + basex;
	const long outputOffset = basez * outputSlicePitch + basey * outputRowPitch + basex;

	*(((T*)dst.ptr) + outputOffset) = 
		sqrt(square(*(((T*)srcx.ptr) + inputOffset))+square(*(((T*)srcy.ptr) + inputOffset))+square(*(((T*)srcz.ptr) + inputOffset)));

};
};};

#endif /* __MODULUS_KERNEL_H__ */