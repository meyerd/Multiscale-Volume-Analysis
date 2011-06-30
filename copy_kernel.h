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

#ifndef __COPY_KERNEL_H__
#define __COPY_KERNEL_H__

#include "global.h"

#include <cuda.h>

#include "GeneralVolume.h"

namespace Cuda { namespace DWT {

const Volume::ConstVolumeSize<16, 16, 1> copyBlockSize;

__global__ void copyGPU(cudaPitchedPtr src, cudaPitchedPtr dst, const Volume::VolumeSize size)
{
	const long xIndex = blockIdx.x * copyBlockSize.x + threadIdx.x;
	const long yIndex = blockIdx.y * copyBlockSize.y + threadIdx.y;
	const long zIndex = blockIdx.z * copyBlockSize.z + threadIdx.z;

	long long index = zIndex * size.x * size.y + yIndex * size.x + xIndex;

	((float*)dst.ptr)[index] = ((float*)src.ptr)[index];
};

};};

#endif /* __COPY_KERNEL_H__ */
