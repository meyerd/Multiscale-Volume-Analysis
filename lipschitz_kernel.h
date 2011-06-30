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

#ifndef __LIPSCHITZ_KERNEL_H__
#define __LIPSCHITZ_KERNEL_H__

#include "global.h"

#include <math_constants.h>
#include <cuda.h>

#include <float.h>

//#define MAX(x, y) ((x) < (y) ? (y) : (x))
//#define MIN(x, y) ((x) < (y) ? (x) : (y))
#include "cutil_replacement.h"
#include "kernel_vector_helper.h"

#include "maxima_kernel.h"

#include "GeneralVolume.h"



namespace Cuda { namespace Lipschitz {

/* TODO: this does not work on linux, especially the allocation of the __shared__ mem ... maybe problem with nvcc compiler */
const Volume::ConstVolumeSize<16, 16, 1> lipschitzBlockSize;
// TODO: get rid of these defines (only to fix linux compile error)
#define lipschitzBlockSizex		16
#define lipschitzBlockSizey		16
#define lipschitzBlockSizez		1

__constant__ __device__ const float anglesTolerance = DEFAULT_ANGLE_TOLERANCE;

template<typename T>
__device__ __inline__ T getAt(cudaPitchedPtr* srcData, size_t srcDataPtrPitch, long lLevel, long x, long y, long z, long rowPitch, long slicePitch) {
	return *(((T*)((srcData + (srcDataPtrPitch * lLevel))->ptr)) + (z * slicePitch + y * rowPitch + x));
}

template<typename T>
__device__ __inline__ void trace_levels(cudaPitchedPtr* moduli, cudaPitchedPtr* anglesxz, cudaPitchedPtr* anglesy,
	size_t moduliPitch, T* m, long level, long levels, long *max_trace,	long xIndex, long yIndex, long zIndex, 
	const Volume::VolumeSize size, long rowPitch, long slicePitch) {
	// trace with no search
	/*m[level] = getAt<T>(moduli, moduliPitch, level, xIndex, yIndex, zIndex, rowPitch, slicePitch);

	for(long l = level+1; l < levels; l++) {
		float tmp = getAt<T>(moduli, moduliPitch, l, xIndex, yIndex, zIndex, rowPitch, slicePitch);
		if(tmp != 0.0f) {
			m[l] = tmp;
		} else {
			*max_trace = l-1;
			return;
		}
	}
	*max_trace = level-1;
	return;*/

	// trace maximas

	// http://www.ece.cmu.edu/~ee899/project/deepak_mid.htm

	long xcenter = xIndex;
	long ycenter = yIndex;
	long zcenter = zIndex;

	m[level] = getAt<T>(moduli, moduliPitch, level, xcenter, ycenter, zcenter, rowPitch, slicePitch);
	T axz_ref = getAt<T>(anglesxz, moduliPitch, level, xcenter, ycenter, zcenter, rowPitch, slicePitch);
	T ay_ref = getAt<T>(anglesy, moduliPitch, level, xcenter, ycenter, zcenter, rowPitch, slicePitch);
	float3 dir_ref = vec_from_angles(axz_ref, ay_ref);

	for(long l = level+1; l < levels; l++) {
		T prev_level_val = m[l-1];
		long ysmallest = ycenter;
		long xsmallest = xcenter;
		long zsmallest = zcenter;
		T smallest_diff = FLT_MAX;
		T smallest_found = 0.0f;
		// respect the cone of influence of wavelet transform
		// original kernel radius + original kernel radius * inserted zeroes
		//int radius_in_level = 2 + 2 * ((2<<l) - 1);
		// fixed search radius
		int radius_in_level = 3;
			
		// look at center
		T tmp = getAt<T>(moduli, moduliPitch, l, xcenter, ycenter, zcenter, rowPitch, slicePitch);
#if TRACE_ONLY_ASCENDING_MAXIMA == 1 || TRACE_ONLY_DESCENDING_MAXIMA == 1
		T last_val = tmp;
#endif
		T diff = abs(prev_level_val - tmp);
		smallest_diff = diff;

		smallest_found = tmp;

		// just search, no preference
		long xf = MAX(xcenter - radius_in_level, 0);
		long xt = MIN(xcenter + radius_in_level, size.x-1);
		long yf = MAX(ycenter - radius_in_level, 0);
		long yt = MIN(ycenter + radius_in_level, size.y-1);
		long zf = MAX(zcenter - radius_in_level, 0);
		long zt = MIN(zcenter + radius_in_level, size.z-1);

		for(long z = zf; z < zt; z++) {
			for(long y = yf; y < yt; y++) {
				for(long x = xf; x < xt; x++) {
					tmp = getAt<T>(moduli, moduliPitch, l, x, y, z, rowPitch, slicePitch);
					diff = abs(prev_level_val - tmp);
					T axz = getAt<T>(anglesxz, moduliPitch, level, xcenter, ycenter, zcenter, rowPitch, slicePitch);
					T ay = getAt<T>(anglesy, moduliPitch, level, xcenter, ycenter, zcenter, rowPitch, slicePitch);
					if(dir_ref * vec_from_angles(axz, ay) >= anglesTolerance) {
#if TRACE_ONLY_ASCENDING_MAXIMA == 1
						if(diff < smallest_diff && tmp >= last_val) {
#elif TRACE_ONLY_DESCENDING_MAXIMA == 1
						if(diff < smallest_diff && tmp <= last_val) {
#else
						if(diff < smallest_diff) {
#endif
							smallest_found = tmp;
							smallest_diff = diff;
							xsmallest = x;
							ysmallest = y;
							zsmallest = z;
						}
					}
#if TRACE_ONLY_ASCENDING_MAXIMA == 1 || TRACE_ONLY_DESCENDING_MAXIMA == 1
					last_val = tmp;
#endif
				}
			}
		}


		m[l] = smallest_found;
		if(smallest_found == 0.0f) {
			// we lost the modulus maxima here ... :(
			*max_trace = l;
			//*max_trace = radius_in_level;
			return;
		}
		xcenter = xsmallest;
		ycenter = ysmallest;
		zcenter = zsmallest;
	}

	*max_trace = levels;
	return;
}

__device__ const long max_iterations = GRADIENT_DESCENT_MAX_ITERATIONS;

__device__ const float lambda = GRADIENT_DESCENT_LAMBDA_STEPSIZE;
__device__ const float ftol = GRADIENT_DESCENT_TOLERANCE;
__device__ const float epsilon = 1e-10f;

__device__ const float ln2 = 0.69314718055994530941723212145818f; // log(2.0f);

__device__ float3 cost_function_simple_deriv_manual(const float3& p, float* m, long levels) {
	float K = p.x;
	float alpha = p.y;

	float firstpartsum = 0.0f;
	float alphasum = 0.0f;
	for(int j = 0; j < levels; ++j) {
		float aj = m[j];
		float twoexp = exp2f((float)(2*(j+1)));
		float firstpart = 2.0f * (log2(abs(aj)) - log2(p.x) - p.y * (float)(j+1));
		firstpartsum += firstpart;
		alphasum += firstpart * -1.0f * (float)(j+1);
	}
	float dK = firstpartsum * (-1.0f / (K * ln2));
	float dalpha = alphasum;

	return make_float3(dK, dalpha, 0.0f);
};

__device__ float cost_function_simple(const float3& p, float* m, long levels) {
	float sum = 0.0f;
	for(int j = 0; j < levels; ++j) {
		float aj = m[j];
		sum += square(log2(abs(aj)) - log2(p.x) - p.y * (float)(j+1));
	}
	return sum;	
};

#define cost_function cost_function_simple
#define gradient cost_function_simple_deriv_manual

__device__ float3 conjugate_gradient_descent(float* m, const float3& start, long level, long max_trace) {
	float3 p = start;
	float3 oldp = p;
	float x = cost_function(p, m, max_trace);
	float oldx = x;
	float3 xi = gradient(p, m, max_trace);

	// initialization
	float3 g, h;
	g = -xi;
	xi = h = g;

	long i = 0;
	do {
		// save old value
		oldx = x;

		// one step in conjugate gradient direction
		p = p + lambda * xi;

		// cost & gradient
		x = cost_function(p, m, max_trace);
		xi = gradient(p, m, max_trace);

		// calculate conjugate gradient direction
		float gg = g*g;
		float dgg = (xi + g) * xi;
		if(gg == 0.0f)
			break;

		g = -xi;
		xi = h = g + (dgg/gg) * h;

		i++;
	} while((i < max_iterations) && (2 * abs(x-oldx) > ftol * (abs(x) + abs(oldx) + epsilon)));

	p.z = (float)i;

	return p;
};

__device__ __inline__ cudaPitchedPtr* getLevelPtr(cudaPitchedPtr* srcData, size_t srcDataPtrPitch, long lLevel) {
	return (srcData + (srcDataPtrPitch * lLevel));
};

__device__ float3 simple_linear(float* m, long level, long max_trace) {
	float3 p = make_float3(0.0f, 0.0f, 0.0f);
	for(int j = 1; j < max_trace; ++j) {
		p.y += (log2(abs(m[j])) - log2(abs(m[j-1]))) / 1.0f;  // alpha
	}
	p.y = p.y / max_trace; 
	return p;
};

template<typename T, int max_levels>
__global__ void calculateLipschitz(cudaPitchedPtr* moduli, cudaPitchedPtr* anglesxz, cudaPitchedPtr* anglesy,
	size_t srcDataPtrPitch, cudaPitchedPtr dst, const Volume::VolumeSize size, long level, long levels) 
{

	const long inputRowPitch = getLevelPtr(moduli, srcDataPtrPitch, level)->pitch / sizeof(T);
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dst.pitch / sizeof(T);
	const long outputSlicePitch = size.y * outputRowPitch;

	const long blocksPerSliceY = size.y / lipschitzBlockSize.y;

	const long basex = (long)blockIdx.x * lipschitzBlockSize.x + (long)threadIdx.x;
	const long basey = ((long)blockIdx.y % blocksPerSliceY) * lipschitzBlockSize.y + (long)threadIdx.y;
	const long basez = (long)blockIdx.y / blocksPerSliceY;


	T m[max_levels] = {0};

	float mod = getAt<T>(moduli, srcDataPtrPitch, level, basex, basey, basez, inputRowPitch, inputSlicePitch);

	if(mod != 0.0f) {
		long max_trace = level;
		trace_levels<T>(moduli, anglesxz, anglesy, srcDataPtrPitch, m, level, levels, &max_trace, 
			basex, basey, basez, size, inputRowPitch, inputSlicePitch);

		float3 p = make_float3(1.0f, 0.0f, 0.0f);

#if ONLY_TRACING == 0
#if LIPSCHITZ_METHOD == 2
		// linear method
		p = simple_linear(m, level, max_trace);
#else
		// conjugate gradient 
		p = conjugate_gradient_descent(m, p, level, max_trace);
#endif
		
		*(((T*)dst.ptr) + (basez * outputSlicePitch + basey * outputRowPitch + basex)) = p.y;
#else
		*(((T*)dst.ptr) + (basez * outputSlicePitch + basey * outputRowPitch + basex)) = (float)max_trace;
#endif
	} else {
		*(((T*)dst.ptr) + (basez * outputSlicePitch + basey * outputRowPitch + basex)) = -10.0f;
	}
};

};};

#endif /* __LIPSCHITZ_KERNEL_H__ */