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

#include <omp.h>
#include <float.h>
#include <math.h>

#include "cpu_vector_helper.h"

#include "timing.h"

using namespace std;
using namespace Volume;

// this is a CPU implementation of the lipschitz estimation for debugging purposes

namespace CPU { namespace Lipschitz {

#define MAX max
#define MIN min

template<typename T>
inline T getAt(T* moduli, size_t moduliLevelPitch, long lLevel, long x, long y, long z, long rowPitch, long slicePitch) {
	return *((T*)(moduli) + (moduliLevelPitch * lLevel) + (z * slicePitch + y * rowPitch + x));
};

template<typename T>
void trace_levels(T* moduli, size_t moduliLevelPitch, T* m, long level, long levels, long *max_trace,
	long xIndex, long yIndex, long zIndex, const Volume::VolumeSize size, long rowPitch, long slicePitch)
{
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

	m[level] = getAt<T>(moduli, moduliLevelPitch, level, xcenter, ycenter, zcenter, rowPitch, slicePitch);

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
		int radius_in_level = 3;
			
		// look at center
		T tmp = getAt<T>(moduli, moduliLevelPitch, l, xcenter, ycenter, zcenter, rowPitch, slicePitch);
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
					tmp = getAt<T>(moduli, moduliLevelPitch, l, x, y, z, rowPitch, slicePitch);
					diff = abs(prev_level_val - tmp);
					if(diff < smallest_diff) {
						smallest_found = tmp;
						smallest_diff = diff;
						xsmallest = x;
						ysmallest = y;
						zsmallest = z;
					}
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
};

const long max_iterations = 3000;

const float lambda = 1e-2f;
const float ftol = 1e-8f;
const float epsilon = 1e-10f;

const float ln2 = 0.69314718055994530941723212145818f; // log(2.0f);

float3 cost_function_simple_deriv_manual(const float3& p, float* m, long levels) {
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

float cost_function_simple(const float3& p, float* m, long levels) {
	float sum = 0.0f;
	for(int j = 0; j < levels; ++j) {
		float aj = m[j];
		sum += square(log2(abs(aj)) - log2(p.x) - p.y * (float)(j+1));
	}
	return sum;	
};

#define cost_function cost_function_simple
#define gradient cost_function_simple_deriv_manual

float3 conjugate_gradient_descent(float* m, const float3& start, long level, long max_trace) {
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

template<typename T, int max_levels>
void lipschitzKernel(long basex, long basey, long basez, T* moduli, long moduliLevelPitch, long moduliPitch, 
	T* dst, long dstPitch, const Volume::VolumeSize size, long level, long levels) 
{
	const long inputRowPitch = moduliPitch;
	const long inputSlicePitch = size.y * inputRowPitch;
	const long outputRowPitch = dstPitch;
	const long outputSlicePitch = size.y * outputRowPitch;

	T m[max_levels] = {0};

	float mod = getAt<T>(moduli, moduliLevelPitch, level, basex, basey, basez, inputRowPitch, inputSlicePitch);
	if(mod != 0.0f) {
		long max_trace = level;
		trace_levels<T>(moduli, moduliLevelPitch, m, level, levels, &max_trace, basex, basey, basez, size, inputRowPitch, inputSlicePitch);

		float3 p = make_float3(1.0f, 0.0f, 0.0f);

		/* OUT_INFO("%lix%lix%li, mt: %li [", basex, basey, basez, max_trace);
		for(int mi = level; mi < max_trace; mi++) {
			OUT_INFO("%.3f,", m[mi]);
		}
		OUT_INFO("]\n"); */

		// linear method
		//p = simple_linear(m, level, max_trace);
		
		// conjugate gradient 
		p = conjugate_gradient_descent(m, p, level, max_trace-1);

		//*(((T*)dst) + (basez * outputSlicePitch + basey * outputRowPitch + basex)) = (float)max_trace;
		*(((T*)dst) + (basez * outputSlicePitch + basey * outputRowPitch + basex)) = p.y;
		//*(((T*)dst) + (basez * outputSlicePitch + basey * outputRowPitch + basex)) = p.z;
		//*(((T*)dst) + (basez * outputSlicePitch + basey * outputRowPitch + basex)) = mod;
	} else {
		*(((T*)dst) + (basez * outputSlicePitch + basey * outputRowPitch + basex)) = 0.0f;
	}
}

/**
  Calculates the lipschitz/hölder exponent for each voxel.

  @param input The angles in the xz plane and the angle in y direction, the modulus maxima
 */
Volume::MultilevelVolumeContainer<float, 3>* calculateLipschitz(Volume::MultilevelVolumeContainer<float, 3>& input) {
	if(input.x.sSize != input.y.sSize || input.y.sSize != input.z.sSize)
		throw CpuLipschitzError("Input erroneous input sizes.");
	if(input.x.m_lLevels != input.y.m_lLevels || input.y.m_lLevels != input.z.m_lLevels)
		throw CpuLipschitzError("Input erroneous levels.");
	VolumeSize sSize = input.x.sSize;
	long lLevels = input.x.m_lLevels;
	MultilevelVolumeContainer<float, 3>* output = new MultilevelVolumeContainer<float, 3>(sSize, lLevels);

	//for(long l = 0; l < lLevels; ++l) {
	//	for(long z = 0; z < sSize.z; ++z) {
	//		OUT_INFO("----------------------------------------\n");
	//		OUT_INFO("level: %li, slicez: %li\n", l, z);
	//		input.z.dumpLevelXYSlice(l, z);
	//		getchar();
	//	}
	//}

	CTimer ctAllLevelsTime;
	CTimer ctOneLevelTime;

	ctAllLevelsTime.Reset();

	for(long l = 0; l < lLevels; ++l) {
	ctOneLevelTime.Reset();
	OUT_INFO("calculateLipschitzCPU: Lipschitz level %li ... ", l);
#pragma omp parallel for
	for(long z = 0; z < sSize.z; ++z) {
		for(long y = 0; y < sSize.y; ++y) {
			for(long x = 0; x < sSize.x; ++x) {
				lipschitzKernel<float, 5>(x, y, z, input.z.getData(), input.z.sSize.x * input.z.sSize.y * input.z.sSize.z,
					input.z.sSize.x, output->x.getLevel(l), output->x.sSize.x, sSize, l, lLevels);
			}
		}
	}
	OUT_INFO("done [%.5fs].\n", (float)ctOneLevelTime.Query());
	}
	DEBUG_OUT("calculateLipschitzCPU: Total time (incl. copy) %.5fs.\n", (float)ctAllLevelsTime.Query());

	output->x.m_bIsOk = true;
	return output;
};
};};