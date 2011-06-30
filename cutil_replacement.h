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

#ifndef __CUTIL_REPLACEMENT_H__
#define __CUTIL_REPLACEMENT_H__

#include "global.h"

#include <stdexcept>
#include <sstream>

class CUtilReplacementError : public std::runtime_error {
public:
	CUtilReplacementError(const std::string& sWhat, const std::string& sReason) : runtime_error(sWhat + ": " + sReason) {};
};

#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
// TODO: check if this is a emulator bug on linux, disable it for now...
#ifdef WIN32
#define cutilCheckMsg(msg)           __cutilCheckMsg     (msg, __FILE__, __LINE__)
#else
#define cutilCheckMsg(msg)					 {};
#endif

inline void __cudaSafeCall( cudaError err, const char *file, const int line ) {
    if( cudaSuccess != err)
		throw CUtilReplacementError("Runtime API error", cudaGetErrorString(err));
}

inline void __cutilCheckMsg( const char *errorMessage, const char *file, const int line ){
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) {
		std::stringstream out;
		out << file << "(" << line << ") : CUDA error: " << errorMessage << " ";
		throw CUtilReplacementError(out.str(), cudaGetErrorString(err));
	}
#ifdef _DEBUG
    err = cudaThreadSynchronize();
    if(cudaSuccess != err) {
		std::stringstream out;
		out << file << "(" << line << ") : CUDA cudaThreadSynchronize error: " << errorMessage << " ";
		throw CUtilReplacementError(out.str(), cudaGetErrorString(err));
	}
#endif
}

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

// This function returns the best GPU (with maximum GFLOPS)
inline int cutGetMaxGflopsDeviceId()
{
	int current_device   = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device  = 0;
	int device_count     = 0, best_SM_arch     = 0;
    int arch_cores_sm[3] = { 1, 8, 32 };
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
		    sm_per_multiproc = 1;
		} else if (deviceProp.major <= 2) {
			sm_per_multiproc = arch_cores_sm[deviceProp.major];
		} else {
			sm_per_multiproc = arch_cores_sm[2];
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if( compute_perf  > max_compute_perf ) {
            // If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 ) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch) {	
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			} else {
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

#endif /* __CUTIL_REPLACEMENT_H__ */
