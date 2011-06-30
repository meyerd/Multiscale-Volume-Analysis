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

#ifndef __KERNEL_VECTOR_HELPER_H__
#define __KERNEL_VECTOR_HELPER_H__

#include <math.h>

//#define square(x) ((x) * (x))
__device__ __inline__ float square(const float& x) {
	return x * x;
}

__device__ __inline__ float operator* (const float3& a, const float3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __inline__ float3 operator* (const float& a, const float3& b) {
	float3 r;
	r.x = a * b.x;
	r.y = a * b.y;
	r.z = a * b.z;
	return r;
}

__device__ __inline__ float3 operator* (const float3& a, const float& b) {
	return b*a;
}

__device__ __inline__ float3 operator+ (const float& a, const float3& b) {
	float3 r;
	r.x = a + b.x;
	r.y = a + b.y;
	r.z = a + b.z;
	return r;
}

__device__ __inline__ float3 operator+ (const float3& a, const float& b) {
	return b+a;
}

__device__ __inline__ float3 operator+ (const float3& a, const float3& b) {
	float3 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;
	return r;
}

__device__ __inline__ float3 operator- (const float3& a, const float3& b) {
	float3 r;
	r.x = a.x - b.x;
	r.y = a.y - b.y;
	r.z = a.z - b.z;
	return r;
}

__device__ __inline__ float3 operator- (const float3& a) {
	float3 r;
	r.x = -a.x;
	r.y = -a.y;
	r.z = -a.z;
	return r;
}

__device__ __inline__ float3 vec_from_angles(const float& xz, const float& y) {
	float3 r;
	r.z = sin(xz);
	r.x = cos(xz);
	r.y = sin(y);
	float l = 1.0f / sqrt(r * r);
	r.z *= l;
	r.x *= l;
	r.y *= l;
	return r;
}

#endif /* __KERNEL_VECTOR_HELPER_H__ */
