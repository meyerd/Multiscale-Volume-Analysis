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

#ifndef __CPU_VECTOR_HELPER_H__
#define __CPU_VECTOR_HELPER_H__

const float ln2 = log(2.0f);

inline float log2(const float& x) {
	return (logf(x)/ln2);
}

inline float square(const float& x) {
	return x * x;
}

inline float exp2f(const float& x) {
	return powf(2.0f, x);
}

typedef struct {
	float x;
	float y;
	float z;
} float3;

inline float3 make_float3(const float& x, const float& y, const float& z) {
	float3 r;
	r.x = x;
	r.y = y; 
	r.z = z;
	return r;
};

inline float operator* (const float3& a, const float3& b) {
	float r;
	r = a.x * b.x + a.y * b.y + a.z * b.z;
	return r;
};

inline float3 operator* (const float& a, const float3& b) {
	float3 r;
	r.x = a * b.x;
	r.y = a * b.y;
	r.z = a * b.z;
	return r;
};

inline float3 operator* (const float3& a, const float& b) {
	return b*a;
};

inline float3 operator+ (const float& a, const float3& b) {
	float3 r;
	r.x = a + b.x;
	r.y = a + b.y;
	r.z = a + b.z;
	return r;
};

inline float3 operator+ (const float3& a, const float& b) {
	return b+a;
};

inline float3 operator+ (const float3& a, const float3& b) {
	float3 r;
	r.x = a.x + b.x;
	r.y = a.y + b.y;
	r.z = a.z + b.z;
	return r;
};

inline float3 operator- (const float3& a, const float3& b) {
	float3 r;
	r.x = a.x - b.x;
	r.y = a.y - b.y;
	r.z = a.z - b.z;
	return r;
};

inline float3 operator- (const float3& a) {
	float3 r;
	r.x = -a.x;
	r.y = -a.y;
	r.z = -a.z;
	return r;
};

#endif /* __CPU_VECTOR_HELPER_H__ */
