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

#include "../global.h"

#include <stdlib.h>
#include <time.h>

#include <stdexcept>
#include <string>
#include <cmath>

#include "../BasicVolume.h"

#define TYPE float

int main(int argc, char* argv[]) {
	if(argc < 6) {
		fprintf(stderr, "usage: raw_resample <input.raw> <x> <slicex> <y> <slicey> <z> <slicez> <output.raw>\n");
		return 1;
	}

	long targetSizeX = atol(argv[2]);
	float sliceSizeX = atof(argv[3]);
	long targetSizeY = atol(argv[4]);
	float sliceSizeY = atof(argv[5]);
	long targetSizeZ = atol(argv[6]);
	float sliceSizeZ = atof(argv[7]);

	Volume::BasicVolume<TYPE> in(argv[1]);
	
	TYPE* pData = new TYPE[targetSizeX * targetSizeY * targetSizeZ];

	printf("loaded %lix%lix%li.\n", in.sSize.x, in.sSize.y, in.sSize.z);
	printf("target %lix%lix%li.\n", targetSizeX, targetSizeY, targetSizeZ);
	printf("sliceThickness %.4fx%.4fx%.4f.\n", sliceSizeX, sliceSizeY, sliceSizeZ);

	printf("resampling...\n");
	float xfactor = (float)in.sSize.x / ((float)targetSizeX * sliceSizeX);
	float yfactor = (float)in.sSize.y / ((float)targetSizeY * sliceSizeY);
	float zfactor = (float)in.sSize.z / ((float)targetSizeZ * sliceSizeZ);
#pragma omp parallel for
	for(long z = 0; z < targetSizeZ; ++z) {
		for(long y = 0; y < targetSizeY; ++y) {
			for(long x = 0; x < targetSizeX; ++x) {
				long xl = max(0, min((long)floor(x * xfactor), in.sSize.x - 1));
				long xr = max(0, min((long)ceil(x * xfactor), in.sSize.x - 1));
				long yl = max(0, min((long)floor(y * yfactor), in.sSize.y - 1));
				long yr = max(0, min((long)ceil(y * yfactor), in.sSize.y - 1));
				long zl = max(0, min((long)floor(z * zfactor), in.sSize.z - 1));
				long zr = max(0, min((long)ceil(z * zfactor), in.sSize.z - 1));

				float x1 = (in.m_pData[zl * in.sSize.x * in.sSize.y +  yl * in.sSize.x + xl] + 
					in.m_pData[zl * in.sSize.x * in.sSize.y + yl * in.sSize.x + xr]) / 2.0f;
				float x2 = (in.m_pData[zl * in.sSize.x * in.sSize.y +  yr * in.sSize.x + xl] + 
					in.m_pData[zl * in.sSize.x * in.sSize.y + yr * in.sSize.x + xr]) / 2.0f;;
				float x3 = (in.m_pData[zr * in.sSize.x * in.sSize.y +  yl * in.sSize.x + xl] + 
					in.m_pData[zr * in.sSize.x * in.sSize.y + yl * in.sSize.x + xr]) / 2.0f;
				float x4 = (in.m_pData[zr * in.sSize.x * in.sSize.y +  yr * in.sSize.x + xl] + 
					in.m_pData[zr * in.sSize.x * in.sSize.y + yr * in.sSize.x + xr]) / 2.0f;

				float y1 = (x1 + x2) / 2.0f;
				float y2 = (x3 + x4) / 2.0f;

				float fin = (y1 + y2) / 2.0f;

				pData[z * targetSizeX * targetSizeY + y * targetSizeX + x] = fin;
			}
		}
	}

	try {
		Volume::BasicVolume<TYPE> vol(targetSizeX, targetSizeX, targetSizeX);
		vol.m_pData = pData;
		vol.m_lLevels = 1;
		vol.m_bIsOk = true;

		vol.store(argv[8]);
		//vol.writeSliceToFile(64, "testvol_midout.raw", 0);
		std::stringstream sst;
		sst << argv[8] << "_vis";
		vol.storeRawDat(sst.str());
	} catch(std::exception& e) {
		OUT_ERR("Exception: %s: %s\n", typeid(e).name(), e.what());
	}

	return 0;
}

