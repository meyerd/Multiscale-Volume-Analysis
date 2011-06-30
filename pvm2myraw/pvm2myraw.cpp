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

#include "codebase.h"
#include "ddsbase.h"

#include "../global.h"

#include <stdlib.h>
#include <time.h>

#include <stdexcept>
#include <string>

#include "../BasicVolume.h"

#define TYPE float

int main(int argc, char* argv[]) {
	if(argc < 3) {
		fprintf(stderr, "usage: pvm2myraw <input.pvm> <output.raw>\n");
		return 1;
	}
	
	printf("reading \"%s\" ...\n", argv[1]);
	unsigned int width = 0;
	unsigned int height = 0;
	unsigned int depth = 0;
	unsigned int components;
	float scalex,scaley,scalez;

	unsigned char* cVol = readPVMvolume(argv[1], &width, &height, &depth, &components, &scalex, &scaley, &scalez);
	if(cVol == NULL) {
		printf("error\n");
		return 1;
	}
	
	printf("found volume with width=%d height=%d depth=%d components=%d\n",
          width,height,depth,components);

	if (scalex!=1.0f || scaley!=1.0f || scalez!=1.0f)
		printf("  and edge length %g/%g/%g\n",scalex,scaley,scalez);

	long volSizeX = width;
	long volSizeY = height;
	long volSizeZ = depth;

	TYPE* pData = new TYPE[volSizeX * volSizeY * volSizeZ];

	printf("converting ...");
		
	for(long z = 0; z < volSizeZ; ++z) {
		for(long y = 0; y < volSizeY; ++y) {
			for(long x = 0; x < volSizeX; ++x) {
				pData[z * volSizeX * volSizeY + y * volSizeX + x] = (TYPE)(cVol[z * volSizeX * volSizeY + y * volSizeX + x]);
			}
		}
	}
	printf("done\n");

	SAFE_DELETE_ARRAY(cVol);

	try {
		Volume::BasicVolume<TYPE> vol(volSizeX, volSizeY, volSizeZ);
		vol.m_pData = pData;
		vol.m_lLevels = 1;
		vol.m_bIsOk = true;

		vol.store(argv[2]);
		//vol.writeSliceToFile(64, "testvol_midout.raw", 0);
		std::stringstream sst;
		sst << argv[2] << "_vis";
		vol.storeRawDat(sst.str());
	} catch(std::exception& e) {
		OUT_ERR("Exception: %s: %s\n", typeid(e).name(), e.what());
	}

	return 0;
}

