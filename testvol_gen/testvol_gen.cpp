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

#include "../BasicVolume.h"

#define TYPE float

int main(int argc, char* argv[]) {
	long volSizeX = 128;
	long volSizeY = 128;
	long volSizeZ = 128;

	TYPE* pData = new TYPE[volSizeX * volSizeY * volSizeZ];
	
	srand((unsigned)time(NULL));

	for(long z = 0; z < volSizeZ; ++z) {
		for(long y = 0; y < volSizeY; ++y) {
			for(long x = 0; x < volSizeX; ++x) {
				/*if(x >= 110 && x <= 160 && y >= 110 && y <= 160 && z >= 110 && z <= 160) {
					pData[z * volSizeY * volSizeX + y * volSizeX + x] = 127.0f;
				} else {
					pData[z * volSizeY * volSizeX + y * volSizeX + x] = -128.0f;
				}*/
				if(x >= 50 && x <= 70 && y >= 50 && y <= 70 && z >= 50 && z <= 70) {
					pData[z * volSizeY * volSizeX + y * volSizeX + x] = 5.0f;
				} else {
					pData[z * volSizeY * volSizeX + y * volSizeX + x] = 0.0f;
				}
				/*if(x == 64) {
					pData[z * volSizeY * volSizeX + y * volSizeX + x] = 127.0f;
				} else {
					pData[z * volSizeY * volSizeX + y * volSizeX + x] = -128.0f;
				}*/
				/*if(x == 64 && y == 64 && z == 64) {
					pData[z * volSizeY * volSizeX + y * volSizeX + x] = 127.0f;
				} else {
					pData[z * volSizeY * volSizeX + y * volSizeX + x] = -128.0f;
				}*/
				//pData[z * volSizeY * volSizeX + y * volSizeX + x] = (float)rand() / (RAND_MAX + 1) * (100.0f - 0.0f) + 0.0f;
				//pData[z * volSizeY * volSizeX + y * volSizeX + x] = 100.0f;
			}
		}
	}

	try {
		Volume::BasicVolume<TYPE> vol(volSizeX, volSizeY, volSizeZ);
		vol.m_pData = pData;
		vol.m_lLevels = 1;
		vol.m_bIsOk = true;

		vol.store("vol_box_128.raw");
		//vol.writeSliceToFile(64, "testvol_midout.raw", 0);
		vol.storeRawDat("vol_box_128_vis");
	} catch(std::exception& e) {
		OUT_ERR("Exception: %s: %s\n", typeid(e).name(), e.what());
	}

	getchar();
	return 0;
}

