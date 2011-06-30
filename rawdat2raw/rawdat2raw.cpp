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

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <string>
#include <cctype>
#include <omp.h>

#include "../BasicVolume.h"
#include "../FileHandle.h"

#ifndef WIN32
#define MAX_PATH 4096
#include <libgen.h>
#endif

using namespace std;

int main(int argc, char* argv[])
{
	if(argc < 3) {
		fprintf(stderr, "usage: rawdat2raw <input.dat> <output.raw>\n");
		return 1;
	}

	try {
		FileHandle dat(argv[1], "r");

		char drive[MAX_PATH];
		char path[MAX_PATH];
		char filename[MAX_PATH];
		char ext[MAX_PATH];
		
#ifdef WIN32
		_splitpath_s<MAX_PATH, MAX_PATH, MAX_PATH, MAX_PATH>(argv[1], drive, path, filename, ext);
#else
		drive[0] = '\0'; // not neccessary on linux
		strcpy(path, dirname(argv[1]));
		filename[0] = '\0'; // unused
#endif

		string rawname("");
		Volume::VolumeSize size(0,0,0);
		string format;

		while(!dat.eof()) {
			char tag[MAX_PATH];
#ifdef WIN32
			fscanf_s(dat," %[^:]:", tag, MAX_PATH);
#else
			fscanf(dat, " %[^:]:", tag);
#endif
			string sTag(tag);
			transform(sTag.begin(), sTag.end(), sTag.begin(), ::toupper);
			char str[MAX_PATH];
			if (sTag==string("OBJECTFILENAME")) {
#ifdef WIN32
				fscanf_s(dat," %[^\n^ ^\t] ", str, MAX_PATH);	// filename
#else
				fscanf(dat, " %[^\n^ ^\t] ", str);
#endif
				rawname = string(str);
			} else if (sTag==string("RESOLUTION")) {
				int i;
#ifdef WIN32
				fscanf_s(dat,"%i",&i);
#else
				fscanf(dat,"%i",&i);
#endif
				size.x = static_cast<long>(i);
#ifdef WIN32
				fscanf_s(dat,"%i",&i);
#else
				fscanf(dat,"%i",&i);
#endif
				size.y = static_cast<long>(i);
#ifdef WIN32
				fscanf_s(dat,"%i",&i);
#else
				fscanf(dat,"%i",&i);
#endif
				size.z = static_cast<long>(i);				
			} else if (sTag==string("FORMAT")) {
#ifdef WIN32
				fscanf_s(dat," %[^\n] ",str,MAX_PATH);
#else
				fscanf(dat, " %[^\n] ", str);
#endif
				string sFormat(str);
				transform(sFormat.begin(),sFormat.end(),sFormat.begin(),::toupper);
				format = sFormat;
#ifdef WIN32
				fscanf_s(dat," %[^\n] ",str,MAX_PATH);	// remove rest of line
#else
				fscanf(dat, " %[^\n] ", str);
#endif
			} else { // non-parsed information
#ifdef WIN32
				fscanf_s(dat," %[^\n] ",str,MAX_PATH);
#else
				fscanf(dat, " %[^\n] ", str);
#endif
			}	
		}

		OUT_INFO("Parsed dat:\n");
		OUT_INFO(" Raw Dat: \"%s\"\n", rawname.c_str());
		OUT_INFO(" Resolution: %lix%lix%li\n", size.x, size.y, size.z);
		OUT_INFO(" Format: %s\n", format.c_str());

		if(rawname == string("") || size.x == 0 || size.y == 0 || size.z == 0 || 
			(format != "FLOAT" && format != "UCHAR")) {
				throw runtime_error(string("Error parsing dat file"));
		}

		Volume::BasicVolume<float> vol(size);
		
		OUT_INFO("Reading \"%s\" ... ", (string(drive) + string(path) + rawname).c_str());
		FileHandle raw(string(drive) + string(path) + rawname, "rb");
		if(format == "FLOAT") {
			size_t sRead = raw.read(vol.m_pData, sizeof(float), size.x * size.y * size.z);
			if(sRead < (size_t)(size.x * size.y * size.z)) {
				throw runtime_error(string("Error reading raw file"));
			}
			OUT_INFO("done\n");
		} else if(format == "UCHAR") {
			unsigned char* buffer = new unsigned char[size.x * size.y * size.z];
			size_t sRead = raw.read(buffer, sizeof(unsigned char), size.x * size.y * size.z);
			if(sRead < (size_t)(size.x * size.y * size.z)) {
				throw runtime_error(string("Error reading raw file"));
			}
			OUT_INFO("done\n");
			OUT_INFO("Converting to float ... ");
#pragma omp parallel for
			for(long z = 0; z < size.z; ++z) {
				for(long y = 0; y < size.y; ++y) {
					for(long x = 0; x < size.x; ++x) {
						vol.m_pData[z * size.x * size.y + y * size.x + x] = (float)buffer[z * size.x * size.y + y * size.x + x];
					}
				}
			}
			OUT_INFO("done\n");
			SAFE_DELETE_ARRAY(buffer);
		}
		vol.m_bIsOk = true;
		vol.store(argv[2]);
	} catch(exception& e) {
		OUT_ERR("Exception: %s: %s\n", typeid(e).name(), e.what());
	}

	return 0;
}

