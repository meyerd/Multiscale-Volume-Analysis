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


/* Prepend a volume header to an existing .raw file ... helper binary */

#include "../global.h"

#include "../FileHandle.h"

#include "../GeneralVolume.h"
#include "../BasicVolume.h"

#define BUFFER_SIZE 4*1024*1024

int main(int argc, char* argv[]) {
	if(argc < 7) {
		OUT_ERR("usage: volume_header_prepend <xsize> <ysize> <zsize> <datatype> <infile> <outfile>\n");
		return 1;
	}

	Volume::VolumeFileType eVolumeType = Volume::TYPE_UNKNOWN;

	long xsize = atol(argv[1]);
	long ysize = atol(argv[2]);
	long zsize = atol(argv[3]);

	char* typestr = argv[4];
	if(strcmp(typestr, "basic_float") == 0) {
		eVolumeType = Volume::TYPE_BASIC_FLOAT;
	} else if(strcmp(typestr, "basic_int") == 0) {
		eVolumeType = Volume::TYPE_BASIC_INT;
	} else if(strcmp(typestr, "basic_unsigned_char") == 0) {
		eVolumeType = Volume::TYPE_BASIC_UNSIGNED_CHAR;
	} else {
		OUT_ERR("\"%s\" is no valid volume type\n", typestr);
		return 3;
	}

	OUT_INFO("Volume information: %lix%lix%li type: %s\n", xsize, ysize, zsize, typestr);

	if(xsize < 1 || ysize < 1 || zsize < 1) {
		OUT_ERR("one of the sizes is wrong\n");
		return 4;
	}

	FILE* fpin = NULL;
#ifdef WIN32
	fopen_s(&fpin, argv[5], "rb");
#else
	fpin = fopen(argv[5], "rb");
#endif
	if(!fpin) {
		OUT_ERR("could not open \"%s\" for input\n", argv[5]);
		return 2;
	}

	Volume::VolumeHeader sHeader;
	sHeader.eFileType = eVolumeType;
	sHeader.lLevels = 1;
	sHeader.sSize.x = xsize;
	sHeader.sSize.y = ysize;
	sHeader.sSize.z = zsize;

	FILE* fpout = NULL;
#ifdef WIN32
	fopen_s(&fpout, argv[6], "rb");
#else
	fpout = fopen(argv[6], "rb");
#endif
	if(fpout != NULL) {
		OUT_WARN("\"%s\" already exists, exiting.\n", argv[6]);
		fclose(fpout);
		return 5;
	}
#ifdef WIN32
	fopen_s(&fpout, argv[6], "wb");
#else
	fpout = fopen(argv[6], "wb");
#endif
	if(!fpout) {
		OUT_ERR("could not open \"%s\" for writing.\n", argv[6]);
		return 6;
	}

	OUT_INFO("Writing header %li bytes ...\n", sizeof(Volume::VolumeHeader));

	if(fwrite(&sHeader, sizeof(Volume::VolumeHeader), 1, fpout) != 1) {
		OUT_ERR("could not write %li bytes header to \"%s\".\n", sizeof(Volume::VolumeHeader), argv[6]);
		fclose(fpout);
		return 7;
	}

	OUT_INFO("Writing data ...\n");
	unsigned char *buffer = new(std::nothrow) unsigned char[BUFFER_SIZE]; // 4MB buffer

	size_t sBytesRead = 0;
	size_t sBytesWritten = 0;
	do {
		sBytesRead = fread(buffer, sizeof(unsigned char), BUFFER_SIZE, fpin);
		if(ferror(fpin)) {
			OUT_ERR("error reading input file\n");
			fclose(fpin);
			fclose(fpout);
			SAFE_DELETE(buffer);
			return 8;
		}
		sBytesWritten = fwrite(buffer, sizeof(unsigned char), sBytesRead, fpout);
		if(sBytesWritten != sBytesRead) {
			OUT_ERR("error writing output file\n");
			fclose(fpin);
			fclose(fpout);
			SAFE_DELETE(buffer);
			return 9;
		}
	} while(!feof(fpin));
	SAFE_DELETE(buffer);

	OUT_INFO("done.\n");
	
	return 0;
}

