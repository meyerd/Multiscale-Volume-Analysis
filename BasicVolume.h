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

#ifndef __BASIC_VOLUME_H__
#define __BASIC_VOLUME_H__

#include "global.h"
#include "string.h"

#include "GeneralVolume.h"

namespace Volume {

#ifndef WIN32
#	define fprintf_s fprintf
#endif

template<typename T>
class BasicVolume :
	public GeneralVolume<T> {
public:
	BasicVolume(std::string sFilename) {
		load(sFilename);
		this->m_bIsOk = true;
	};
	BasicVolume(long lSizeX, long lSizeY, long lSizeZ) {
			this->sSize = VolumeSize(lSizeX, lSizeY, lSizeZ);
			this->m_lLevels = 1;
			this->allocateDataBuffer();
	};
	BasicVolume(const VolumeSize& sSize) {
		this->sSize = sSize;
		this->m_lLevels = 1;
		this->allocateDataBuffer();
	};
	BasicVolume(void) {};
	virtual ~BasicVolume(void) {};

	T inline getAt(long lX, long lY, long lZ) {
		return this->m_pData[lZ * this->sSize.x * this->sSize.y + lY * this->sSize.x + lX];
	}

	void load(const std::string& sFilename) {
		OUT_INFO("BasicVolume: Reading from file \"%s\" ... ", sFilename.c_str());

		this->checkFile(sFilename);
		
		FileHandle fp(sFilename, "rb");
		fp.read(&this->m_sFileHeader, sizeof(VolumeHeader), 1);

		if(this->m_sFileHeader.eFileType == TYPE_BASIC_FLOAT) {
			if(strcmp(typeid(float).name(), typeid(T).name()) != 0) {
				throw VolumeFiletypeError("Wrong filetype", "TYPE_BASIC_FLOAT", typeid(T).name(), sFilename);
			}
		} else if(this->m_sFileHeader.eFileType == TYPE_BASIC_INT) {
			if(strcmp(typeid(int).name(), typeid(T).name()) != 0) {
				throw VolumeFiletypeError("Wrong filetype", "TYPE_BASIC_INT", typeid(T).name(), sFilename);
			}
		} else if(this->m_sFileHeader.eFileType == TYPE_BASIC_UNSIGNED_CHAR) {
			if(strcmp(typeid(unsigned char).name(), typeid(T).name()) != 0) {
				throw VolumeFiletypeError("Wrong filetype", "TYPE_BASIC_UNSIGNED_CHAR", typeid(T).name(), sFilename);
			}
		} else {
			throw VolumeFiletypeError("Wrong filetype", typeid(T).name(), "UNKNOWN", sFilename);
		}

		this->sSize = this->m_sFileHeader.sSize;
		this->m_lLevels = this->m_sFileHeader.lLevels;

		this->allocateDataBuffer();

		size_t sReadCount = this->sSize.x * this->sSize.y * this->sSize.z * this->m_lLevels;
		size_t sElemsRead = fp.read(this->m_pData, sizeof(T), sReadCount);

		OUT_INFO("done [%li bytes].\n", sElemsRead * sizeof(T));
	};

	void store(const std::string& sFilename, bool bOverwrite = true) {
		OUT_INFO("BasicVolume: Writing volume to file \"%s\" ... ", sFilename.c_str());
		if(!this->m_bIsOk)
			throw VolumeError("Can't save volume, that hasn't m_bIsOk set.");

		if(!this->m_pData)
			throw VolumeError("Volume has no data set.");

		if(this->sSize.x < 1 || this->sSize.y < 1 || this->sSize.z < 1 || this->m_lLevels < 1)
			throw VolumeError("Volume size not initialized.");

		if(strcmp(typeid(float).name(), typeid(T).name()) == 0) {
			this->m_sFileHeader.eFileType = TYPE_BASIC_FLOAT;
		} else if(strcmp(typeid(int).name(), typeid(T).name()) == 0) {
			this->m_sFileHeader.eFileType = TYPE_BASIC_INT;
		} else if(strcmp(typeid(unsigned char).name(), typeid(T).name()) == 0) {
			this->m_sFileHeader.eFileType = TYPE_BASIC_UNSIGNED_CHAR;
		} else {
			throw VolumeError("Volume has unsupported type.");
		}

		this->m_sFileHeader.sSize = this->sSize;
		this->m_sFileHeader.lLevels = this->m_lLevels;

		this->writeHeader(sFilename, bOverwrite);

		FileHandle fp(sFilename, "ab");

		size_t sWriteSize = this->sSize.x * this->sSize.y * this->sSize.z * this->m_lLevels;
		size_t sSizeWritten = 0;
		sSizeWritten = fp.write(this->m_pData, sizeof(T), sWriteSize);
		
		OUT_INFO("done [%li bytes].\n", sSizeWritten * sizeof(T));
	};

	virtual void storeRawDat(const std::string& sFilename, long lLevel = 0, bool bOverwrite = true) {
		std::string sDatFilename = (sFilename+".dat");
		// strip the path from the filename
		std::string sRawBasename("");
#ifdef WIN32
		std::string::size_type pos = sFilename.find_last_of("\\");
#else
		std::string::size_type pos = sFilename.find_last_of("\\");
#endif
		if(pos != std::string::npos) {
			sRawBasename = sFilename.substr(pos);
		} else {
			sRawBasename = sFilename;
		}
		std::string sRawFilename = (sFilename+".raw");
		std::string sRawFilenameDatFile = (sRawBasename+".raw");
		OUT_INFO("BasicVolume: Writing volume raw/dat to file \"%s\"/\"%s\" ... ", sRawFilename.c_str(), sDatFilename.c_str());

		if(!this->m_bIsOk)
			throw VolumeError("Can't save volume, that hasn't m_bIsOk set.");

		if(!this->m_pData)
			throw VolumeError("Volume has no data set.");

		if(this->sSize.x < 1 || this->sSize.y < 1 || this->sSize.z < 1 || this->m_lLevels < 1)
			throw VolumeError("Volume size not initialized.");

		if(lLevel > this->m_lLevels - 1)
			throw VolumeError("Volume does not have this many levels.");

		if(strcmp(typeid(float).name(), typeid(T).name()) == 0) {
			this->m_sFileHeader.eFileType = TYPE_BASIC_FLOAT;
		} else if(strcmp(typeid(int).name(), typeid(T).name()) == 0) {
			throw VolumeError(".dat-files only support float and uchar.");
		} else if(strcmp(typeid(unsigned char).name(), typeid(T).name()) == 0) {
			this->m_sFileHeader.eFileType = TYPE_BASIC_UNSIGNED_CHAR;
		} else {
			throw VolumeError("Volume has unsupported type.");
		}

		this->m_sFileHeader.sSize = this->sSize;
		this->m_sFileHeader.lLevels = this->m_lLevels;

		if(!bOverwrite) {
			Global::checkIfFileExists(sDatFilename);
			Global::checkIfFileExists(sRawFilename);
		}

		FileHandle fp(sDatFilename, "w");

		fprintf_s(fp, "ObjectFileName:\t%s\n", sRawFilenameDatFile.c_str());
		fprintf_s(fp, "Resolution:\t%li %li %li\n", this->sSize.x, this->sSize.y, this->sSize.z);
		fprintf_s(fp, "SliceThickness:\t1.0 1.0 1.0\n");
		fprintf_s(fp, "Format:\t\t%s\n", (this->m_sFileHeader.eFileType == TYPE_BASIC_FLOAT?"FLOAT":
										   (this->m_sFileHeader.eFileType == TYPE_BASIC_UNSIGNED_CHAR?"UCHAR":"FAIL!")));

		FileHandle fpr(sRawFilename, "wb");
		size_t sWriteSize = this->sSize.x * this->sSize.y * this->sSize.z * this->m_lLevels;
		size_t sSizeWritten = 0;
		sSizeWritten = fpr.write(this->m_pData, sizeof(T), sWriteSize);
		
		OUT_INFO("done [%li bytes].\n", sSizeWritten * sizeof(T));
	};

	void writeSliceToFile(long lSlice, const std::string& sFilename, long lLevel = 0) {
		if(lLevel >= this->m_lLevels)
			throw VolumeError("Level greater than available levels.");
		if(lSlice >= this->sSize.z)
			throw VolumeError("Slice greater than volume size z.");
		FileHandle fp(sFilename, "wb");

		size_t sWriteSize = this->sSize.x * this->sSize.y;
		size_t sDataOffset = lSlice * sWriteSize + lLevel * this->sSize.z * sWriteSize;
		fp.write(this->m_pData + sDataOffset, sizeof(T), sWriteSize);
	}

	void writeAllSlicesToFile(long lSlice, const std::string& sFilemask, long lLevel = 0) {
		char fname[4096];
		for(long z = 0; z < this->sSize.z; ++z) {
#ifdef WIN32
			sprintf_s<4096>(fname, sFilemask.c_str(), z);
#else
			sprintf(fname, sFilemask.c_str(), z);
#endif
			writeSliceToFile(z, fname, lLevel);
		}
	}
};
};

#endif /* __BASIC_VOLUME_H__ */


