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

#ifndef __MULTILEVEL_VOLUME_H__
#define __MULTILEVEL_VOLUME_H__

#include "global.h"

#include <vector>

#include "BasicVolume.h"

namespace Volume {

template<typename T>
class MultilevelVolume;

typedef struct VolumeContainerHeader_t {
	VolumeContainerHeader_t() : eFileType(TYPE_UNKNOWN), lLevels(0), lLength(0) {};
	VolumeFileType eFileType;
	VolumeSize sSize;
	long lLevels;
	long lLength;
} VolumeContainerHeader;

template<typename T, int length>
struct MultilevelVolumeContainer {
public:
	MultilevelVolumeContainer(void) {
		for(int i = 0; i < length; ++i) {
			volumes.push_back(MultilevelVolume<T>());
		}
	};
	MultilevelVolumeContainer(long lSizeX, long lSizeY, long lSizeZ, long lLevels) {
		for(int i = 0; i < length; ++i) {
			volumes.push_back(MultilevelVolume<T>(lSizeX, lSizeY, lSizeZ, lLevels));
		}
	};
	MultilevelVolumeContainer(const VolumeSize& sSize, long lLevels) {
		for(int i = 0; i < length; ++i) {
			volumes.push_back(MultilevelVolume<T>(sSize, lLevels));
		}
	};

	std::vector< MultilevelVolume<T> > volumes;
};

template<typename T>
struct MultilevelVolumeContainer<T, 2> {
public:
	MultilevelVolumeContainer(void) : x(MultilevelVolume<T>()), y(MultilevelVolume<T>()) {};
	MultilevelVolumeContainer(long lSizeX, long lSizeY, long lSizeZ, long lLevels) :
		x(lSizeX, lSizeY, lSizeZ), y(lSizeX, lSizeY, lSizeZ) {};
	MultilevelVolumeContainer(const VolumeSize& sSize, long lLevels) : 
		x(sSize, lLevels), y(sSize, lLevels) {};

	MultilevelVolume<T> x;
	MultilevelVolume<T> y;
};

template<typename T>
struct MultilevelVolumeContainer<T, 3> {
public:
	MultilevelVolumeContainer(void) : x(), y(), z() {};
	MultilevelVolumeContainer(long lSizeX, long lSizeY, long lSizeZ, long lLevels) :
		x(lSizeX, lSizeY, lSizeZ), y(lSizeX, lSizeY, lSizeZ), z(lSizeX, lSizeY, lSizeZ) {};
	MultilevelVolumeContainer(const VolumeSize& sSize, long lLevels) : 
		x(sSize, lLevels), y(sSize, lLevels), z(sSize, lLevels) {};
	MultilevelVolumeContainer(const std::string& sFilename) {
		this->load(sFilename);
	};

	void writeHeader(const std::string& sFilename, bool bOverwrite = true) {
		if(!bOverwrite)
			Global::checkIfFileExists(sFilename);	
		FileHandle fp(sFilename, "wb");
		fp.write(&m_sFileHeader, sizeof(VolumeContainerHeader), 1);
	};

	void checkFile(const std::string& sFilename) {
		FileHandle fp(sFilename, "rb");

		__int64 i64Filesize = fp.getSize();

		if(i64Filesize < sizeof(VolumeContainerHeader))
			throw VolumeFilesizeError("Volumefile too small (doesn't even include the header)", sFilename);
	};
	
	void store(const std::string& sFilename, bool bOverwrite = true) {
		OUT_INFO("MultilevelVolumeContainer: Writing to file \"%s\" ...", sFilename.c_str());
		if(!this->x.m_bIsOk || !this->y.m_bIsOk || !this->z.m_bIsOk) 
			throw VolumeError("Can't save volume, that hasn't m_bIsOk set.");

		if(!this->x.m_pData || !this->y.m_pData || !this->z.m_pData)
			throw VolumeError("Volume has no data set.");
		if(this->x.sSize.x < 1 || this->x.sSize.y < 1 || this->x.sSize.z < 1 || this->x.m_lLevels < 1 ||
			 this->y.sSize.x < 1 || this->y.sSize.y < 1 || this->y.sSize.z < 1 || this->y.m_lLevels < 1 ||
			 this->z.sSize.x < 1 || this->z.sSize.y < 1 || this->z.sSize.z < 1 || this->z.m_lLevels < 1)
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

		this->m_sFileHeader.sSize = this->x.sSize;
		this->m_sFileHeader.lLevels = this->x.m_lLevels;
		this->m_sFileHeader.lLength = 3;

		this->writeHeader(sFilename, bOverwrite);

		FileHandle fp(sFilename, "ab");

		size_t sWriteSize = this->x.sSize.x * this->x.sSize.y * this->x.sSize.z * this->x.m_lLevels;
		size_t sSizeWritten = 0;
		sSizeWritten += fp.write(this->x.m_pData, sizeof(T), sWriteSize);
		sSizeWritten += fp.write(this->y.m_pData, sizeof(T), sWriteSize);
		sSizeWritten += fp.write(this->z.m_pData, sizeof(T), sWriteSize);

		OUT_INFO("done [%li bytes].\n", sSizeWritten * sizeof(T));
	};

	void load(const std::string& sFilename) {
		OUT_INFO("MultilevelVolumeContainer: Reading from file \"%s\" ... ", sFilename.c_str());

		this->checkFile(sFilename);

		FileHandle fp(sFilename, "rb");
		fp.read(&this->m_sFileHeader, sizeof(VolumeContainerHeader), 1);

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

		if(this->m_sFileHeader.lLength != 3)
			throw VolumeError("Wrong filetype, VolumeContainer not of length 3");

		this->x.sSize = this->m_sFileHeader.sSize;
		this->x.m_lLevels = this->m_sFileHeader.lLevels;
		this->y.sSize = this->m_sFileHeader.sSize;
		this->y.m_lLevels = this->m_sFileHeader.lLevels;
		this->z.sSize = this->m_sFileHeader.sSize;
		this->z.m_lLevels = this->m_sFileHeader.lLevels;

		this->x.allocateDataBuffer();
		this->y.allocateDataBuffer();
		this->z.allocateDataBuffer();
		
		size_t sReadCount = this->x.sSize.x * this->x.sSize.y * this->x.sSize.z * this->x.m_lLevels;
		size_t sElemsRead = fp.read(this->x.m_pData, sizeof(T), sReadCount);
		sElemsRead += fp.read(this->y.m_pData, sizeof(T), sReadCount);
		sElemsRead += fp.read(this->z.m_pData, sizeof(T), sReadCount);

		OUT_INFO("done [%li bytes].\n", sElemsRead * sizeof(T));
	};

	VolumeContainerHeader m_sFileHeader;

	MultilevelVolume<T> x;
	MultilevelVolume<T> y;
	MultilevelVolume<T> z;
};

template<typename T>
class MultilevelVolume :
	public BasicVolume<T> {
public:
	MultilevelVolume(long lSizeX, long lSizeY, long lSizeZ, long lLevels) {
			this->sSize = VolumeSize(lSizeX, lSizeY, lSizeZ);
			this->m_lLevels = lLevels;
			this->allocateDataBuffer();
	};
	MultilevelVolume(const VolumeSize& sSize, long lLevels) {
		this->sSize = sSize;
		this->m_lLevels = lLevels;
		this->allocateDataBuffer();
	};
	MultilevelVolume(const std::string& sFilename) {
		this->load(sFilename);
	};
	MultilevelVolume(void) {};
	~MultilevelVolume(void) {};

	T inline getAt(long lX, long lY, long lZ, long lLevel) {
		return this->m_pData[lLevel * this->sSize.x * this->sSize.y * this->sSize.z + 
			lZ * this->sSize.x * this->sSize.y + lY * this->sSize.x + lX];
	};

	T* getLevel(long lLevel) {
		if(lLevel >= this->m_lLevels)
			throw VolumeError("Invalid level.");
		return this->m_pData + lLevel * this->sSize.y * this->sSize.y * this->sSize.z;
	};

	/**
	 Dumps a x/y-slice to the console

	 @param lLevel the level of which the slice is to be dumped
	 @param lZ the slice in z-direction to be dumped
	 */
	void dumpLevelXYSlice(long lLevel, long lZ) {
		if(lLevel >= this->m_lLevels)
			throw VolumeError("Invalid level.");
		if(lZ >= this->sSize.z)
			throw VolumeError("Invalid z-slice.");
		T* pData = this->m_pData + (lLevel * this->sSize.y * this->sSize.y * this->sSize.z);
		for(long y = 60; y <= 67; ++y) {
			for(long x = 60; x <= 67; ++x) {
				size_t sL = lZ * this->sSize.x * this->sSize.y + y * this->sSize.x + x;
				OUT_INFO("%.3f, ", pData[sL]);
				if(x == 67) {
					OUT_INFO("%.3f, \n", pData[sL]);
				}
			}
		}
	};

	/**
	 Dumps a x/z-slice to the console

	 @param lLevel the level of which the slice is to be dumped
	 @param lY the slice in y-direction to be dumped
	 */
	void dumpLevelXZSlice(long lLevel, long lY) {
		if(lLevel >= this->m_lLevels)
			throw VolumeError("Invalid level.");
		if(lY >= this->sSize.z)
			throw VolumeError("Invalid y-slice.");
		T* pData = this->m_pData + (lLevel * this->sSize.y * this->sSize.y * this->sSize.z);
		for(long z = 60; z <= 67; ++z) {
			for(long x = 60; x <= 67; ++x) {
				size_t sL = z * this->sSize.x * this->sSize.y + lY * this->sSize.x + x;
				OUT_INFO("%.3f, ", pData[sL]);
				if(x == 67) {
					OUT_INFO("%.3f, \n", pData[sL]);
				}
			}
		}
	};

	/**
	 Dumps a y/z-slice to the console

	 @param lLevel the level of which the slice is to be dumped
	 @param lX the slice in x-direction to be dumped
	 */
	void dumpLevelYZSlice(long lLevel, long lX) {
		if(lLevel >= this->m_lLevels)
			throw VolumeError("Invalid level.");
		if(lX >= this->sSize.x)
			throw VolumeError("Invalid x-slice.");
		T* pData = this->m_pData + (lLevel * this->sSize.y * this->sSize.y * this->sSize.z);
		for(long z = 60; z <= 67; ++z) {
			for(long y = 60; y <= 67; ++y) {
				size_t sL = z * this->sSize.x * this->sSize.y + y * this->sSize.x + lX;
				OUT_INFO("%.3f, ", pData[sL]);
				if(y == 67) {
					OUT_INFO("%.3f, \n", pData[sL]);
				}
			}
		}
	};

	void storeRawDat(const std::string& sFilename, long lLevel = 0, bool bOverwrite = true) {
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
		size_t sWriteSize = this->sSize.x * this->sSize.y * this->sSize.z;
		size_t sSizeWritten = 0;
		sSizeWritten = fpr.write(this->m_pData + (lLevel * sWriteSize), sizeof(T), sWriteSize);
		
		OUT_INFO("done [%li bytes].\n", sSizeWritten * sizeof(T));
	};
};
};

#endif /* __MULTILEVEL_VOLUME_H__ */


