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

#ifndef __GENERAL_VOLUME_H__
#define __GENERAL_VOLUME_H__

// TODO: convert from fread/fwrite/fopen/fclose to fstream ... maybe the own filehandle class is okay ..

#include "global.h"

#include <stdexcept>
#include <string>

#include "FileHandle.h"

namespace Volume {

typedef enum VolumeFileType_t {
	TYPE_UNKNOWN = 0,
	TYPE_BASIC_FLOAT,
	TYPE_BASIC_INT,
	TYPE_BASIC_UNSIGNED_CHAR,
	TYPE_MULTILEVEL_FLOAT,
	TYPE_MULTILEVEL_INT,
	TYPE_MULTILEVEL_UNSIGNED_CHAR,
} VolumeFileType;

typedef struct VolumeSize_t {
	VolumeSize_t() : x(0), y(0), z(0) {};
	VolumeSize_t(long lX, long lY, long lZ) : x(lX), y(lY), z(lZ) {};
	void operator= (const struct VolumeSize_t& o) {
		x = o.x;
		y = o.y;
		z = o.z;
	};
	long x;
	long y;
	long z;
} VolumeSize;

bool operator== (const VolumeSize& a, const VolumeSize& b);
bool operator!= (const VolumeSize& a, const VolumeSize& b);

template<long tx, long ty, long tz>
struct ConstVolumeSize_t {
	ConstVolumeSize_t() {};
	static const long x = tx;
	static const long y = ty;
	static const long z = tz;
	bool operator== (const struct VolumeSize_t& o) {
		return o.x == x && o.y == y && o.z == z;
	};
	bool operator!= (const struct VolumeSize_t& o) {
		return !(o.x == x && o.y == y && o.z == z);
	};
};
template<long x, long y, long z>
struct ConstVolumeSize:ConstVolumeSize_t<x, y, z> {
	ConstVolumeSize() {};
};

typedef struct VolumeHeader_t {
	VolumeHeader_t() : eFileType(TYPE_UNKNOWN), lLevels(0) {};
	VolumeFileType eFileType;
	VolumeSize sSize;
	long lLevels;
} VolumeHeader;

class VolumeError : public std::runtime_error {
public:
	VolumeError(const std::string& sWhat) : runtime_error(sWhat) {};
};
class VolumeFileError : public VolumeError {
public:
	VolumeFileError(const std::string& sWhat, const std::string& sFilename) : 
	  VolumeError(sWhat + " (" + sFilename + ")") {};
};
class VolumeOverwriteError : public VolumeFileError {
public:
	VolumeOverwriteError(const std::string& sWhat, const std::string& sFilename) :
		VolumeFileError(sWhat, sFilename) {};
};
class VolumeFilesizeError : public VolumeFileError {
public:
	VolumeFilesizeError(const std::string& sWhat, const std::string& sFilename) :
	  VolumeFileError(sWhat, sFilename) {};
};
class VolumeFiletypeError : public VolumeFileError {
public:
	VolumeFiletypeError(const std::string& sWhat, const std::string& sIsType, 
		const std::string& sExpectedType, const std::string& sFilename) :
	VolumeFileError(sWhat + " (Expected Type " + sExpectedType + " but file is type " + sIsType + ")", sFilename) {};
};
class VolumeCopyWarning : public VolumeError {
public:
	VolumeCopyWarning(void) : VolumeError("About to copy the volume, do you really want to do that?") {};
};

/** 
	Generic Volume Class used to derive special volumes BasicVolume, Multilevel Volume ...
	Should not be used directly
 */
template<typename T>
class GeneralVolume {
public:
	GeneralVolume(const GeneralVolume<T>& other) {
		sSize = other.sSize;
		m_lLevels = other.m_lLevels;
		m_bIsOk = other.m_bIsOk;
		m_sFileHeader = other.m_sFileHeader;
		m_i64Filesize = other.m_i64Filesize;
		throw VolumeCopyWarning();
	};
	GeneralVolume(void)	: m_lLevels(0),
		m_i64Filesize(0), m_pData(NULL), m_bIsOk(false) {};

	virtual ~GeneralVolume(void) {
		SAFE_DELETE_ARRAY(m_pData);
	};

	/**
		Get a pointer to the internal data.
	 */
	T* getData() {
		return m_pData;
	}
	
	/**
		Explicitly copy the data and attributes from another volume.
		Use this instead of the "=" operator, which is supposed to 
		throw an exception.
		@param other the volume to copy from
	 */
	void copy(const GeneralVolume<T>& other) {
		sSize = other.sSize;
		m_lLevels = other.m_lLevels;
		m_bIsOk = other.m_bIsOk;
		m_sFileHeader = other.m_sFileHeader;
		m_i64Filesize = other.m_i64Filesize;
		allocateDataBuffer();
		size_t sThisBufferSize = sSize.x * sSize.y * sSize.z * m_lLevels * sizeof(T);
		if(memcpy_s(m_pData, sThisBufferSize, other.m_pData, sThisBufferSize) != 0)
			throw VolumeError("Error copying volume.");
	}

	void checkFile(const std::string& sFilename) {
		FileHandle fp(sFilename, "rb");
		
		m_i64Filesize = fp.getSize();

		if(m_i64Filesize < sizeof(VolumeHeader))
			throw VolumeFilesizeError("Volumefile too small (doesn't even include the header)", sFilename);
	};
	
	/**
		Write only the header to output file
		@param sFilename filename
		@param bOverwrite allow overwriting existing files
	 */
	void writeHeader(const std::string& sFilename, bool bOverwrite = true) {
		if(!bOverwrite) {
			Global::checkIfFileExists(sFilename);
		}
		
		FileHandle fp(sFilename, "wb");
		fp.write(&m_sFileHeader, sizeof(VolumeHeader), 1);
	};

	/**
	 Allocates the internal data buffer.
	 The buffer will not be initialized!
	 */
	void allocateDataBuffer() {
		if(sSize.x < 1 || sSize.y < 1 || sSize.z < 1 || m_lLevels < 1)
			throw VolumeError("Volumesize not set.");

		SAFE_DELETE_ARRAY(m_pData);
		m_pData = new T[sSize.x * sSize.y * sSize.z * m_lLevels];
	};

	virtual void load(const std::string& sFilename) = 0;
	virtual void store(const std::string& sFilename, bool bOverwrite = true) = 0;

	VolumeSize sSize; ///< The size of the volume, size of one level if it is a multilevel volume.
	long m_lLevels; ///< Number of levels of the volume.

	T* m_pData; ///< Pointer to the volume data

	VolumeHeader m_sFileHeader; ///< Copy of the file header, if loaded from file.
	__int64 m_i64Filesize; ///< Filesize, if loaded from file.

	bool m_bIsOk; ///< Indicate if volume data has been set and is ok
private:
};
};

#endif /* __GENERAL_VOLUME_H__ */
