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

#ifndef __FILE_HANDLE_H__
#define __FILE_HANDLE_H__

#include "global.h"

#include <stdexcept>
#include <string>
#include <sstream>

class FileError : public std::runtime_error {
public:
	FileError(const std::string& sWhat, const std::string& sFilename) :
	  runtime_error(sWhat + " (" + sFilename + ")") {};
};

class FileWriteError : public FileError {
public:
	FileWriteError(const size_t sWritten, const size_t sTarget, const std::string& sFilename) :
		FileError("", "") {
		std::stringstream out;
		out << "Could only write " << sWritten << " bytes of " << sTarget << " bytes";
		FileError(out.str(), sFilename);
	};
};

class FileReadError : public FileError {
public:
	FileReadError(const size_t sWritten, const size_t sTarget, const std::string& sFilename) :
		FileError("", "") {
		std::stringstream out;
		out << "Could only read " << sWritten << " bytes of " << sTarget << " bytes";
		FileError(out.str(), sFilename);
	};
};

class FileHandle {
private:
	FILE* p;
	std::string m_sFilename;
public:
	FileHandle(const std::string& sFilename, const std::string& sAccessMode) :
	  m_sFilename(sFilename) {
		p = NULL;
#ifdef WIN32
		fopen_s(&p, sFilename.c_str(), sAccessMode.c_str());
#else
		p = fopen(sFilename.c_str(), sAccessMode.c_str());
#endif
		if(!p)
			throw FileError("Could not open file.", sFilename);
	};
	~FileHandle() {
		fclose(p);
	};

	operator FILE*() {
		return p;
	};

	size_t write(const void* pSrc, size_t sElemSize, size_t sNumElems) {
		size_t sElemsWritten = 0;
		sElemsWritten = fwrite(pSrc, sElemSize, sNumElems, p);
		if(sElemsWritten < sNumElems)
			throw FileWriteError(sElemSize * sElemsWritten, sElemSize * sNumElems, m_sFilename);
		return sElemsWritten;
	}

	size_t read(void* pDest, size_t sElemSize, size_t sNumElems) {
		size_t sElemsRead = 0;
		sElemsRead = fread(pDest, sElemSize, sNumElems, p);
		if(ferror(p))
			throw FileReadError(sElemSize * sElemsRead, sElemSize * sNumElems, m_sFilename);
		return sElemsRead;
	}

	bool eof() {
		return feof(p)?true:false;
	}

	__int64 getSize() {
		__int64 i64Filesize = 0;
#ifdef WIN32
		_fseeki64(p, 0, SEEK_END);
		i64Filesize = _ftelli64(p);
		_fseeki64(p, 0, SEEK_SET);
#else
		fseek(p, 0, SEEK_END);
		i64Filesize = (__int64)ftell(p);
		fseek(p, 0, SEEK_SET);
#endif
		return i64Filesize;
	};
};


#endif /* __FILE_HANDLE_H__ */
