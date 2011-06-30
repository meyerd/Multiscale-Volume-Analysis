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

#ifndef __GLOBAL_H__
#define __GLOBAL_H__

/** configuration parameters 
 *  ... sorry no configuration file yet 
 */

/* Maxima calculation */
// if 1 do search for maxima in the modulus values
#define DO_MAXIMA 1

/* Thresholding */
// activate thresholding
#define THRESHOLDING 0
// default threshold value, if no automatic thresholding is selected
#define DEFAULT_THRESHOLD 40.0f
// automatically determine thershold value, overrides DEFAULT_THRESHOLD
// if selected 1) T = sqrt(sigma^2 * 2 * ln(N))
// if selected 2) T = MAD sqrt(ln(N)) // MAD - median absolute deviation
#define AUTO_THRESHOLD 0
// do soft thresholding instead of hard thresholding, if 2 is selected then first soft thresholding
// followed by hard thresholding is done
#define SOFT_THRESHOLDING 0

/* Maxima tracing */
// 1 for only tracing the maxima without calculating lipschitz
// the resulting volume will contain the levels up to which the maxima could be traced
#define ONLY_TRACING 0
#define TRACE_ONLY_ASCENDING_MAXIMA 0
#define TRACE_ONLY_DESCENDING_MAXIMA 0
// tolerance inbetween which gradient directions will be regarded as same (dot product!)
#define DEFAULT_ANGLE_TOLERANCE 0.7f

/* Lipschitz exponent gradient descent parameters */
// select which method is done to estimate the lipschitz alpha
#define LIPSCHITZ_METHOD 1						// 1 - conjugate gradient descent, 2 - linear
// max. iterations after which the gradient descent is given up if no result is found
#define GRADIENT_DESCENT_MAX_ITERATIONS 3000
// stepsize taken in each step towards the local minimum
#define GRADIENT_DESCENT_LAMBDA_STEPSIZE 1e-2f
// tolerance below which the gradient descent is stopped
#define GRADIENT_DESCENT_TOLERANCE 1e-6f

/* end configuration parameters */

#ifdef WIN32
#	ifdef DEBUG
#		define _DEBUG
#	endif
#	ifdef _DEBUG
#		define DEBUG
#	endif
#endif

/* fix for compiling cuda on "praktikumsrechner" ... */
#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include "targetver.h"
//#include <tchar.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
/* end fix */

#include "typenaming.h"

#define SAFE_DELETE(x) {if((x)) {delete (x); (x) = NULL;}};
#define SAFE_DELETE_ARRAY(x) {if((x)) {delete[] (x); (x) = NULL;}};

#define TRUE 1
#define FALSE 0

#ifdef WIN32
	#define FILE_SEP "\\"
#else
	#define FILE_SEP "/"
#endif

#ifndef WIN32
typedef int64_t __int64;
//#define __int64 int64_t
#endif

/* stolen from cutil ... */
// Give a little more for Windows : the console window often disapears before we can read the message
#ifdef WIN32

#include <Windows.h>
#pragma warning( disable : 4996 ) // disable deprecated warning 

# if 1 //ndef UNICODE
#  ifdef _DEBUG // Do this only in debug mode...
	inline void VSPrintf(FILE *fp, const char* file, const int line, LPCSTR fmt, ...) {
		size_t tmp_sz = 4096;
		char* tmp = (char*)malloc(sizeof(char)*tmp_sz);
		_snprintf(tmp, tmp_sz, "%s(%i) : %s", file, line, fmt);
		size_t fmt2_sz	= 2048;
		char *fmt2		= (char*)malloc(fmt2_sz);
		va_list  vlist;
		va_start(vlist, fmt);
		while((_vsnprintf(fmt2, fmt2_sz, tmp, vlist)) < 0) // means there wasn't anough room
		{
			fmt2_sz *= 2;
			if(fmt2) free(fmt2);
			fmt2 = (char*)malloc(fmt2_sz);
		}
		OutputDebugStringA(fmt2);
		fprintf(fp, fmt2);
		fflush(fp);
		free(fmt2);
		free(tmp);
	};
	inline void VSPrintfS(FILE *fp, LPCSTR fmt, ...) {
		size_t fmt2_sz	= 2048;
		char *fmt2		= (char*)malloc(fmt2_sz);
		va_list  vlist;
		va_start(vlist, fmt);
		while((_vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
		{
			fmt2_sz *= 2;
			if(fmt2) free(fmt2);
			fmt2 = (char*)malloc(fmt2_sz);
		}
		OutputDebugStringA(fmt2);
		fprintf(fp, fmt2);
		fflush(fp);
		free(fmt2);
	};
#endif
#endif
#endif //win32

// NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
// when the user double clicks on the error line in the Output pane. Like any compile error.

/* end stolen from cutil */

#if defined(_WIN32)
# if defined(_WIN64)
#  define FORCE_UNDEFINED_SYMBOL(x) __pragma(comment (linker, "/export:" #x))
# else
#  define FORCE_UNDEFINED_SYMBOL(x) __pragma(comment (linker, "/export:_" #x))
# endif
#else
# define FORCE_UNDEFINED_SYMBOL(x) extern "C" void x(void); void (*__ ## x ## _fp)(void)=&x;
#endif

#ifdef DEBUG
#ifdef WIN32
	//#define DEBUG_OUT(format, ...) {VSPrintf(stderr, __FILE__, __LINE__, format, __VA_ARGS__);};
	#define DEBUG_OUT(format, ...) {VSPrintfS(stderr, format, __VA_ARGS__);};
	//#define DEBUG_OUT(format, ...) {fprintf(stdout, format, __VA_ARGS__);};
#else
	#define DEBUG_OUT(format, args...) {fprintf(stdout, format, ##args);};
#endif
#else
#ifdef WIN32
	#define DEBUG_OUT(format, ...) {};
#else
#define DEBUG_OUT(format, args...) {};
#endif
#endif

#ifdef DEBUG
#	define OUT_INFO DEBUG_OUT
#	define OUT_WARN DEBUG_OUT
#	define OUT_ERR	DEBUG_OUT
#else
#	ifdef WIN32
#		define OUT_INFO(format, ...) {fprintf(stdout, format, __VA_ARGS__);};
#		define OUT_WARN(format, ...) {fprintf(stdout, format, __VA_ARGS__);};
#		define OUT_ERR(format, ...) {fprintf(stderr, format, __VA_ARGS__);};
#	else
#		define OUT_INFO(format, args...) {fprintf(stdout, format, ##args);};
#		define OUT_WARN(format, args...) {fprintf(stdout, format, ##args);};
#		define OUT_ERR(format, args...) {fprintf(stderr, format, ##args);};
#	endif
#endif

#include <string>

namespace Global {
	void checkIfFileExists(const std::string& sFilename);
};

#endif
