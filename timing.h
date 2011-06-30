/**************************************************************************************************************

(c) 2006 by Jens Schneider, Joachim Georgii, TUM.3D
	mailto:jens.schneider@in.tum.de
	mailto:georgii@in.tum.de
	Computer Graphics and Visualization Group
    Institute for Computer Science I15
	Technical University of Munich

	Supported Platforms
	- Win32  (QueryPerformanceCounter & GetTickCount)
	- Linux  (gettimeofday)
	- Win CE (GetTickCount)
**************************************************************************************************************/

#ifndef __TUM3D__CTIMER__
#define __TUM3D__CTIMER__

#ifdef _DEBUG
#define _TUM3D_DEBUG_	///< switches on debugging output - can be changed per-class.
#endif
#ifdef _WIN32
#ifdef _TUM3D_DEBUG_
#pragma message("    [TUM.3D:CTimer] DEBUG ON.\n")
#else
#pragma message("    [TUM.3D:CTimer] DEBUG OFF.\n")
#endif
#endif

// WINCE: winbase.h, link with coredll.lib

#ifdef _WIN32
#include<windows.h>
#include<winbase.h>
#include<stdio.h>

// Codepath for plain win32 & win ce
class CTimer {
	public:
		CTimer() {	
			Init();
		}
		~CTimer() {};

		inline void Reset(void) {			
			if (m_bHasPerfCounter) GetPerfCounter(m_liTime);
			else m_dwTicks=GetTickCount();
			m_bStopped=false;
			m_dLastTime=0.0;
		}
		
		inline double Pause(void) {
			m_dLastTime=Query();
			m_bStopped=true;
			return m_dLastTime;
		}

		inline double Resume(void) {
			m_bStopped=false;
			if (m_bHasPerfCounter) GetPerfCounter(m_liTime);
			else m_dwTicks=GetTickCount();
			return m_dLastTime;
		}

		inline double Query(void) const {
			if (!m_bStopped) {
				if (m_bHasPerfCounter) {				
					LARGE_INTEGER liCurrent;				
					GetPerfCounter(liCurrent);
					return (double(liCurrent.QuadPart-m_liTime.QuadPart))/double(m_liFrequency.QuadPart)+m_dLastTime;
				}
				else {
					DWORD dwCurrent=GetTickCount();
					return double(dwCurrent-m_dwTicks)/1000.0+m_dLastTime;
				}
			}
			else return m_dLastTime;
		}

	private:
		inline void Init(void) {
			if (QueryPerformanceFrequency(&m_liFrequency)!=0) m_bHasPerfCounter=true;
			Reset();
		}
		inline void	GetPerfCounter(LARGE_INTEGER &liCounter) const {
			#ifdef _TUM3D_DEBUG
				if (QueryPerformanceCounter(&liCounter)==0) debugmessage("ERROR calling QueryPerformanceCounter\n");
			#else
				QueryPerformanceCounter(&liCounter);
			#endif
		}

		inline void	debugmessage(const char* str) const {
			fprintf(stderr,str); fflush(stderr);
		}
		// Performance Counter state
		LARGE_INTEGER	m_liFrequency;
		LARGE_INTEGER	m_liTime;	
		bool			m_bHasPerfCounter;
		// Tick Counter state
		DWORD			m_dwTicks;
		double			m_dLastTime;
		bool			m_bStopped;
	};
#else
	// Codepath for non win32
	#  include <sys/time.h>

	class CTimer {
	public:
		CTimer() {
		  Reset();
		};

		~CTimer() {};

		inline double Pause(void) {
			m_dLastTime=Query();
			m_bStopped=true;
			return m_dLastTime;
		}

		inline double Resume(void) {
			m_bStopped=false;
			gettimeofday(&m_tsTime,0);
			return m_dLastTime;
		}

		inline void Reset(void) {
			gettimeofday(&m_tsTime,0);
			m_dLastTime = 0.0;
			m_bStopped  = false;
		}

		inline double Query(void) const {
			struct timeval tsCurrent;
			gettimeofday(&tsCurrent,0);
			return (double(tsCurrent.tv_sec - m_tsTime.tv_sec)+double(tsCurrent.tv_usec-m_tsTime.tv_usec)/1e6)+m_dLastTime;
		}

	private:
		struct	timeval m_tsTime;
		double	m_dLastTime;
		bool	m_bStopped;
	};

#endif

#endif

