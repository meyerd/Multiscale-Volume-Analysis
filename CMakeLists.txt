#The name of the project
PROJECT(vol)

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_BUILD_TYPE Debug)

find_package(OpenMP)

if(${OPENMP_FOUND}) 
	set(LIBIMPROC_OPEN_MP ON)
else(${OPENMP_FOUND})
	set(LIBIMPROC_OPEN_MP OFF)
endif(${OPENMP_FOUND})

if(${CMAKE_HOST_UNIX})
	set(CUDA_TOOLKIT_ROOT_DIR ~/CUDA_SDK)
	set(CUDA_SDK_ROOT_DIR ~/CUDA_SDK)
else(${CMAKE_HOST_UNIX})
	set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
	# set(CUDA_TOOLKIT_ROOT_DIR "c:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v3.2/")
	# set(CUDA_SDK_ROOT_DIR $ENV{CUDA_PATH})
endif(${CMAKE_HOST_UNIX})
if(${CMAKE_HOST_UNIX})
	set(CUDA_NVCC_FLAGS --compiler-bindir ~/CUDA_SDK/gcc/)
endif(${CMAKE_HOST_UNIX})

set(CUDA_BUILD_EMULATION ON)
set(CUDA_BUILD_CUBIN ON)

INCLUDE(FindCUDA)
FIND_PACKAGE(CUDA REQUIRED)

if(${CMAKE_HOST_UNIX})
	LINK_LIBRARIES("-L/${CUDA_SDK_ROOT_DIR}/lib")
else(${CMAKE_HOST_UNIX})
	# LINK_LIBRARIES("${CUDA_SDK_ROOT_DIR}/lib")
endif(${CMAKE_HOST_UNIX})
INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/include "./")


#set(CMAKE_VERBOSE_MAKEFILE ON)

# add the binary tree to the search path for include files
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}")
LINK_DIRECTORIES("${PROJECT_BINARY_DIR}")

# == vol ==

set(VOL_SOURCES vol.cpp typenaming.cpp VolumeProcessor.cpp dwtForward.cu calculateLipschitz.cu GeneralVolume.cpp modulusMaxima.cu global.cpp)
set(VOL_HEADERS BasicVolume.h MultilevelVolume.h FileHandle.h cuda_type_conversion.h cutil_replacement.h dwtForward.h GeneralVolume.h global.h targetver.h typenaming.h VolumeProcessor.h 2ddwt_kernel.h copy_kernel.h angle_kernel.h calculateLipschitz.h lipschitz_kernel.h maxima_kernel.h modulus_kernel.h modulusMaxima.h timing.h)

CUDA_ADD_EXECUTABLE(vol ${VOL_SOURCES} ${VOL_HEADERS})

# == testvol gen ==

set(TESTVOL_SOURCES testvol_gen/testvol_gen.cpp GeneralVolume.cpp global.cpp)
set(TESTVOL_HEADERS BasicVolume.h GeneralVolume.h MultilevelVolume.h FileHandle.h)

ADD_EXECUTABLE(testvol_gen/testvol_gen ${TESTVOL_SOURCES} ${TESTVOL_HEADERS})

# == pvm2myraw ==

set(PVM2MYRAW_SOURCES pvm2myraw/pvm2myraw.cpp pvm2myraw/ddsbase.cpp GeneralVolume.cpp global.cpp)
set(PVM2MYRAW_HEADERS BasicVolume.h GeneralVolume.h MultilevelVolume.h pvm2myraw/codebase.h pvm2myraw/ddsbase.h FileHandle.h)

ADD_EXECUTABLE(pvm2myraw/pvm2myraw ${PVM2MYRAW_SOURCES} ${PVM2MYRAW_HEADERS})

# == rawdat2raw ==

set(RAWDAT2RAW_SOURCES rawdat2raw/rawdat2raw.cpp GeneralVolume.cpp global.cpp)
set(RAWDAT2RAW_HEADERS BasicVolume.h GeneralVolume.h MultilevelVolume.h FileHandle.h)

ADD_EXECUTABLE(rawdat2raw/rawdat2raw ${RAWDAT2RAW_SOURCES} ${RAWDAT2RAW_HEADERS})

# == raw_resample ==

set(RAW_RESAMPLE_SOURCES raw_resample/raw_resample.cpp GeneralVolume.cpp global.cpp)
set(RAW_RESAMPLE_HEADERS BasicVolume.h GeneralVolume.h MultilevelVolume.h FileHandle.h)

ADD_EXECUTABLE(raw_resample/raw_resample ${RAW_RESAMPLE_SOURCES} ${RAW_RESAMPLE_HEADERS})

# == volume_header_prepend ==

set(VOLUME_HEADER_PREPEND_SOURCES volume_header_prepend/volume_header_prepend.cpp GeneralVolume.cpp global.cpp)
set(VOLUME_HEADER_PREPEND_HEADERS BasicVolume.h GeneralVolume.h MultilevelVolume.h FileHandle.h)

ADD_EXECUTABLE(volume_header_prepend/volume_header_prepend ${VOLUME_HEADER_PREPEND_SOURCES} ${VOLUME_HEADER_PREPEND_HEADERS})

# == volume_channel_combine ==

set(VOLUME_CHANNEL_COMBINE_SOURCES volume_channel_combine/volume_channel_combine.cpp GeneralVolume.cpp global.cpp)
set(VOLUME_CHANNEL_COMBINE_HEADERS BasicVolume.h GeneralVolume.h MultilevelVolume.h FileHandle.h)

ADD_EXECUTABLE(volume_channel_combine/volume_channel_combine ${VOLUME_CHANNEL_COMBINE_SOURCES} ${VOLUME_CHANNEL_COMBINE_HEADERS})
