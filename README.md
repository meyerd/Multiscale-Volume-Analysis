vol & volume\_helpers
=====================

Multiscale volume processing and Lipschitz/Hoelder exponent estimation on
volumetric datasets. It is the implementation of the work presented in the
master's thesis "GPU-based Multiscale Analysis of Volume Data", available 
here https://github.com/downloads/meyerd/Multiscale-Volume-Analysis/thesis.pdf.

The "vol" project contains the main program for multiscale processing 
of volumes. vol.cpp contains a example on how to use the processing library.

The "volume\_helpers" project contains helper programs for converting 
and pre/pos-processing of volumes:

* pvm2myraw:
  convert from the pvm format the the internally used raw format
  shamelessly taken from http://www9.informatik.uni-erlangen.de/External/vollib/
  example volumes are also available from there
* raw\_resample:
  resample a raw volume to a different resolution
* rawdat2raw:
  convert from the visualization raw/dat format for eqRay to the internally
  used raw format
* testvol\_gen:
  create simple test volumes
* volume\_channel\_combine:
  combine multiple raw volumes to a multichannel raw/dat volume file for visualization
  purposes
* volume\_header\_prepend:
  prepend a raw volume header to unstructured raw data files


Configuration
=============

All compile time configuration parameters can be tweaked by editing
global.h, except the CUDA block grid sizes. Those can be configured
in each respective \*\_kernel.h file.

Building
========

For building vol the CUDA 3.2 toolkit is required.


Windows
-------

For building on windows just use the vol.sln and volume\_helpers.sln project
files for building. If the CUDA 3.2 toolkit and the CUDA SDK examples were
installed correctly everything should work.

If you like to build the project with cmake customize the CMakeLists.txt
and customize the variable CUD\_TOOLKIT\_ROOT\_DIR to 

     set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})

for example.

Then run cmake and build the generated project files.

Linux
-----

On Linux use the supplied cmake files.
Customize in the CMakeLists.txt

     set(CUDA_TOOLKIT_ROOT_DIR ~/CUDA_SDK)
     set(CUDA_SDK\_ROOT_DIR ~/CUDA_SDK)
     
     set(CUDA_NVCC_FLAGS --compiler-bindir ~/CUDA_SDK/gcc/)
     
     set(CUDA_BUILD_EMULATION OFF)
     set(CUDA_BUILD_CUBIN ON)

then run cmake and build the projects with make.
