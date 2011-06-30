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

#include <vector>
#include <iomanip>

#include <float.h>

#include "../GeneralVolume.h"
#include "../BasicVolume.h"
#include "../MultilevelVolume.h"

#include "../FileHandle.h"

using namespace std;
using namespace Volume;

#define TYPE float

//#define LEVEL_OF_FIRST_FIXED 1

int main(int argc, char* argv[])
{
	if(argc < 3) {
		OUT_ERR("usage: volume_channel_combine <normalize_to_channel> <input_channel_01.raw> <input_channel_02.raw> ... <output_vis>\n");
		OUT_ERR("         use -1 as normalization channel specification to indicate no normalization\n");
		return 1;
	}

	int iVolumeChannels = argc - 3;
	if(iVolumeChannels > 4) {
		OUT_ERR("only up to 4 channels supported\n");
		return 1;
	}

	int iNormalizeTo = 0;
	iNormalizeTo = atoi(argv[1]);
	iNormalizeTo -= 1;
	bool bNormalization = true;
	if(iNormalizeTo == -2)
		bNormalization = false;
	if(bNormalization && (iNormalizeTo < 0 || iNormalizeTo >= iVolumeChannels)) {
		OUT_ERR("normalize must indicate a given channel (0-%i) (%i)\n", iVolumeChannels - 1, iNormalizeTo);
		return 1;
	}

	try {
		vector<MultilevelVolume<TYPE>*> vols;
		vector<vector<long> > levels;
		vector<TYPE> normMin;
		vector<TYPE> normMax;
		levels.resize(iVolumeChannels);

		for(int i = 0; i < iVolumeChannels; ++i) {
			//OUT_INFO("Loading volume \"%s\" ...\n", argv[i + 1]);
			MultilevelVolume<TYPE>* vol = new MultilevelVolume<TYPE>(argv[i + 2]);
			vols.push_back(vol);
		}

		long lLevels = 0;
		for(int i = 0; i < iVolumeChannels; ++i) {
			if(vols[i]->m_lLevels > lLevels)
				lLevels = vols[i]->m_lLevels;
		}
		if(lLevels <= 0) {
			throw runtime_error("wrong number of levels");
		}
#ifdef LEVEL_OF_FIRST_FIXED
		for(long l = 0; l < lLevels; ++l) {
			levels[0].push_back(LEVEL_OF_FIRST_FIXED);
		}
		for(int i = 1; i < iVolumeChannels; ++i) {
#else
		for(int i = 0; i < iVolumeChannels; ++i) {
#endif
			if(vols[i]->m_lLevels != lLevels) {
				OUT_WARN("not all volumes have the same number of levels!\n");
				for(long l = 0; l < vols[i]->m_lLevels; ++l) {
					levels[i].push_back(l);
				}
				for(long l = vols[i]->m_lLevels; l < lLevels; ++l) {
					levels[i].push_back(vols[i]->m_lLevels - 1);
				}
			} else {
				for(long l = 0; l < lLevels; ++l) {
					levels[i].push_back(l);
				}
			}
		}

		OUT_INFO("Level info: \n");
		for(int i = 0; i < iVolumeChannels; ++i) {
			OUT_INFO(" InputVolumeChannel %02i: ", i);
			OUT_INFO("%02i\n", vols[i]->m_lLevels);
		}

		OUT_INFO("Level mapping: \n");
		OUT_INFO("                         ");
		for(long l = 0; l < lLevels; ++l) {
			OUT_INFO("%02i ", l);
		}
		OUT_INFO("\n");
		for(int i = 0; i < iVolumeChannels; ++i) {
			OUT_INFO(" OutputVolumeChannel %02i: ", i);
			for(long l = 0; l < lLevels; ++l) {
				OUT_INFO("%02i ", levels[i][l]);
			}
			OUT_INFO("\n");
		}


		normMin.resize(lLevels);
		normMax.resize(lLevels);
		for(long l = 0; l < lLevels; ++l) {
			normMin[l] = FLT_MAX;
			normMax[l] = FLT_MIN;
		}

		VolumeSize sSize = vols[0]->sSize;
		for(int i = 0; i < iVolumeChannels; ++i) {
			if(vols[i]->sSize != sSize)
				throw runtime_error("not all volumes have the same size\n");
		}

		if(bNormalization) {
			OUT_INFO("Finding min/max for normalization ... ");
			#pragma omp parallel for
			for(long l = 0; l < lLevels; ++l) {
			for(long z = 0; z < sSize.z; ++z) {
				for(long y = 0; y < sSize.y; ++y) {
					for(long x = 0; x < sSize.x; ++x) {
						TYPE tmp = vols[iNormalizeTo]->getLevel(levels[iNormalizeTo][l])[z * sSize.x * sSize.y + y * sSize.x + x];
						if(tmp < normMin[l])
							normMin[l] = tmp;
						if(tmp > normMax[l])
							normMax[l] = tmp;
					}
				}
			}
			}
			//OUT_INFO("min: %.3f, max: %.3f.\n", normMin[0], normMax[0]);
			OUT_INFO("done.\n");
			OUT_INFO("min/max info: \n");
			for(long l = 0; l < lLevels; ++l) {
				OUT_INFO(" Level %02li: min: %.4f, max: %.4f\n", l, normMin[l], normMax[l]);
			}
		}

		TYPE* pOutputData = new TYPE[sSize.x * sSize.y * sSize.z * iVolumeChannels];
		for(long l = 0; l < lLevels; ++l) {
			OUT_INFO("Putting together data for level %li... ", l);
			
			for(int i = 0; i < iVolumeChannels; ++i) {
				if(bNormalization && i != iNormalizeTo) {
					TYPE lmin = FLT_MAX;
					TYPE lmax = FLT_MIN;
					for(long z = 0; z < sSize.z; ++z) {
						for(long y = 0; y < sSize.y; ++y) {
							for(long x = 0; x < sSize.x; ++x) {
								TYPE tmp = vols[i]->getLevel(levels[i][l])[z * sSize.x * sSize.y + y * sSize.x + x];
								if(tmp < lmin)
									lmin = tmp;
								if(tmp > lmax)
									lmax = tmp;
							}
						}
					}
					#pragma omp parallel for
					for(long z = 0; z < sSize.z; ++z) {
						for(long y = 0; y < sSize.y; ++y) {
							for(long x = 0; x < sSize.x; ++x) {
								pOutputData[(z * sSize.x * sSize.y + y * sSize.x + x) * iVolumeChannels + i] = 
									(((vols[i]->getLevel(levels[i][l])[z * sSize.x * sSize.y + y * sSize.x + x] - lmin) / 
									 (lmax - lmin)) * (normMax[l] - normMin[l])) + normMin[l];
							}
						}
					}
				} else {
					#pragma omp parallel for
					for(long z = 0; z < sSize.z; ++z) {
						for(long y = 0; y < sSize.y; ++y) {
							for(long x = 0; x < sSize.x; ++x) {
								pOutputData[(z * sSize.x * sSize.y + y * sSize.x + x) * iVolumeChannels + i] = 
									vols[i]->getLevel(levels[i][l])[z * sSize.x * sSize.y + y * sSize.x + x];
							}
						}
					}
				}
			}
			OUT_INFO("done.\n");

		
			stringstream ssOutputFilename;
			ssOutputFilename << argv[iVolumeChannels + 2];
			ssOutputFilename.width(3);
			ssOutputFilename << std::setfill('0') << l;
			string sDatFilename = (ssOutputFilename.str()+".dat");
			string sRawFilename = (ssOutputFilename.str()+".raw");

			OUT_INFO("Writing volume raw/dat to file \"%s\"/\"%s\" ... ", sRawFilename.c_str(), sDatFilename.c_str());

			FileHandle fp(sDatFilename, "w");
			fprintf_s(fp, "ObjectFileName:\t%s\n", sRawFilename.c_str());
			fprintf_s(fp, "Resolution:\t%li %li %li\n", sSize.x, sSize.y, sSize.z);
			fprintf_s(fp, "SliceThickness:\t1.0 1.0 1.0\n");
			stringstream ssFormat;
			ssFormat << "FLOAT" << iVolumeChannels;
			fprintf_s(fp, "Format:\t\t%s\n", ssFormat.str().c_str());

			OUT_INFO("Writing raw data ... ");
			FileHandle fpr(sRawFilename, "wb");
			size_t sWriteSize = sSize.x * sSize.y * sSize.z * iVolumeChannels;
			size_t sSizeWritten = 0;
			sSizeWritten = fpr.write(pOutputData, sizeof(TYPE), sWriteSize);
		
			OUT_INFO("done [%li bytes].\n", sSizeWritten * sizeof(TYPE));
		}
		SAFE_DELETE_ARRAY(pOutputData);
		for(int i = 0; i < iVolumeChannels; ++i) {
			SAFE_DELETE(vols[i]);
		}
		vols.clear();
	} catch(exception& e) {
		OUT_ERR("Exception: %s: %s\n", typeid(e).name(), e.what());
	}

	return 0;
}

