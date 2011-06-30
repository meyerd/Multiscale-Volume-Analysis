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

#ifndef __VOLUME_PROCESSOR_H__
#define __VOLUME_PROCESSOR_H__

#include "global.h"

#include <stdexcept>

#include "BasicVolume.h"
#include "MultilevelVolume.h"

class VolumeProcessorError : public std::runtime_error {
public:
	VolumeProcessorError(const std::string& sWhat) : runtime_error(sWhat) {};
};

class VolumeProcessor {
public:
	VolumeProcessor(void);
	VolumeProcessor(int iDeviceId);
	~VolumeProcessor(void);

	Volume::MultilevelVolumeContainer<float, 3>* dwtForward(const Volume::BasicVolume<float>& input, long lLevels, Volume::MultilevelVolume<float>* lowpass = NULL);
	Volume::MultilevelVolumeContainer<float, 3>* modulusMaximaAngles(Volume::MultilevelVolumeContainer<float, 3>& input);
	Volume::MultilevelVolumeContainer<float, 3>* calculateLipschitz(Volume::MultilevelVolumeContainer<float, 3>& input);

private:
	bool initializeCuda(bool bChooseFastest = true, int iDeviceId = 0);

	bool m_bCudaInitialized;
};

#endif /* __VOLUME_PROCESSOR_H__ */
