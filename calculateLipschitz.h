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

#ifndef __CALCULATE_LIPSCHITZ_H__
#define __CALCULATE_LIPSCHITZ_H__

#include "global.h"

#include <string>

#include "BasicVolume.h"
#include "MultilevelVolume.h"

using namespace Volume;

namespace Cuda { namespace Lipschitz {
	class CudaLipschitzError : public std::runtime_error {
	public:
		CudaLipschitzError(const std::string& sWhat) : runtime_error(sWhat) {};
	};
	Volume::MultilevelVolumeContainer<float, 3>* calculateLipschitz(Volume::MultilevelVolumeContainer<float, 3>& input);
};};

namespace CPU { namespace Lipschitz {
	class CpuLipschitzError : public std::runtime_error {
	public:
		CpuLipschitzError(const std::string& sWhat) : runtime_error(sWhat) {};
	};
	Volume::MultilevelVolumeContainer<float, 3>* calculateLipschitz(Volume::MultilevelVolumeContainer<float, 3>& input);
};};

#endif /* __CALCULATE_LIPSCHITZ_H__ */