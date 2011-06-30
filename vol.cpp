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

#include "global.h"

#include <stdexcept>
#include <typeinfo>
#include <sstream>

using namespace std;

#include "VolumeProcessor.h"


// this is a example on how to use the volume processor
//  * it calculates the dwt of the input volume
//  * then searches for modulus maxima
//  * and estimates the lipschitz alpha
// each step will be saved to a intermediate output .raw file
// visualization with the eqRay tool can be done via saving 
// to raw/dat files

int main(int argc, char* argv[]) {
	try {
		VolumeProcessor processor;

		Volume::BasicVolume<float> in("d:\\meyerd\\VolumesRaw\\mri_head_128.raw");

		const long lLevels = 5;

		Volume::MultilevelVolume<float>* lowpass = new Volume::MultilevelVolume<float>(in.sSize, lLevels);
		Volume::MultilevelVolumeContainer<float, 3>* out = processor.dwtForward(in, lLevels, lowpass);
		
		// store intermediate dwt transformed data
		out->store("dwt.multiraw");

		// example for debugging outputs
		//out->x.writeSliceToFile(out->x.sSize.z / 2, "d:\\meyerd\\vol_out\\midout.raw", 0);

		/*out->x.dumpLevelSlice(0, 64);
		out->x.dumpLevelSlice(1, 64);
		out->x.dumpLevelSlice(2, 64);*/

		SAFE_DELETE(lowpass);

		// example for debugging outputs
		//out->x.store("d:\\meyerd\\vol_out\\dwtx.raw");
		//out->y.store("d:\\meyerd\\vol_out\\dwty.raw");
		//out->z.store("d:\\meyerd\\vol_out\\dwtz.raw");

		// example for debugging outputs
		//for(int i = 0; i < lLevels; ++i) {
		//	stringstream ssx;
		//	ssx << "d:\\meyerd\\vol_out\\dwtx_vis" << i;
		//	out->x.storeRawDat(ssx.str(), i);
		//	stringstream ssy;
		//	ssy << "d:\\meyerd\\vol_out\\dwty_vis" << i;
		//	out->y.storeRawDat(ssy.str(), i);
		//	stringstream ssz;
		//	ssz << "d:\\meyerd\\vol_out\\dwtz_vis" << i;
		//	out->z.storeRawDat(ssz.str(), i);
		//}

		Volume::MultilevelVolumeContainer<float, 3>* angles_modulus = processor.modulusMaximaAngles(*out);
		angles_modulus->store("d:\\meyerd\\vol_out\\modulus.multiraw");

		SAFE_DELETE(out);

		Volume::MultilevelVolumeContainer<float, 3>* lipschitz = processor.calculateLipschitz(*angles_modulus);

		lipschitz->x.store("d:\\meyerd\\vol_out\\lipschitz.raw");

		SAFE_DELETE(angles_modulus);
		SAFE_DELETE(lipschitz);
	} catch(exception& e) {
		OUT_ERR("Exception: %s: %s\n", typeid(e).name(), e.what());
	}
#ifdef WIN32
	
	OUT_INFO("done.\n");
	getchar();
#endif
	return 0;
}


