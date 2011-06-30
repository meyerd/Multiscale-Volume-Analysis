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

#ifndef __TYPENAMING_H__
#define __TYPENAMING_H__

#ifndef WIN32
#include <typeinfo>
#endif

namespace VolumeTypenaming {

template<typename Typenaming>
struct TypeNameTraits;

#define REGISTER_TYPE_NAME(X) template <> struct TypeNameTraits<X> \
    { static const char* name; } ; const char* TypeNameTraits<X>::name = #X

}
#endif /* __TYPENAMING_H__ */
