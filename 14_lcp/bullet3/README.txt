The source files in this directory are copied from Bulletphysics3 from the following location.

    https://github.com/bulletphysics/bullet3/commit/a1d96646c8ca28b99b2581dcfc4d74cc3b4de018

which is the head of the master branch as at Jun 11, 2022.

The source files have been editted at one location to change the path to the include header, and
at two locations to remove BT_PROFILE() macro, which is used for profiling.

The LICENSE of bulletphysics/bullet3 is as follows:

====================================

The files in this repository are licensed under the zlib license, except for the files under 'Extras' and examples/ThirdPartyLibs.

Bullet Continuous Collision Detection and Physics Library
http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

====================================
