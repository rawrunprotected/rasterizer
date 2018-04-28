# rasterizer

This project is a state-of-the-art software occlusion culling system.

It's similar in spirit to Intel's https://github.com/GameTechDev/OcclusionCulling, but uses completely different techniques and is 2-3 times faster in single-threaded AVX mode when rendering the full set of occluders (no minimum size).

Checkout http://threadlocalmutex.com/?p=144 for some implementation details.

Sample Data
===========

The folder Sponza contains a copy of Crytek's public domain Sponza model.

The folder Castle contains a copy of Intel's Occlusion Culling sample scene, redistributed here under the terms of the Intel Code Samples License.

Controls are WASD and cursor keys for the camera.

Requirements
============
- An AVX2-capable CPU (Haswell, Excavator or later)
- Visual Studio 2015 or higher

License
============

[!["Creative Commons Licence"](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).