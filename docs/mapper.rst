Mapper
======
This module contains functions and classes that translate between Volumes of data and the corresponding Vertices of a surface using any of several samplers. That is, if you have some data defined on a per-voxel basis (whether it's BOLD or a warp vector doesn't matter), you can create a Volume and use the map() function to sample from that data at the vertices of a surface. And you can specify the sampling method, be it nearest-neighbor, trilinear, sinc, etc...

Important Classes: