cimport numpy as np
cimport stdio

cpdef object readVTK(bytes fname):
	cdef np.ndarray pts
	cdef np.ndarray polys

	cdef char line[8192]
	cdef FILE* fp = fopen(fname, 'r')
	if fp is NULL:
		raise IOError

	