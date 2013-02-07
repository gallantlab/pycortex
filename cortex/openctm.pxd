cdef extern from "openctm.h":
	ctypedef enum CTMenum:
		CTM_NONE              = 0x0000, 
		                              
		CTM_INVALID_CONTEXT   = 0x0001, 
		CTM_INVALID_ARGUMENT  = 0x0002, 
		CTM_INVALID_OPERATION = 0x0003, 
		CTM_INVALID_MESH      = 0x0004, 
		CTM_OUT_OF_MEMORY     = 0x0005, 
		CTM_FILE_ERROR        = 0x0006, 
		CTM_BAD_FORMAT        = 0x0007, 
		CTM_LZMA_ERROR        = 0x0008, 
		CTM_INTERNAL_ERROR    = 0x0009, 
		CTM_UNSUPPORTED_FORMAT_VERSION = 0x000A, 

		CTM_IMPORT            = 0x0101, 
		CTM_EXPORT            = 0x0102, 

		CTM_METHOD_RAW        = 0x0201, 
		CTM_METHOD_MG1        = 0x0202, 
		CTM_METHOD_MG2        = 0x0203, 

		CTM_VERTEX_COUNT      = 0x0301, 
		CTM_TRIANGLE_COUNT    = 0x0302, 
		CTM_HAS_NORMALS       = 0x0303, 
		CTM_UV_MAP_COUNT      = 0x0304, 
		CTM_ATTRIB_MAP_COUNT  = 0x0305, 
		CTM_VERTEX_PRECISION  = 0x0306, 
		CTM_NORMAL_PRECISION  = 0x0307, 
		CTM_COMPRESSION_METHOD = 0x0308, 
		CTM_FILE_COMMENT      = 0x0309, 

		CTM_NAME              = 0x0501, 
		CTM_FILE_NAME         = 0x0502, 
		CTM_PRECISION         = 0x0503, 

		CTM_INDICES           = 0x0601, 
		CTM_VERTICES          = 0x0602, 
		CTM_NORMALS           = 0x0603, 
		CTM_UV_MAP_1          = 0x0700, 
		CTM_UV_MAP_2          = 0x0701, 
		CTM_UV_MAP_3          = 0x0702, 
		CTM_UV_MAP_4          = 0x0703, 
		CTM_UV_MAP_5          = 0x0704, 
		CTM_UV_MAP_6          = 0x0705, 
		CTM_UV_MAP_7          = 0x0706, 
		CTM_UV_MAP_8          = 0x0707, 
		CTM_ATTRIB_MAP_1      = 0x0800, 
		CTM_ATTRIB_MAP_2      = 0x0801, 
		CTM_ATTRIB_MAP_3      = 0x0802, 
		CTM_ATTRIB_MAP_4      = 0x0803, 
		CTM_ATTRIB_MAP_5      = 0x0804, 
		CTM_ATTRIB_MAP_6      = 0x0805, 
		CTM_ATTRIB_MAP_7      = 0x0806, 
		CTM_ATTRIB_MAP_8      = 0x0807

	ctypedef void* CTMcontext

	CTMcontext ctmNewContext(CTMenum)
	void ctmFreeContext(CTMcontext)
	CTMenum ctmGetError(CTMcontext)
	char* ctmErrorString(CTMenum)
	unsigned int ctmGetInteger(CTMcontext, CTMenum)
	float ctmGetFloat(CTMcontext, CTMenum)
	unsigned int* ctmGetIntegerArray(CTMcontext, CTMenum)
	float* ctmGetFloatArray(CTMcontext, CTMenum)
	CTMenum ctmGetNamedUVMap(CTMcontext, char*)
	char* ctmGetUVMapString(CTMcontext, CTMenum, CTMenum)
	float ctmGetUVMapFloat(CTMcontext, CTMenum, CTMenum)
	CTMenum ctmGetNamedAttribMap(CTMcontext, char*)
	char* ctmGetAttribMapString(CTMcontext, CTMenum, CTMenum)
	float ctmGetAttribMapFloat(CTMcontext, CTMenum, CTMenum)
	char* ctmGetString(CTMcontext, CTMenum)
	void ctmCompressionMethod(CTMcontext, CTMenum)
	void ctmCompressionLevel(CTMcontext, unsigned int)
	void ctmVertexPrecision(CTMcontext, float)
	void ctmVertexPrecisionRel(CTMcontext, float)
	void ctmNormalPrecision(CTMcontext, float)
	void ctmUVCoordPrecision(CTMcontext, CTMenum, float)
	void ctmAttribPrecision(CTMcontext, CTMenum, float)
	void ctmFileComment(CTMcontext, char*)
	void ctmDefineMesh(CTMcontext, float*, unsigned int, unsigned int*, unsigned int, float*)
	CTMenum ctmAddUVMap(CTMcontext, float*, char*, char*)
	CTMenum ctmAddAttribMap(CTMcontext, float*, char*)
	void ctmLoad(CTMcontext, char*)
	void ctmSave(CTMcontext, char*)