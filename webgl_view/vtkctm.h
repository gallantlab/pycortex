#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <openctm.h>

typedef struct Mesh {
    CTMuint npts;
    CTMuint npolys;
    CTMuint nelem;
    CTMfloat* pts;
    CTMuint* polys;
} Mesh;

typedef struct Hemi {
    Mesh* fiducial;
    Mesh* flat;
    Mesh* between[6];
    char names[6][1024];
    unsigned int nbetween;
    CTMfloat* datamap;
} Hemi;

typedef struct Subject {
    char name[128];
    float xfm[16];
    Hemi left;
    Hemi right;
} Subject;

enum States { READ_HEAD, READ_PTNUM, READ_POLYNUM, SKIP_PTS, SKIP_POLYS, READ_PTS, READ_POLYS };

Mesh* readVTK(const char* filename, bool readpolys);
void meshMinMax(Mesh* mesh, float* min, float* max);
void meshShift(Mesh* mesh, float* add, float* div);
void meshNudge(Mesh* mes, bool right);
void meshFree(Mesh* mesh);