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
    CTMfloat* aux;
} Hemi;

typedef struct Subject {
    char name[128];
    float xfm[16];
    Hemi left;
    Hemi right;
} Subject;

typedef struct MinMax {
    float min[3];
    float max[3];
} MinMax;

enum States { READ_HEAD, READ_PTNUM, READ_POLYNUM, SKIP_PTS, SKIP_POLYS, READ_PTS, READ_POLYS };

Mesh* readVTK(const char* filename, bool readpolys);
MinMax* meshMinMax(Mesh* mesh);
void meshShift(Mesh* mesh, MinMax*);
void meshNudge(Mesh* mesh, bool right);
void meshFree(Mesh* mesh);
void minmaxFree(MinMax* minmax);