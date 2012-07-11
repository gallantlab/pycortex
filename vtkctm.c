#include "vtkctm.h"

Mesh* readCTM(const char* filename, bool readpolys) {
    size_t bytes;
    CTMcontext* ctx = ctmNewContext(CTM_IMPORT);
    Mesh* mesh = calloc(1, sizeof(Mesh));
    if (mesh == NULL)
        return NULL;
    mesh->nelem = 3;

    ctmLoad(ctx, filename);
    if(ctmGetError(ctx) == CTM_NONE) {
        mesh->npts = ctmGetInteger(ctx, CTM_VERTEX_COUNT);
        mesh->npolys = ctmGetInteger(ctx, CTM_TRIANGLE_COUNT);
        bytes = mesh->npts*3*sizeof(CTMfloat);
        mesh->pts = malloc(bytes);
        memcpy(mesh->pts, ctmGetFloatArray(ctx, CTM_VERTICES), bytes);
        if (readpolys) {
            bytes = (mesh->npolys*3*sizeof(CTMuint));
            mesh->polys = malloc(bytes);
            memcpy(mesh->polys, ctmGetIntegerArray(ctx, CTM_INDICES), bytes);
        } else {
            mesh->npolys = 0;
        }
    }
    ctmFreeContext(ctx);
    return mesh;
}

Mesh* readVTK(const char* filename, bool readpolys) {
    char* buf;
    char bufline[8192];
    bool polycomplete = false, ptcomplete = false;
    unsigned int npts = 0, ptcount = 0;
    enum States state = READ_HEAD;
    FILE* fp = fopen(filename, "r");
    Mesh* mesh = calloc(1, sizeof(Mesh));
    if (mesh == NULL)
        return NULL;
    mesh->nelem = 3;

    while (!feof(fp) && (!ptcomplete || (!polycomplete && readpolys))) {
        if (fgets(bufline, 8192, fp) == NULL) {
            if (!feof(fp))
                return NULL;
            else
                return mesh;
        }
        buf = strtok(bufline, " \n");
        while (buf != NULL) {
            switch (state) {
                case READ_HEAD:
                    if (strcmp(buf, "POINTS") == 0) {
                        state = READ_PTNUM;
                    } else if (strcmp(buf, "POLYGONS") == 0) {
                        state = READ_POLYNUM;
                    }
                    break;
                case READ_PTNUM:
                    mesh->npts = (CTMuint) atoi(buf);
                    mesh->pts = malloc(sizeof(CTMfloat)*mesh->npts*3);
                    if (mesh->pts == NULL)
                        return NULL;
                    npts = 0;
                    state = SKIP_PTS;
                    break;
                case READ_POLYNUM:
                    mesh->npolys = (CTMuint) atoi(buf);
                    mesh->polys = malloc(sizeof(CTMuint)*mesh->npolys*3);
                    if (mesh->polys == NULL)
                        return NULL;
                    npts = 0;
                    state = SKIP_POLYS;
                    break;
                case SKIP_PTS:
                    state = READ_PTS;
                    break;
                case SKIP_POLYS:
                    state = READ_POLYS;
                    break;
                case READ_PTS:
                    mesh->pts[npts++] = (CTMfloat) atof(buf);
                    if (npts == mesh->npts*3) {
                        state = READ_HEAD;
                        ptcomplete = true;
                    }
                    break;
                case READ_POLYS:
                    if (ptcount++ % 4)
                        mesh->polys[npts++] = (CTMuint) atoi(buf);
                    if (npts == mesh->npolys*3) {
                        state = READ_HEAD;
                        polycomplete = true;
                    }
                    break;
            }
            buf = strtok(NULL, " \n");
        }
    }
    return mesh;
}
void meshMinMax(Mesh* mesh, float* min, float* max) {
    int i, j;
    for (j = 0; j < mesh->nelem; j++) {
        min[j] = FLT_MAX;
        max[j] = FLT_MIN;
    }
    
    for (i = 0; i < mesh->npts; i++) {
        for (j = 0; j < mesh->nelem; j++) {
            if (mesh->pts[i*mesh->nelem+j] < min[j])
                min[j] = mesh->pts[i*mesh->nelem+j];
            else if (mesh->pts[i*mesh->nelem+j] > max[j])
                max[j] = mesh->pts[i*mesh->nelem+j];
        }
    }
}
void meshResize(Mesh* mesh, CTMuint nelem) {
    int i, j;
    if (nelem > mesh->nelem) {
        mesh->pts = realloc(mesh->pts, mesh->npts*nelem*sizeof(CTMfloat));
        if (mesh->pts == NULL) {
            printf("Error reallocating for mesh resize!\n");
            exit(1);
        }
        for (i=mesh->npts-1; i >= 0; i--) {
            for (j=nelem-1; j >= 0; j--) {
                if (j >= mesh->nelem)
                    mesh->pts[i*nelem+j] = 0;
                else
                    mesh->pts[i*nelem+j] = mesh->pts[mesh->nelem*i+j];
            }
        }
    } else if (nelem < mesh->nelem) {
        printf("Warning: Truncating elements\n");
        for (i=0; i < mesh->npts; i++) {
            for (j=0; j < nelem; j++) {
                mesh->pts[i*nelem+j] = mesh->pts[mesh->nelem*i+j];
            }
        }
        mesh->pts = realloc(mesh->pts, mesh->npts*nelem*sizeof(CTMfloat));
        if (mesh->pts == NULL) {
            printf("Error reallocating for mesh resize!\n");
            exit(1);
        }
    }
    mesh->nelem = nelem;
}
void meshShift(Mesh* mesh, float* add, float* div) {
    int i, j;
    for (i=0; i < mesh->npts; i++)
        for (j=0; j < mesh->nelem; j++)
            mesh->pts[i*mesh->nelem+j] = (mesh->pts[i*mesh->nelem+j]+add[j]) / div[j];
}
void meshNudge(Mesh* mesh, bool right) {
    float min[3], max[3], shift[3] = {0,0,0}, div[3] = {1,1,1};
    meshMinMax(mesh, min, max);
    shift[1] = right ? -min[1] : -max[1];
    meshShift(mesh, shift, div);
}
void meshFree(Mesh* mesh) {
    free(mesh->pts);
    free(mesh->polys);
    free(mesh);
}


Subject* newSubject(const char* name) {
    Subject* subj = calloc(1, sizeof(Subject));
    strcpy(subj->name, name);
    subj->left.fiducial = NULL;
    subj->right.fiducial = NULL;
    subj->left.flat = NULL;
    subj->right.flat = NULL;
    subj->left.datamap = NULL;
    subj->right.datamap = NULL;
    return subj;
}

void hemiAddFid(Hemi* hemi, const char* filename) {
    hemi->fiducial = readVTK(filename, true);
}
void hemiAddFlat(Hemi* hemi, const char* filename) {
    hemi->flat = readVTK(filename, true);
}
void hemiAddSurf(Hemi* hemi, const char* filename, const char* name) {
    unsigned int idx = hemi->nbetween++;
    hemi->between[idx] = readVTK(filename, false);
    if (name == NULL)
        sprintf(hemi->names[idx], "morphTarget%d", idx);
    else
        strcpy(hemi->names[idx], name);
}
void hemiAddMap(Hemi* hemi, const int* datamap) {
    int i;
    hemi->datamap = (CTMfloat*) malloc(hemi->fiducial->npts*sizeof(float)*2);
    for (i=0; i < hemi->fiducial->npts; i++) {
        hemi->datamap[i*2+0] = (CTMfloat) floor(datamap[i] / 256);
        hemi->datamap[i*2+1] = (CTMfloat) (datamap[i] % 256);
    }
}

void subjFree(Subject* subj) {
    int i;
    if (subj->left.fiducial != NULL)
        meshFree(subj->left.fiducial);
    if (subj->right.fiducial != NULL)
        meshFree(subj->right.fiducial);
    if (subj->left.flat != NULL)
        meshFree(subj->left.flat);
    if (subj->right.flat != NULL)
        meshFree(subj->right.flat);

    if (subj->left.datamap != NULL)
        free(subj->left.datamap);
    if (subj->right.datamap != NULL)
        free(subj->right.datamap);

    for (i = 0; i < subj->left.nbetween; i++)
        meshFree(subj->left.between[i]);
    for (i = 0; i < subj->right.nbetween; i++)
        meshFree(subj->right.between[i]);
    free(subj);
}

void saveCTM(Subject* subj, char* leftname, char* rightname, CTMenum compmeth, CTMuint complevel) {
    int i, j;
    Mesh* mesh;
    CTMenum idx, err;
    CTMcontext* ctx[2];
    Hemi* hemis[2];
    char* filenames[2];
    char comment[8192];
    float leftmin[2], leftmax[2];
    float rightmin[2], rightmax[2];
    float flatmin[2], flatmax[2];

    assert(subj->left.fiducial != NULL);
    assert(subj->right.fiducial != NULL);
    assert(subj->left.flat != NULL);
    assert(subj->right.flat != NULL);
    assert(subj->left.nbetween == subj->right.nbetween);

    filenames[0] = leftname;
    filenames[1] = rightname;
    ctx[0] = ctmNewContext(CTM_EXPORT);
    ctx[1] = ctmNewContext(CTM_EXPORT);
    hemis[0] = &(subj->left);
    hemis[1] = &(subj->right);

    meshResize(hemis[0]->flat, 2);
    meshResize(hemis[1]->flat, 2);
    meshNudge(hemis[0]->flat, false);
    meshNudge(hemis[1]->flat, true);
    meshMinMax(hemis[0]->flat, leftmin, leftmax);
    meshMinMax(hemis[1]->flat, rightmin, rightmax);
    for (i = 0; i < 2; i++) {
        flatmin[i] = leftmin[i] < rightmin[i] ? leftmin[i] : rightmin[i];
        flatmax[i] = leftmax[i] > rightmax[i] ? leftmax[i] : rightmax[i];
    }
    for (i = 0; i < 2; i++) {
        flatmax[i] = flatmax[i] - flatmin[i];
        flatmin[i] = - flatmin[i];
    }
    sprintf(comment, "{flat:{min:[%f,%f], max:[%f,%f]}}", 
        flatmin[0], flatmin[1], flatmax[0], flatmax[1]);
    for (i = 0; i < 2; i++) {
        mesh = hemis[i]->fiducial;
        ctmDefineMesh(ctx[i], mesh->pts, mesh->npts, mesh->polys, mesh->npolys, NULL);
        meshShift(hemis[i]->flat, flatmin, flatmax);
        idx = ctmAddUVMap(ctx[i], hemis[i]->flat->pts, "texcoord", NULL);
        if (idx == CTM_NONE)
            printf("CTM error!\n");
        err = ctmGetError(ctx[i]);
        if (err != CTM_NONE)
            printf("CTM error: %s, %d\n", ctmErrorString(err), err);
        for (j = 0; j < hemis[i]->nbetween; j++) {
            mesh = hemis[i]->between[j];
            meshNudge(mesh, i);
            meshResize(mesh, 4);
            assert(mesh->nelem == 4);
            idx = ctmAddAttribMap(ctx[i], mesh->pts, hemis[i]->names[j]);
            if (idx == CTM_NONE)
                printf("CTM error!\n");
            err = ctmGetError(ctx[i]);
            if (err != CTM_NONE)
                printf("CTM error add surface: %s\n", ctmErrorString(err));
        }
        if (hemis[i]->datamap != NULL) {
            idx = ctmAddUVMap(ctx[i], hemis[i]->datamap, "datamap", NULL);
            if (idx == CTM_NONE)
                printf("CTM error!\n");
            err = ctmGetError(ctx[i]);
            if (err != CTM_NONE)
                printf("CTM error add datamap: %s\n", ctmErrorString(err));
        }
        ctmFileComment(ctx[i], comment);
        ctmCompressionMethod(ctx[i], compmeth);
        ctmCompressionLevel(ctx[i], complevel);
        printf("Saving to %s...\n", filenames[i]);
        ctmSave(ctx[i], filenames[i]);
        err = ctmGetError(ctx[i]);
        if (err != CTM_NONE)
            printf("CTM error saving: %s\n", ctmErrorString(err));
        ctmFreeContext(ctx[i]);
    }
}
