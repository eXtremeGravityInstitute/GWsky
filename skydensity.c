/* gcc -I/opt/homebrew/include -L/opt/homebrew/lib -o skydensity skydensity.c -lgsl -lchealpix -lm */

/* Standard Includes */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <chealpix.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_vector.h>

#define PIn 3.1415926535897931159979634685442

double *double_vector(long N);
void free_double_vector(double *v);
int *int_vector(long N);
void free_int_vector(int *v);
int is_power_of_two(long n);
double clamp_double(double x, double xmin, double xmax);
void write_map(const char *filename, double *map, long Npix);
long make_credible_boundary(long Nside, long Npix, double *map, gsl_permutation *perm, long k90, double level90, double *boundary, double *region);
void nested_neighbors(long nside, long ipnest, long neighbors[8]);
void nest2xyf_local(long nside, long ipnest, long *ix, long *iy, long *face);
long xyf2nest_local(long nside, long ix, long iy, long face);
long edge_neighbor_nest(long nside, long ix, long iy, long face, int dir);
long mod4(long x);

int main(int argc, char* argv[])
{

  long i, k, Nside, Npix, Nsample, k90, boundary_count;
  double theta, phi;
  double costh, x, y, level90;
  double logL, DL;
  double *map, *boundary, *region;
  char line[4096];
    
  FILE *out;
  FILE *in;
    
  if(argc != 3) { printf("./skydensity file Nside\n"); return 0;}
    

    in = fopen(argv[1],"r");
    if(in == NULL)
    {
        fprintf(stderr, "Could not open chain file %s\n", argv[1]);
        return 1;
    }
    
    Nsample = 0;
    while (fgets(line, sizeof(line), in) != NULL)
    {
        if(sscanf(line,"%ld%lf%lf%lf%lf", &k, &logL, &phi, &costh, &DL) == 5) Nsample++;
    }
    rewind(in);

    if(Nsample <= 0)
    {
        fprintf(stderr, "No sky samples found in %s\n", argv[1]);
        fclose(in);
        return 1;
    }
    
    
    // Nside must be a power of 2
    Nside = atoi(argv[2]);
    if(!is_power_of_two(Nside))
    {
        fprintf(stderr, "Nside must be a power of 2 for nested-neighbor boundary finding\n");
        fclose(in);
        return 1;
    }

    Npix = 12*Nside*Nside;
    
    map = double_vector(Npix);
    boundary = double_vector(Npix);
    region = double_vector(Npix);

    
    for(i=0; i< Npix; i++)
    {
        map[i] = 0.0;
        boundary[i] = 0.0;
        region[i] = 0.0;
    }
    
    out = fopen("skycheck.dat","w");
    if(out == NULL)
    {
        fprintf(stderr, "Could not open skycheck.dat for writing\n");
        fclose(in);
        free_double_vector(map);
        free_double_vector(boundary);
        free_double_vector(region);
        return 1;
    }
    i = 0;
    while (fgets(line, sizeof(line), in) != NULL && i < Nsample)
    {
        if(sscanf(line,"%ld%lf%lf%lf%lf", &k, &logL, &phi, &costh, &DL) != 5) continue;
        theta = acos(clamp_double(costh, -1.0, 1.0));
        if(phi < 0.0) phi += 2.0*PIn;
        if(phi > 2.0*PIn) phi -= 2.0*PIn;
        fprintf(out,"%ld %f %f\n", i, phi, theta);
        ang2pix_ring(Nside, theta, phi, &k);
        //printf("%ld %ld %f %f\n", i, k, theta, phi);
        map[k] += 1.0;
        i++;
    }
    fclose(in);
    fclose(out);
    
    for(i=0; i< Npix; i++) map[i] /= (double)(Nsample);
    
    gsl_vector *mv = gsl_vector_alloc (Npix);
    
    for(i=0; i< Npix; i++) gsl_vector_set(mv, i, map[i]);
    
    gsl_permutation *perm = gsl_permutation_alloc(Npix);
    
    gsl_sort_vector_index(perm, mv);
    
    k = 0;
    x = 0.0;
    do
    {
        k++;
        x += gsl_vector_get(mv,perm->data[Npix-k]);
    }while(x < 0.5);
    
    y = (double)(k)/(double)(Npix)*4.0*PIn*(180.0/PIn)*(180.0/PIn);
    printf("50 percent credible interval = %f  square degrees\n", y);
    
    do
    {
        k++;
        x += gsl_vector_get(mv,perm->data[Npix-k]);
    }while(x < 0.9);

    k90 = k;
    if(k90 < Npix) level90 = gsl_vector_get(mv,perm->data[Npix-k90-1]);
    else level90 = 0.0;
    
    y = (double)(k)/(double)(Npix)*4.0*PIn*(180.0/PIn)*(180.0/PIn);
    printf("90 percent credible interval = %f  square degrees\n", y);

    boundary_count = make_credible_boundary(Nside, Npix, map, perm, k90, level90, boundary, region);
    y = (double)(boundary_count)/(double)(Npix)*4.0*PIn*(180.0/PIn)*(180.0/PIn);
    printf("90 percent credible boundary = %ld pixels = %f square degrees\n", boundary_count, y);
    
    pix2ang_ring(Nside, perm->data[Npix-1], &theta, &phi);
    
    printf("MAP location theta = %f phi = %f\n", theta*180.0/PIn, phi*180.0/PIn);
    
    gsl_permutation_free(perm);
    gsl_vector_free(mv);
    
  write_map("sky.dat", map, Npix);
  write_map("sky90_region.dat", region, Npix);
  write_map("sky90_boundary.dat", boundary, Npix);
    
  free_double_vector(map);
  free_double_vector(boundary);
  free_double_vector(region);

  return 0;

}


double *double_vector(long N)
{
    return malloc( (N+1) * sizeof(double) );
}

void free_double_vector(double *v)
{
    free(v);
}

int *int_vector(long N)
{
    return malloc( (N+1) * sizeof(int) );
}

void free_int_vector(int *v)
{
    free(v);
}


int is_power_of_two(long n)
{
    return (n > 0 && (n & (n-1)) == 0);
}

double clamp_double(double x, double xmin, double xmax)
{
    if(x < xmin) return xmin;
    if(x > xmax) return xmax;
    return x;
}


void write_map(const char *filename, double *map, long Npix)
{
    long i;
    FILE *out;

    out = fopen(filename, "w");
    if(out == NULL)
    {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        exit(1);
    }

    for(i=0; i< Npix; i++) fprintf(out,"%f\n", map[i]);
    fclose(out);
}


long make_credible_boundary(long Nside, long Npix, double *map, gsl_permutation *perm, long k90, double level90, double *boundary, double *region)
{
    long i, j, iring, inest, nneighbor, count;
    long neighbors[8];
    double *map_nest;
    double *boundary_nest;
    int *inside90;

    map_nest = double_vector(Npix);
    boundary_nest = double_vector(Npix);
    inside90 = int_vector(Npix);

    for(i=0; i<Npix; i++)
    {
        ring2nest(Nside, i, &inest);
        map_nest[inest] = map[i];
        boundary_nest[inest] = 0.0;
        inside90[inest] = 0;
        boundary[i] = 0.0;
        region[i] = 0.0;
    }

    for(j=0; j<k90; j++)
    {
        iring = (long)perm->data[Npix-1-j];
        ring2nest(Nside, iring, &inest);
        inside90[inest] = 1;
    }

    count = 0;
    for(inest=0; inest<Npix; inest++)
    {
        if(!inside90[inest]) continue;

        nested_neighbors(Nside, inest, neighbors);

        for(j=0; j<8; j++)
        {
            nneighbor = neighbors[j];
            if(nneighbor >= 0 && nneighbor < Npix && !inside90[nneighbor] && map_nest[nneighbor] <= level90 && boundary_nest[nneighbor] == 0.0)
            {
                boundary_nest[nneighbor] = 1.0;
                count++;
            }
        }
    }

    for(iring=0; iring<Npix; iring++)
    {
        ring2nest(Nside, iring, &inest);
        boundary[iring] = boundary_nest[inest];
        region[iring] = (double)inside90[inest];
    }

    free_double_vector(map_nest);
    free_double_vector(boundary_nest);
    free_int_vector(inside90);

    return count;
}


void nested_neighbors(long nside, long ipnest, long neighbors[8])
{
    static const int dx[8] = {-1, -1,  0,  1,  1,  1,  0, -1};
    static const int dy[8] = { 0,  1,  1,  1,  0, -1, -1, -1};
    long ix, iy, face, ixn, iyn;
    int i;

    nest2xyf_local(nside, ipnest, &ix, &iy, &face);

    for(i=0; i<8; i++)
    {
        ixn = ix + dx[i];
        iyn = iy + dy[i];
        if(ixn >= 0 && ixn < nside && iyn >= 0 && iyn < nside)
        {
            neighbors[i] = xyf2nest_local(nside, ixn, iyn, face);
        }
        else
        {
            neighbors[i] = edge_neighbor_nest(nside, ix, iy, face, i);
        }
    }
}


void nest2xyf_local(long nside, long ipnest, long *ix, long *iy, long *face)
{
    long ipf, bit, shift;

    *face = ipnest/(nside*nside);
    ipf = ipnest - (*face)*nside*nside;
    *ix = 0;
    *iy = 0;

    for(bit=1, shift=0; bit<nside; bit <<= 1, shift++)
    {
        if(ipf & (1L << (2*shift))) *ix |= bit;
        if(ipf & (1L << (2*shift+1))) *iy |= bit;
    }
}


long xyf2nest_local(long nside, long ix, long iy, long face)
{
    long ipf, bit, shift;

    ipf = 0;
    for(bit=1, shift=0; bit<nside; bit <<= 1, shift++)
    {
        if(ix & bit) ipf |= (1L << (2*shift));
        if(iy & bit) ipf |= (1L << (2*shift+1));
    }

    return face*nside*nside + ipf;
}


long edge_neighbor_nest(long nside, long ix, long iy, long face, int dir)
{
    long N, r, face2, ix2, iy2;

    N = nside-1;
    face2 = -1;
    ix2 = -1;
    iy2 = -1;

    if(face < 4)
    {
        r = face;
        switch(dir)
        {
            case 0: face2 = face+4;       ix2 = N;    iy2 = iy;   break; // SW
            case 1:
                if(ix == 0 && iy == N) return -1;
                if(ix == 0) { face2 = face+4;       ix2 = N;    iy2 = iy+1; }
                else        { face2 = mod4(r+3);   ix2 = N;    iy2 = ix-1; }
                break; // W
            case 2: face2 = mod4(r+3);   ix2 = N;    iy2 = ix;   break; // NW
            case 3:
                if(ix == N && iy == N) { face2 = mod4(r+2); ix2 = N;    iy2 = N;    }
                else if(iy == N)       { face2 = mod4(r+3); ix2 = N;    iy2 = ix+1; }
                else                   { face2 = mod4(r+1); ix2 = iy+1; iy2 = N;    }
                break; // N
            case 4: face2 = mod4(r+1);   ix2 = iy;   iy2 = N;    break; // NE
            case 5:
                if(ix == N && iy == 0) return -1;
                if(ix == N) { face2 = mod4(r+1);   ix2 = iy-1; iy2 = N; }
                else        { face2 = 4+mod4(r+1); ix2 = ix+1; iy2 = N; }
                break; // E
            case 6: face2 = 4+mod4(r+1); ix2 = ix;   iy2 = N;    break; // SE
            case 7:
                if(ix == 0 && iy == 0) { face2 = face+8;      ix2 = N;    iy2 = N;    }
                else if(ix == 0)       { face2 = face+4;      ix2 = N;    iy2 = iy-1; }
                else                   { face2 = 4+mod4(r+1); ix2 = ix-1; iy2 = N;    }
                break; // S
        }
    }
    else if(face < 8)
    {
        r = face-4;
        switch(dir)
        {
            case 0: face2 = 8+mod4(r+3);  ix2 = N;    iy2 = iy;   break; // SW
            case 1:
                if(ix == 0 && iy == N) { face2 = 4+mod4(r+3); ix2 = N;    iy2 = 0;    }
                else if(ix == 0)       { face2 = 8+mod4(r+3); ix2 = N;    iy2 = iy+1; }
                else                   { face2 = mod4(r+3);   ix2 = ix-1; iy2 = 0;    }
                break; // W
            case 2: face2 = mod4(r+3);    ix2 = ix;   iy2 = 0;    break; // NW
            case 3:
                if(ix == N && iy == N) return -1;
                if(iy == N) { face2 = mod4(r+3); ix2 = ix+1; iy2 = 0;    }
                else        { face2 = r;         ix2 = 0;    iy2 = iy+1; }
                break; // N
            case 4: face2 = r;            ix2 = 0;    iy2 = iy;   break; // NE
            case 5:
                if(ix == N && iy == 0) { face2 = 4+mod4(r+1); ix2 = 0;    iy2 = N;    }
                else if(ix == N)       { face2 = r;           ix2 = 0;    iy2 = iy-1; }
                else                   { face2 = 8+r;         ix2 = ix+1; iy2 = N;    }
                break; // E
            case 6: face2 = 8+r;          ix2 = ix;   iy2 = N;    break; // SE
            case 7:
                if(ix == 0 && iy == 0) return -1;
                if(ix == 0) { face2 = 8+mod4(r+3); ix2 = N;    iy2 = iy-1; }
                else        { face2 = 8+r;         ix2 = ix-1; iy2 = N;    }
                break; // S
        }
    }
    else
    {
        r = face-8;
        switch(dir)
        {
            case 0: face2 = 8+mod4(r+3);  ix2 = iy;   iy2 = 0;    break; // SW
            case 1:
                if(ix == 0 && iy == N) return -1;
                if(ix == 0) { face2 = 8+mod4(r+3); ix2 = iy+1; iy2 = 0; }
                else        { face2 = 4+r;         ix2 = ix-1; iy2 = 0; }
                break; // W
            case 2: face2 = 4+r;          ix2 = ix;   iy2 = 0;    break; // NW
            case 3:
                if(ix == N && iy == N) { face2 = r;             ix2 = 0;    iy2 = 0;    }
                else if(iy == N)       { face2 = 4+r;           ix2 = ix+1; iy2 = 0;    }
                else                   { face2 = 4+mod4(r+1);   ix2 = 0;    iy2 = iy+1; }
                break; // N
            case 4: face2 = 4+mod4(r+1);  ix2 = 0;    iy2 = iy;   break; // NE
            case 5:
                if(ix == N && iy == 0) return -1;
                if(ix == N) { face2 = 4+mod4(r+1); ix2 = 0;    iy2 = iy-1; }
                else        { face2 = 8+mod4(r+1); ix2 = 0;    iy2 = ix+1; }
                break; // E
            case 6: face2 = 8+mod4(r+1);  ix2 = 0;    iy2 = ix;   break; // SE
            case 7:
                if(ix == 0 && iy == 0) { face2 = 8+mod4(r+2); ix2 = 0;    iy2 = 0;    }
                else if(ix == 0)       { face2 = 8+mod4(r+3); ix2 = iy-1; iy2 = 0;    }
                else                   { face2 = 8+mod4(r+1); ix2 = 0;    iy2 = ix-1; }
                break; // S
        }
    }

    if(face2 < 0) return -1;
    return xyf2nest_local(nside, ix2, iy2, face2);
}


long mod4(long x)
{
    x %= 4;
    if(x < 0) x += 4;
    return x;
}
