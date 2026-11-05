/*
 * attn_tiled.c -- Q3b: cache and register tiling.
 *
 * Placeholder: replace the body of attention() with your implementation.
 * Keep the driver (main) below unchanged. See attn_naive.c for the input and
 * output format.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ------------------------------------------------------------------------- */
/* attention(): write your implementation here.                               */
/* ------------------------------------------------------------------------- */
// Each lane is an independent key/output feature; no horizontal reductions.
typedef float Vec __attribute__((vector_size(64)));
static inline Vec load16(const float *p) { Vec v; memcpy(&v, p, sizeof v); return v; }
static inline void store16(float *p, Vec v) { memcpy(p, &v, sizeof v); }

static float *packed_keys(const float *K, int N, int d, int stride)
{
    float *packed = malloc((size_t)stride*d*sizeof(float));
    if (!packed) { perror("packed keys"); exit(EXIT_FAILURE); }
    for (int j = 0; j < stride; j += 16)
        for (int k = 0; k < d; ++k)
            for (int l = 0; l < 16; ++l)
                packed[(size_t)j*d+16*(size_t)k+l] = j+l < N ? K[(size_t)(j+l)*d+k] : 0;
    return packed;
}

// Register tile: four query rows by sixteen keys, traversing a packed K panel.
static inline void scores(const float *Q, const float *packed, float *S,
                          int rows, int j, int d, int stride, float scale)
{
    const float *q1 = rows > 1 ? Q+d : Q;
    const float *q2 = rows > 2 ? Q+2*(size_t)d : Q;
    const float *q3 = rows > 3 ? Q+3*(size_t)d : Q;
    Vec a = {0}, b = {0}, c = {0}, e = {0};
    const float *p = packed+(size_t)j*d;
    for (int k = 0; k < d; ++k) {
        Vec key = load16(p+16*(size_t)k);
        a += Q[k]*key; b += q1[k]*key;
        c += q2[k]*key; e += q3[k]*key;
    }
    store16(S+j, a*scale);
    if (rows>1) store16(S+stride+j, b*scale);
    if (rows>2) store16(S+2*(size_t)stride+j, c*scale);
    if (rows>3) store16(S+3*(size_t)stride+j, e*scale);
}

static inline float exponentiate(float *row, int N)
{
    float max = row[0];
    for (int j=1;j<N;++j) if (row[j]>max) max=row[j];
    float sum=0;
    for (int j=0;j<N;++j) { row[j]=expf(row[j]-max); sum+=row[j]; }
    return 1.0f/sum;
}

// Fixed widths retain a whole output row in vector registers across key rows.
#define VALUE_KERNEL(D) \
static inline void value##D(const float *row, const float *V, float *out, int N, float inv, int add) { \
    Vec acc[D/16]; \
    _Pragma("clang loop unroll(full)") \
    for(int k=0;k<D/16;++k) acc[k]=(Vec){0}; \
    for(int j=0;j<N;++j) { \
        _Pragma("clang loop unroll(full)") \
        for(int k=0;k<D/16;++k) acc[k]+=row[j]*load16(V+(size_t)j*D+16*k); \
    } \
    _Pragma("clang loop unroll(full)") \
    for(int k=0;k<D/16;++k) store16(out+16*k,acc[k]*inv+(add?load16(out+16*k):(Vec){0})); \
}
VALUE_KERNEL(64)
VALUE_KERNEL(128)
#undef VALUE_KERNEL

static inline void values(const float *row,const float *V,float *out,int N,int d,float inv,int add)
{
    if (d==64) { value64(row,V,out,N,inv,add); return; }
    if (d==128) { value128(row,V,out,N,inv,add); return; }
    // General shape path; apply the normalization only to the new contribution.
    for (int k=0;k<d;++k) {
        float accum=0;
        for (int j=0;j<N;++j) accum+=row[j]*V[(size_t)j*d+k];
        out[k]=(add?out[k]:0)+accum*inv;
    }
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    enum { IB=32, JB=64 };
    int stride = (N+15)/16*16;
    float *packed=packed_keys(K,N,d,stride);
    float *S=malloc((size_t)N*stride*sizeof(float));
    if(!S) { perror("attention scores"); exit(EXIT_FAILURE); }
    float scale=1.0f/sqrtf((float)d);
    for(int ib=0;ib<N;ib+=IB) {
        int ie=ib+IB<N?ib+IB:N;
        for(int jb=0;jb<stride;jb+=JB) {
            int je=jb+JB<stride?jb+JB:stride;
            for(int i=ib;i<ie;i+=4)
                for(int j=jb;j<je;j+=16)
                    scores(Q+(size_t)i*d,packed,S+(size_t)i*stride,ie-i<4?ie-i:4,j,d,stride,scale);
        }
    }
    free(packed);
    for(int i=0;i<N;++i) {
        float *row=S+(size_t)i*stride;
        float inv=exponentiate(row,N);
        for(int j=0;j<N;++j) row[j]*=inv;
    }
    for(int ib=0;ib<N;ib+=IB) {
        int ie=ib+IB<N?ib+IB:N;
        for(int jb=0;jb<N;jb+=JB) {
            int count=N-jb<JB?N-jb:JB;
            for(int i=ib;i<ie;++i)
                values(S+(size_t)i*stride+jb,V+(size_t)jb*d,O+(size_t)i*d,count,d,1.0f,jb!=0);
        }
    }
    free(S);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
