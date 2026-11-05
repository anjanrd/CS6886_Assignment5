/*
 * attn_omp.c -- Q4: your best variant, parallelised with OpenMP.
 *
 * Implementation below; the supplied timing driver is retained.
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


#if defined(__AVX512F__)
#include <immintrin.h>
// exp(x) for shifted logits x <= 0. Reduce to |r| <= ln(2)/2,
// approximate exp(r) by its degree-six Taylor polynomial, then scale by 2^n.
static inline __m512 exp_shifted(__m512 x)
{
    x=_mm512_max_ps(x,_mm512_set1_ps(-104.0f));
    __m512 n=_mm512_roundscale_ps(_mm512_mul_ps(x,_mm512_set1_ps(1.4426950408889634f)),
                                 _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r=_mm512_fnmadd_ps(n,_mm512_set1_ps(0.693145751953125f),x);
    r=_mm512_fnmadd_ps(n,_mm512_set1_ps(1.428606765330187e-6f),r);
    __m512 p=_mm512_fmadd_ps(r,_mm512_set1_ps(1.0f/720),_mm512_set1_ps(1.0f/120));
    p=_mm512_fmadd_ps(r,p,_mm512_set1_ps(1.0f/24));
    p=_mm512_fmadd_ps(r,p,_mm512_set1_ps(1.0f/6));
    p=_mm512_fmadd_ps(r,p,_mm512_set1_ps(0.5f));
    p=_mm512_fmadd_ps(r,p,_mm512_set1_ps(1));
    p=_mm512_fmadd_ps(r,p,_mm512_set1_ps(1));
    return _mm512_scalef_ps(p,n);
}
#endif

static inline float exponentiate(float *row, int N)
{
    int j=0;
    float max=-INFINITY;
#if defined(__AVX512F__)
    __m512 maxima=_mm512_set1_ps(-INFINITY);
    for(;j+15<N;j+=16) maxima=_mm512_max_ps(maxima,_mm512_loadu_ps(row+j));
    max=_mm512_reduce_max_ps(maxima);
#endif
    for(;j<N;++j) if(row[j]>max) max=row[j];
    float sum=0;
    j=0;
#if defined(__AVX512F__)
    __m512 sums=_mm512_setzero_ps(), offset=_mm512_set1_ps(max);
    for(;j+15<N;j+=16) {
        __m512 e=exp_shifted(_mm512_sub_ps(_mm512_loadu_ps(row+j),offset));
        sums=_mm512_add_ps(sums,e);
        _mm512_storeu_ps(row+j,e);
    }
    sum=_mm512_reduce_add_ps(sums);
#endif
    for(;j<N;++j) { row[j]=expf(row[j]-max); sum+=row[j]; }
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

#ifndef QUERY_BLOCK
#define QUERY_BLOCK 32
#endif

// Reuse one loaded V vector across 4 independent query rows.
static inline void group64(const float *S,const float *V,float *O,
                             const float *inv,int N,int stride) {
    Vec a0[4]={0};
    Vec a1[4]={0};
    Vec a2[4]={0};
    Vec a3[4]={0};
    for(int j=0;j<N;++j) {
        float w0=S[(size_t)0*stride+j];
        float w1=S[(size_t)1*stride+j];
        float w2=S[(size_t)2*stride+j];
        float w3=S[(size_t)3*stride+j];
        #pragma clang loop unroll(full)
        for(int k=0;k<4;++k) {
            Vec value=load16(V+(size_t)j*64+16*k);
            a0[k]+=w0*value; a1[k]+=w1*value; a2[k]+=w2*value; a3[k]+=w3*value;
        }
    }
    #pragma clang loop unroll(full)
    for(int k=0;k<4;++k) {
        store16(O+0+16*k,a0[k]*inv[0]);
    store16(O+64+16*k,a1[k]*inv[1]);
    store16(O+128+16*k,a2[k]*inv[2]);
    store16(O+192+16*k,a3[k]*inv[3]);
    }
}

// Reuse one loaded V vector across 2 independent query rows.
static inline void group128(const float *S,const float *V,float *O,
                             const float *inv,int N,int stride) {
    Vec a0[8]={0};
    Vec a1[8]={0};
    for(int j=0;j<N;++j) {
        float w0=S[(size_t)0*stride+j];
        float w1=S[(size_t)1*stride+j];
        #pragma clang loop unroll(full)
        for(int k=0;k<8;++k) {
            Vec value=load16(V+(size_t)j*128+16*k);
            a0[k]+=w0*value; a1[k]+=w1*value;
        }
    }
    #pragma clang loop unroll(full)
    for(int k=0;k<8;++k) {
        store16(O+0+16*k,a0[k]*inv[0]);
    store16(O+128+16*k,a1[k]*inv[1]);
    }
}
static inline void query_block(const float *Q, const float *packed, const float *V, float *O,
                               float *S, int rows, int N, int d, int stride, float scale)
{
    for (int j=0;j<stride;j+=16)
        for (int i=0;i<rows;i+=4)
            scores(Q+(size_t)i*d,packed,S+(size_t)i*stride,rows-i<4?rows-i:4,j,d,stride,scale);
    float inv[QUERY_BLOCK];
    for(int i=0;i<rows;++i) inv[i]=exponentiate(S+(size_t)i*stride,N);
    int i=0;
    if(d==64)
        for(;i+4<=rows;i+=4) group64(S+(size_t)i*stride,V,O+(size_t)i*d,inv+i,N,stride);
    if(d==128)
        for(;i+2<=rows;i+=2) group128(S+(size_t)i*stride,V,O+(size_t)i*d,inv+i,N,stride);
    for(;i<rows;++i) values(S+(size_t)i*stride,V,O+(size_t)i*d,N,d,inv[i],0);
}

void attention(const float *Q, const float *K, const float *V, float *O, int N, int d)
{
    int stride=(N+15)/16*16;
    float scale=1.0f/sqrtf((float)d);
    float *packed=packed_keys(K,N,d,stride);
    #pragma omp parallel
    {
        float *scratch=malloc((size_t)QUERY_BLOCK*stride*sizeof(float));
        if(!scratch) { perror("score block"); exit(EXIT_FAILURE); }
        #pragma omp for schedule(static)
        for(int i=0;i<N;i+=QUERY_BLOCK)
            query_block(Q+(size_t)i*d,packed,V,O+(size_t)i*d,scratch,
                        N-i<QUERY_BLOCK?N-i:QUERY_BLOCK,N,d,stride,scale);
        free(scratch);
    }
    free(packed);
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
