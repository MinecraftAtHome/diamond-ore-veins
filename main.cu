#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include <math.h>



#define __STDC_FORMAT_MACROS 1

#include <stdlib.h>
#include <stddef.h>
#include <inttypes.h>

#ifdef BOINC
  #include "boinc_api.h"
#if defined _WIN32 || defined _WIN64
  #include "boinc_win.h"
#endif
#endif

#define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
    exit(code);
  }
}

///=============================================================================
///                      Compiler and Platform Features
///=============================================================================

typedef int8_t      i8;
typedef uint8_t     u8;
typedef int16_t     i16;
typedef uint16_t    u16;
typedef int32_t     i32;
typedef uint32_t    u32;
typedef int64_t     i64;
typedef uint64_t    u64;
typedef float       f32;
typedef double      f64;


#define STRUCT(S) typedef struct S S; struct S

#if __GNUC__

#define IABS(X)                 __builtin_abs(X)
#define PREFETCH(PTR,RW,LOC)    __builtin_prefetch(PTR,RW,LOC)
#define likely(COND)            (__builtin_expect(!!(COND),1))
#define unlikely(COND)          (__builtin_expect((COND),0))
#define ATTR(...)               __attribute__((__VA_ARGS__))
#define BSWAP32(X)              __builtin_bswap32(X)
#define UNREACHABLE()           __builtin_unreachable()

#else

#define IABS(X)                 ((int)abs(X))
#define PREFETCH(PTR,RW,LOC)
#define likely(COND)            (COND)
#define unlikely(COND)          (COND)
#define ATTR(...)
__device__ __host__ static inline uint32_t BSWAP32(uint32_t x) {
    x = ((x & 0x000000ff) << 24) | ((x & 0x0000ff00) <<  8) |
        ((x & 0x00ff0000) >>  8) | ((x & 0xff000000) >> 24);
    return x;
}
#if _MSC_VER
#define UNREACHABLE()           
#else
#define UNREACHABLE()           exit(1) // [[noreturn]]
#endif

#endif

/// imitate amd64/x64 rotate instructions

__device__ __host__ static inline ATTR(const, always_inline, artificial)
uint64_t rotl64(uint64_t x, uint8_t b)
{
    return (x << b) | (x >> (64-b));
}

__device__ __host__ static inline ATTR(const, always_inline, artificial)
uint32_t rotr32(uint32_t a, uint8_t b)
{
    return (a >> b) | (a << (32-b));
}

/// integer floor divide
__device__ __host__ static inline ATTR(const, always_inline)
int32_t floordiv(int32_t a, int32_t b)
{
    int32_t q = a / b;
    int32_t r = a % b;
    return q - ((a ^ b) < 0 && !!r);
}

///=============================================================================
///                    C implementation of Java Random
///=============================================================================

__device__ __host__ static inline void setSeed(uint64_t *seed, uint64_t value)
{
    *seed = (value ^ 0x5deece66d) & ((1ULL << 48) - 1);
}

__device__ __host__ static inline int next(uint64_t *seed, const int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int) ((int64_t)*seed >> (48 - bits));
}

__device__ __host__ static inline int nextInt(uint64_t *seed, const int n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (uint64_t)next(seed, 31);
        return (int) ((int64_t) x >> 31);
    }

    do {
        bits = next(seed, 31);
        val = bits % n;
    }
    while (bits - val + m < 0);
    return val;
}

__device__ __host__ static inline uint64_t nextLong(uint64_t *seed)
{
    return ((uint64_t) next(seed, 32) << 32) + next(seed, 32);
}

__device__ __host__ static inline float nextFloat(uint64_t *seed)
{
    return next(seed, 24) / (float) (1 << 24);
}

__device__ __host__ static inline double nextDouble(uint64_t *seed)
{
    uint64_t x = (uint64_t)next(seed, 26);
    x <<= 27;
    x += next(seed, 27);
    return (int64_t) x / (double) (1ULL << 53);
}

/* A macro to generate the ideal assembly for X = nextInt(*S, 24)
 * This is a macro and not an inline function, as many compilers can make use
 * of the additional optimisation passes for the surrounding code.
 */
#define JAVA_NEXT_INT24(S,X)                \
    do {                                    \
        uint64_t a = (1ULL << 48) - 1;      \
        uint64_t c = 0x5deece66dULL * (S);  \
        c += 11; a &= c;                    \
        (S) = a;                            \
        a = (uint64_t) ((int64_t)a >> 17);  \
        c = 0xaaaaaaab * a;                 \
        c = (uint64_t) ((int64_t)c >> 36);  \
        (X) = (int)a - (int)(c << 3) * 3;   \
    } while (0)


/* Jumps forwards in the random number sequence by simulating 'n' calls to next.
 */
__device__ __host__ static inline void skipNextN(uint64_t *seed, uint64_t n)
{
    uint64_t m = 1;
    uint64_t a = 0;
    uint64_t im = 0x5deece66dULL;
    uint64_t ia = 0xb;
    uint64_t k;

    for (k = n; k; k >>= 1)
    {
        if (k & 1)
        {
            m *= im;
            a = im * a + ia;
        }
        ia = (im + 1) * ia;
        im *= im;
    }

    *seed = *seed * m + a;
    *seed &= 0xffffffffffffULL;
}


///=============================================================================
///                               Xoroshiro 128
///=============================================================================

STRUCT(Xoroshiro)
{
    uint64_t lo, hi;
};

__device__ __host__ static inline void xSetSeed(Xoroshiro *xr, uint64_t value)
{
    const uint64_t XL = 0x9e3779b97f4a7c15ULL;
    const uint64_t XH = 0x6a09e667f3bcc909ULL;
    const uint64_t A = 0xbf58476d1ce4e5b9ULL;
    const uint64_t B = 0x94d049bb133111ebULL;
    uint64_t l = value ^ XH;
    uint64_t h = l + XL;
    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;
    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;
    l = l ^ (l >> 31);
    h = h ^ (h >> 31);
    xr->lo = l;
    xr->hi = h;
}

__device__ __host__ static inline uint64_t xNextLong(Xoroshiro *xr)
{
    uint64_t l = xr->lo;
    uint64_t h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

__device__ __host__ static inline int xNextInt(Xoroshiro *xr, uint32_t n)
{
    uint64_t r = (xNextLong(xr) & 0xFFFFFFFF) * n;
    if ((uint32_t)r < n)
    {
        while ((uint32_t)r < (~n + 1) % n)
        {
            r = (xNextLong(xr) & 0xFFFFFFFF) * n;
        }
    }
    return r >> 32;
}

__device__ __host__ static inline double xNextDouble(Xoroshiro *xr)
{
    return (xNextLong(xr) >> (64-53)) * 1.1102230246251565E-16;
}

__device__ __host__ static inline float xNextFloat(Xoroshiro *xr)
{
    return (xNextLong(xr) >> (64-24)) * 5.9604645E-8F;
}

__device__ __host__ static inline void xSkipN(Xoroshiro *xr, int count)
{
    while (count --> 0)
        xNextLong(xr);
}

__device__ __host__ static inline uint64_t xNextLongJ(Xoroshiro *xr)
{
    int32_t a = xNextLong(xr) >> 32;
    int32_t b = xNextLong(xr) >> 32;
    return ((uint64_t)a << 32) + b;
}

__device__ __host__ static inline int xNextIntJ(Xoroshiro *xr, uint32_t n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (xNextLong(xr) >> 33);
        return (int) ((int64_t) x >> 31);
    }

    do {
        bits = (xNextLong(xr) >> 33);
        val = bits % n;
    }
    while (bits - val + m < 0);
    return val;
}

typedef struct {
    Xoroshiro internal;
    int num_calls;
} RNG; 

#define XRSR_MIX1          0xbf58476d1ce4e5b9
#define XRSR_MIX2          0x94d049bb133111eb
#define XRSR_MIX1_INVERSE  0x96de1b173f119089
#define XRSR_MIX2_INVERSE  0x319642b2d24d8ec3
#define XRSR_SILVER_RATIO  0x6a09e667f3bcc909
#define XRSR_GOLDEN_RATIO  0x9e3779b97f4a7c15

__device__ __host__   uint64_t mix64(uint64_t a) {
	a = (a ^ a >> 30) * XRSR_MIX1;
	a = (a ^ a >> 27) * XRSR_MIX2;
	return a ^ a >> 31;
}

__device__ __host__  RNG rng_new() {
    return {{0}};
}

__device__ __host__  static void rng_set_seed(RNG *rng, uint64_t seed) {
    seed ^= XRSR_SILVER_RATIO;
    rng->internal.lo = mix64(seed);
    rng->internal.hi = mix64(seed + XRSR_GOLDEN_RATIO);
}

__device__ __host__  static void rng_set_internal(RNG *rng, uint64_t lo, uint64_t hi) {
    rng->internal.lo = lo;
    rng->internal.hi = hi;
}

__device__ __host__  static uint64_t rng_next(RNG *rng, int32_t bits) {
    rng->num_calls++;
    return xNextLong(&rng->internal) >> (64 - bits);
}

__device__ __host__  static int32_t rng_next_int(RNG *rng, uint32_t bound) {
    uint32_t r = rng_next(rng, 31);
    uint32_t m = bound - 1;
    if ((bound & m) == 0) {
        // (int)((long)p_188504_ * (long)this.next(31) >> 31);
        r = (uint32_t)((uint64_t)bound * (uint64_t)r >> 31);
    }
    else {
        for (uint32_t u = r; (int32_t)(u - (r = u % bound) + m) < 0; u = rng_next(rng, 31));
    }
    return r;
}

__device__ __host__  static float rng_next_float(RNG *rng) {
    return xNextFloat(&rng->internal);
}

__device__ __host__  static double rng_next_double(RNG *rng) { // whoops!
    int32_t i = rng_next(rng, 26);
    int32_t j = rng_next(rng, 27);
    uint64_t k = ((uint64_t)i << 27) + (uint64_t)j;
    return (double)k * (double)1.110223E-16F;
}

__device__ __host__  static int rng_next_between_inclusive(RNG *rng, int i, int j) {
    return rng_next_int(rng, j - i + 1) + i;
}

__device__ __host__  static uint64_t rng_next_long(RNG *rng) {
    int32_t i = rng_next(rng, 32);
    int32_t j = rng_next(rng, 32);
    uint64_t k = (uint64_t)i << 32;
    return k + (uint64_t)j;
}

__device__ __host__  static uint64_t rng_set_feature_seed(RNG *rng, uint64_t p_190065_, int32_t p_190066_, int32_t p_190067_) {
    uint64_t i = p_190065_ + (uint64_t)p_190066_ + (uint64_t)(10000 * p_190067_);
    //printf("Salt = %" PRIu64 "\n", (uint64_t)p_190066_ + (uint64_t)(10000 * p_190067_));
    rng_set_seed(rng, i);
    return i;
}

__device__ __host__  uint64_t reverse_decoration_seed(uint64_t decorator_seed, int index, int step) {
    return decorator_seed - (uint64_t)index - 10000L * (uint64_t)step;
}

__device__ __host__  static uint64_t rng_set_decoration_seed(RNG *rng, uint64_t world_seed, int32_t x, int32_t z) {
    rng_set_seed(rng, world_seed);

    uint64_t a = rng_next_long(rng) | 1L;
    uint64_t b = rng_next_long(rng) | 1L;

    // printf("the k to recover = %" PRIu64 "\n", (a * (uint64_t)x + b * (uint64_t)z));
    uint64_t k = (a * (uint64_t)x + b * (uint64_t)z) ^ world_seed;
    // printf("real k = %" PRIu64 "\n", k);
    // printf("invert k = %" PRIu64 "\n", k ^ world_seed);
    rng_set_seed(rng, k);
    return k;
}

typedef struct /*__align__(16) i think this gets auto-aligned*/ {
    int dx, dz, height;
    bool is_valid;
} Offset;

__device__ __host__ Offset offset_new(int dx, int dz, int height) {
    return {dx, dz, height, true};
}

__device__ __host__ Offset offset_invalid_new() {
    return {-1, -1, -1, false};
}

__device__ __host__ Offset get_position_standard(RNG *rng) {
    int dx = rng_next_int(rng, 16); // spread
    int dz = rng_next_int(rng, 16);

    int i = -144;
    int j = 16;
    int plateau = 0;

    int l = ((j-i) - plateau) / 2;
    int i1 = (j-i) - l;
    int height = i + rng_next_between_inclusive(rng, 0, i1) + rng_next_between_inclusive(rng, 0, l);

    return offset_new(dx, dz, height);
}

__device__ __host__  Offset get_small_diamond_position(RNG *rng, uint64_t chunk_seed) {
    // uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 18, 6);
    // (void)feature_seed;
    
    return get_position_standard(rng);
}

__device__ __host__  Offset get_medium_diamond_position(RNG *rng, uint64_t chunk_seed) {
    // uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 19, 6);
    // (void)feature_seed;
    
    int dx = rng_next_int(rng, 16);
    int dz = rng_next_int(rng, 16);

    int i = -64;
    int j = -4;

    int height = rng_next_between_inclusive(rng, i, j);

    return offset_new(dx, dz, height);
}

__device__ __host__ Offset get_large_diamond_position(RNG *rng, uint64_t chunk_seed) {
    uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 20, 6);
    (void)feature_seed;

    if (!(rng_next_float(rng) < 1.0F / (float)9.0)) {
        return offset_invalid_new();
    }

    return get_position_standard(rng);
}

__device__ __host__  Offset get_buried_diamond_position(RNG *rng, uint64_t chunk_seed) {
    // uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 21, 6);
    // (void)feature_seed;

    return get_position_standard(rng);
}



__device__ __host__  float offset_distance_squared(const Offset *a, const Offset *b) {
    int x1 = a->dx;
    int y1 = a->height;
    int z1 = a->dz;

    int x2 = b->dx;
    int y2 = b->height;
    int z2 = b->dz;

    return ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
}

#define PI 3.14159265358979

__device__ __host__  bool in_range(int y) {
    return (y > -55) && (y < -6);
}

__device__ __host__  bool get_small_diamond_offsets(RNG *rng, uint64_t chunk_seed, Offset *offsets, size_t *sz) {
    uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 18, 6);
    Offset o = get_small_diamond_position(rng, chunk_seed);
    offsets[*sz] = o;
    (*sz)++;
    return in_range(o.height);
}

__device__ __host__  bool get_medium_diamond_offsets(RNG *rng, uint64_t chunk_seed, Offset *offsets, size_t *sz) {
    uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 19, 6);
    Offset o = get_medium_diamond_position(rng, chunk_seed);
    offsets[*sz] = o;
    (*sz)++;
    return in_range(o.height);
}

__device__ __host__  bool get_buried_diamond_offsets(RNG *rng, uint64_t chunk_seed, Offset *offsets, size_t *sz) {
    uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 21, 6);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        Offset o = get_buried_diamond_position(rng, chunk_seed);
        
        if (!in_range(o.height)) {
            return false;
        }

        // rng->num_calls = 0;
        offsets[*sz] = o;
        (*sz)++;
        // advance_rng(&rng, o.dx, o.height, o.dz, buried_diamond_size, buried_diamond_discard); 
        // rng->num_calls = 0;
        rng_next_float(rng);
        rng_next_int(rng, 3);
        rng_next_int(rng, 3);

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            rng_next_double(rng);
        }
    }
    return true;
}

__global__ void kernel(uint64_t s, uint64_t *out) {
    uint64_t chunk_seed = blockDim.x * blockIdx.x + threadIdx.x + s;

    RNG rng = rng_new();
    Offset large = get_large_diamond_position(&rng, chunk_seed);
    
    if (!large.is_valid) {
        return;
    }

    Offset offsets[15] = {0};
    size_t sz = 1;
    offsets[0] = large;

    if (!get_small_diamond_offsets(&rng, chunk_seed, offsets, &sz)) {
        return;
    }
    if (!get_medium_diamond_offsets(&rng, chunk_seed, offsets, &sz)) {
        return;
    }
    if (!get_buried_diamond_offsets(&rng, chunk_seed, offsets, &sz)) {
        return;
    }

    const Offset *cmp = &offsets[0];
    #pragma unroll
    for (int i = 1; i < sz; i++) {
        if (offset_distance_squared(cmp, (const Offset *)&offsets[i]) > 9.0) {
            return;
        }
    }
    out[blockDim.x * blockIdx.x + threadIdx.x] = chunk_seed;
}

#include <time.h>
#include <chrono>
using namespace std::chrono;

#ifdef __GNUC__

#include <unistd.h>
#include <sys/time.h>

#endif

/*
    You can add anything you want to checkpoint_vars.
    Be sure to update the checkpointing sections below to reflect the new item in the struct (to save the data into the struct and then to disk)
*/
struct checkpoint_vars {
    unsigned long long offset;
    uint64_t elapsed_chkpoint;
};

uint64_t elapsed_chkpoint = 0;

int main(int argc, char **argv) {

    /*
        The way this has been written, each loop, it calls 32768 * 32 (1048576) kernel threads that each individually run a single seed.
        We refer to these loops as "blocks" of seeds in this code.
        --start defines the starting block (--start 0 begins at seed 0, --start 1 begins at seed 1048576, --start 2 begins at 2097152)
        --end defines the ending block (--end 0 finishes at seed 0, --end 1 finishes at seed 1048576, --end 3 begins at seed 2097152)
        --device defines which GPU ID runs the cuda kernels. You can check this using nvidia-smi if you're running standalone. Otherwise, if you're running on BOINC, this parameter is unneeded on modern clients. Keep it implemented for old clients.
    */
    uint64_t block_min = 0;
    uint64_t block_max = 0;
    uint64_t checked = 0;
    int device = 0;
    for (int i = 1; i < argc; i += 2) {
		const char *param = argv[i];
		if (strcmp(param, "-d") == 0 || strcmp(param, "--device") == 0) {
			device = atoi(argv[i + 1]);
		} else if (strcmp(param, "-s") == 0 || strcmp(param, "--start") == 0) {
			sscanf(argv[i + 1], "%llu", &block_min);
		} else if (strcmp(param, "-e") == 0 || strcmp(param, "--end") == 0) {
			sscanf(argv[i + 1], "%llu", &block_max);
		} 
        else {
			fprintf(stderr,"Unknown parameter: %s\n", param);
        }
    }
    uint64_t offsetStart = 0;
    uint64_t *out;
    //GPU Params
	int blocks = 32768;
	int threads = 32;
    //BOINC
  	#ifdef BOINC

        BOINC_OPTIONS options;
        boinc_options_defaults(options);
	    options.normal_thread_priority = true;
        boinc_init_options(&options);
        APP_INIT_DATA aid;
	    boinc_get_init_data(aid);
        if (aid.gpu_device_num >= 0) {
            //If BOINC client provided us a device ID
		    device = aid.gpu_device_num;
		    fprintf(stderr,"boinc gpu %i gpuindex: %i \n", aid.gpu_device_num, device);
		} else {
            //If BOINC client did not provide us a device ID
            device = -5;
            for (int i = 1; i < argc; i += 2) {
                //Check for a --device flag, just in case we missed it earlier, use it if it's available. For older clients primarily.
              	if(strcmp(argv[i], "--device") == 0){
                    sscanf(argv[i + 1], "%i", &device);
                }
  
            }
            if(device == -5){
                //Something has gone wrong. It pulled from BOINC, got -1. No --device parameter present.
                fprintf(stderr, "Error: No --device parameter provided! Defaulting to device 0...\n");
                device = 0;
            }
		    fprintf(stderr,"stndalone gpuindex %i (aid value: %i)\n", device, aid.gpu_device_num);
	    }   

        FILE *checkpoint_data = boinc_fopen("checkpoint.txt", "rb");
        if(!checkpoint_data){
            //No checkpoint file was found. Proceed from the beginning.
            fprintf(stderr, "No checkpoint to load\n");

        }
        else{
            //Load from checkpoint. You can put any data in data_store that you need to keep between runs of this program.
            boinc_begin_critical_section();
            struct checkpoint_vars data_store;
            fread(&data_store, sizeof(data_store), 1, checkpoint_data);
            offsetStart = data_store.offset;
            elapsed_chkpoint = data_store.elapsed_chkpoint;
            fprintf(stderr, "Checkpoint loaded, task time %d s, seed pos: %llu\n", elapsed_chkpoint, offsetStart);
            fclose(checkpoint_data);
            boinc_end_critical_section();
        }
    #endif
    cudaSetDevice(device);
    cudaMallocManaged(&out, (blocks * threads) * sizeof(*out));
    for(int i = 0; i < (blocks * threads); i++){
        out[i] = 0;
    }
    auto start = high_resolution_clock::now();
	printf("starting...\n");
    uint64_t checkpointTemp = 0;
    FILE* seedsout = fopen("seeds.txt", "w+");
    for (uint64_t s = (uint64_t)block_min + offsetStart; s < (uint64_t)block_max; s++) {
        //Call GPU kernel
        kernel<<<blocks, threads>>>(blocks * threads * s, out);
        GPU_ASSERT(cudaPeekAtLastError());
        GPU_ASSERT(cudaDeviceSynchronize());  
        //Check error from GPU driver, if any
        checkpointTemp += 1;
        #ifdef BOINC
        if(checkpointTemp >= 15 || boinc_time_to_checkpoint()){
            //Checkpointing for BOINC
            auto checkpoint_end = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(checkpoint_end - start);
            boinc_begin_critical_section(); // Boinc should not interrupt this
            
            // Checkpointing section below
            boinc_delete_file("checkpoint.txt"); // Don't touch, same func as normal fdel
            FILE *checkpoint_data = boinc_fopen("checkpoint.txt", "wb");
            struct checkpoint_vars data_store;
            data_store.offset = s - block_min;
            data_store.elapsed_chkpoint = elapsed_chkpoint + duration.count();
            fwrite(&data_store, sizeof(data_store), 1, checkpoint_data);
            fclose(checkpoint_data);
            checkpointTemp = 0;
            boinc_end_critical_section();
            boinc_checkpoint_completed(); // Checkpointing completed
        }
        //Update boinc client with percentage
        double frac = (double)(s+1 - block_min) / (double)(block_max - block_min);
        boinc_fraction_done(frac);

        #endif
        for (unsigned long long i = 0; i < blocks * threads; i++){
            if(out[i] > 0){
			    fprintf(seedsout,"%llu\n", out[i]);
                out[i] = 0;
                //Grab values from `out` buffer and print to seedsout
                //Set to 0 after to reset
            }

		}
		fflush(seedsout);

    }


    /*
        The end. This prints speed information to stderr.txt - which will be uploaded to the BOINC server, or it can be reviewed locally in a standsalone run.
    */
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    checked = blocks*threads*(block_max - block_min);
    fprintf(stderr, "checked = %" PRIu64 "\n", checked);
    fprintf(stderr, "time taken = %f\n", (double)duration.count()/1000.0);

	double seeds_per_second = checked / ((double)duration.count()/1000.0);
	double speedup = seeds_per_second / 199000;
	fprintf(stderr, "seeds per second: %f\n", seeds_per_second);
	fprintf(stderr, "speedup: %fx\n", speedup);

#ifdef BOINC
    boinc_finish(0);
#endif
}
