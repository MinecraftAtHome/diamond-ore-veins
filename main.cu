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

struct Result {
    uint64_t world_seed;
    int32_t x;
    int32_t z;
};

#define NUM_RESULTS 5012
__managed__ Result results[NUM_RESULTS];
__managed__ unsigned long long int result_count = 0;

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


constexpr uint64_t MOD60 = 1ull << 60;
constexpr uint64_t MASK60 = MOD60 - 1;

#define CUDA_FUNCTION __device__ __forceinline__
#define XRSR_MIX1          0xbf58476d1ce4e5b9
#define XRSR_MIX2          0x94d049bb133111eb
#define XRSR_MIX1_INVERSE  0x96de1b173f119089
#define XRSR_MIX2_INVERSE  0x319642b2d24d8ec3
#define XRSR_SILVER_RATIO  0x6a09e667f3bcc909
#define XRSR_GOLDEN_RATIO  0x9e3779b97f4a7c15

CUDA_FUNCTION uint64_t mix64(uint64_t a) {
	a = (a ^ a >> 30) * XRSR_MIX1;
	a = (a ^ a >> 27) * XRSR_MIX2;
	return a ^ a >> 31;
}

CUDA_FUNCTION uint64_t rotl64(uint64_t x, uint8_t b)
{
    return (x << b) | (x >> (64-b));
}

typedef struct {
    uint64_t lo, hi;
} Xoroshiro;

CUDA_FUNCTION static uint64_t xNextLong(Xoroshiro *xr) {
    uint64_t l = xr->lo;
    uint64_t h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

CUDA_FUNCTION static float xNextFloat(Xoroshiro *xr) {
    return (xNextLong(xr) >> (64-24)) * 5.9604645E-8F;
}

CUDA_FUNCTION static uint64_t xNextLongJ(Xoroshiro *xr)
{
    int32_t a = xNextLong(xr) >> 32;
    int32_t b = xNextLong(xr) >> 32;
    return ((uint64_t)a << 32) + b;
}

CUDA_FUNCTION float dot(float a, float b, float c, float d) {
    return __fmaf_ru(b, d, a * c);
}

CUDA_FUNCTION float lensq(float a, float b) {
    return __fmaf_ru(b, b, a * a);
}

template<typename T>
CUDA_FUNCTION void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

CUDA_FUNCTION uint64_t modinv64(uint64_t value) {
    uint64_t x = ((((value << 1) ^ value) & 4) << 1) ^ value;
    x += x - value * x * x;
    x += x - value * x * x;
    x += x - value * x * x;
    x += x - value * x * x;
    return x;
}

typedef struct {
    Xoroshiro internal;
} RNG; // Bruh I really didn't want to have to do this.

CUDA_FUNCTION RNG rng_new() {
    RNG rng;
    rng.internal = {0};
    return rng;
}

CUDA_FUNCTION static void rng_set_seed(RNG *rng, uint64_t seed) {
    seed ^= XRSR_SILVER_RATIO;
    rng->internal.lo = mix64(seed);
    rng->internal.hi = mix64(seed + XRSR_GOLDEN_RATIO);
}

CUDA_FUNCTION static uint64_t rng_next(RNG *rng, int32_t bits) {
    return xNextLong(&rng->internal) >> (64 - bits);
}

CUDA_FUNCTION static int32_t rng_next_int(RNG *rng, uint32_t bound) {
    uint32_t r = rng_next(rng, 31);
    uint32_t m = bound - 1;
    if ((bound & m) == 0) {
        r = (uint32_t)((uint64_t)bound * (uint64_t)r >> 31);
    }
    else {
        for (uint32_t u = r; (int32_t)(u - (r = u % bound) + m) < 0; u = rng_next(rng, 31));
    }
    return r;
}

CUDA_FUNCTION static float rng_next_float(RNG *rng) {
    return xNextFloat(&rng->internal);
}

CUDA_FUNCTION static int rng_next_between_inclusive(RNG *rng, int i, int j) {
    return rng_next_int(rng, j - i + 1) + i;
}

CUDA_FUNCTION static uint64_t rng_next_long(RNG *rng) {
    int32_t i = rng_next(rng, 32);
    int32_t j = rng_next(rng, 32);
    uint64_t k = (uint64_t)i << 32;
    return k + (uint64_t)j;
}

CUDA_FUNCTION static uint64_t rng_set_feature_seed(RNG *rng, uint64_t p_190065_, int32_t p_190066_, int32_t p_190067_) {
    uint64_t i = p_190065_ + (uint64_t)p_190066_ + (uint64_t)(10000 * p_190067_);
    rng_set_seed(rng, i);
    return i;
}

CUDA_FUNCTION uint64_t reverse_decoration_seed(uint64_t decorator_seed, int index, int step) {
    return decorator_seed - (uint64_t)index - 10000L * (uint64_t)step;
}

CUDA_FUNCTION static uint64_t rng_set_decoration_seed(RNG *rng, uint64_t world_seed, int32_t x, int32_t z) {
    rng_set_seed(rng, world_seed);

    uint64_t a = rng_next_long(rng) | 1L;
    uint64_t b = rng_next_long(rng) | 1L;

    uint64_t k = (a * (uint64_t)x + b * (uint64_t)z) ^ world_seed;
    rng_set_seed(rng, k);
    return k;
}

constexpr int32_t THRESH = 16;

typedef __align__(16) struct {
    int32_t dx, dz, height;
} Offset;

constexpr CUDA_FUNCTION bool close_enough(const Offset& o, const Offset &ref, int32_t thresh = THRESH) {
    int dx = o.dx - ref.dx;
    int dy = o.height - ref.height;
    int dz = o.dz - ref.dz;
    return (o.height > -55) && (o.height < -6) &&
           (dx*dx + dy*dy + dz*dz <= thresh);
}

CUDA_FUNCTION Offset get_position_standard(RNG *rng) {
    int dx = rng_next_int(rng, 16); // spread
    int dz = rng_next_int(rng, 16);

    int i = -144;
    int j = 16;
    int plateau = 0;

    int l = ((j-i) - plateau) / 2;
    int i1 = (j-i) - l;
    int height = i + rng_next_between_inclusive(rng, 0, i1) + rng_next_between_inclusive(rng, 0, l);

    return {dx, dz, height};
}

CUDA_FUNCTION bool get_large_diamond_position(RNG *rng, uint64_t chunk_seed) {
    (void)rng_set_feature_seed(rng, chunk_seed, 20, 6);
    return (rng_next_float(rng) < 0.111111f);
}

CUDA_FUNCTION size_t count_veins(RNG *rng, const Offset &cmp, uint64_t chunk_seed) {
    if (!get_large_diamond_position(rng, chunk_seed)) {
        return 0;
    }
    Offset ref = get_position_standard(rng); // large
    if (!close_enough(cmp, ref)) {
        return 0;
    }    
    return 1;
}

/*
all oriented north (-Z)

0 = top left
1 = top right
2 = bottom right
3 = bottom left
*/

__constant__ int32_t offsets[4][3][2] = {
    {{0, -16}, {-16, -16}, {-16, 0}}, // 0
    {{0, -16}, {16, -16}, {16, 0}}, // 1
    {{0, 16}, {16, 16}, {16, 0}}, // 2
    {{0, 16}, {-16, 16}, {-16, 0}}, // 3
};

__constant__ int32_t cmp_offsets[4][3][2] = {
    {{0, 16}, {16, 16}, {16, 0}},
    {{16, 16}, {0, 16}, {0, 0}},
    {{16, 0}, {0, 0}, {0, 16}},
    {{0, 0}, {16, 0}, {16, 16}}
};

CUDA_FUNCTION void check(uint64_t world_seed, int32_t x, int32_t z, int32_t rotation, int32_t ylevel, Result *out) {
    size_t count = 0;
    RNG rng = rng_new();
    
    #pragma unroll
    for (int32_t i = 0; i < 3; i++) {
        int32_t *offset = offsets[rotation][i];
        int32_t *cmp_offset = cmp_offsets[rotation][i];

        Offset cmp = {cmp_offset[0], cmp_offset[1], ylevel};
        uint64_t c = rng_set_decoration_seed(&rng, world_seed, x + offset[0], z + offset[1]);
        count += count_veins(&rng, cmp, c);
    }
    // Offset cmp = {0, 16, -50};
    // uint64_t c1 = rng_set_decoration_seed(&rng, world_seed, x, z - 16);
    // count += count_veins(&rng, cmp, c1);

    // cmp = {16, 16, -50};
    // uint64_t c2 = rng_set_decoration_seed(&rng, world_seed, x - 16, z - 16);
    // count += count_veins(&rng, cmp, c2);
        
    // cmp = {16, 0, -50};
    // uint64_t c3 = rng_set_decoration_seed(&rng, world_seed, x - 16, z);
    // count += count_veins(&rng, cmp, c3);

    if (count >= 2) {
        // printf("candidate: %ld /tp @a %d -50 %d\n", world_seed, x, z);
        out[atomicAdd(&result_count, 1ull)] = {
            world_seed,
            x,
            z,
        };
    }
}

#define hi32(x) (int32_t)(x >> 32)

__device__ int32_t mul128div60(int64_t a, uint64_t b) {
#if defined(__SIZEOF_INT128__)
    return ((__int128)a * b + (MOD60 >> 1)) >> 60;
#else
    uint64_t lo = a * b;
    uint64_t hi = __mul64hi(a, b);
    uint64_t mid = (lo >> 32) + (hi << 32);
    return (mid + (UINT64_C(1) << (60 - 1 - 32))) >> (60 - 32);
#endif
}

__global__ void kernel(uint64_t s, uint64_t chunk_seed, int32_t rotation, int32_t ylevel, Result *out) {
    uint64_t upper60 = (uint64_t)threadIdx.x + (uint64_t)blockDim.x * (uint64_t)blockIdx.x + s;
    uint64_t world_seed = (upper60 << 4) | (chunk_seed & 0xF);

    uint64_t target = ((chunk_seed ^ world_seed) >> 4) & MASK60;

    Xoroshiro xr = {mix64(world_seed ^ XRSR_SILVER_RATIO), mix64((world_seed ^ XRSR_SILVER_RATIO) + XRSR_GOLDEN_RATIO)};
    uint64_t a = (xNextLongJ(&xr) | 1L);
    uint64_t b = (xNextLongJ(&xr) | 1L);

    int64_t binv = modinv64(b);
    int64_t new_z_center = ((binv * target) & MASK60);

    int64_t c = (-a * binv);
    int64_t a0 = 0;   
    int64_t a1 = MOD60;
    int64_t b0 = 1;      
    int64_t b1 = c << 4 >> 4;
    
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        int32_t hi32_b0 = hi32(b0);
        int32_t hi32_b1 = hi32(b1);
        float d0 = dot(hi32(a0), hi32(a1), hi32_b0, hi32_b1);
        float d1 = lensq(hi32_b0, hi32_b1);
        int32_t q = __float2int_rn(d0 / d1);
        a0 -= q * b0;
        a1 -= q * b1;
        swap(a0, b0);
        swap(a1, b1);
    }

    // return ((__int128)a * b + (MOD_60 >> 1)) >> 60;
    // int32_t lx = ((__int128)(-a0) * -new_z_center + (1ull << 59)) >> 60;
    // int32_t lz = ((__int128)(+b0) * -new_z_center + (1ull << 59)) >> 60;

    int32_t lx = mul128div60(-a0, -new_z_center);
    int32_t lz = mul128div60(+b0, -new_z_center);

    int32_t x = lx * b0 + lz * a0;
    int32_t z = lx * b1 + lz * a1 + new_z_center;

    uint64_t result = (a * x + b * z) & MASK60;
    if (!(result ^ target) && x < 1875000 && x > -1875000 && z < 1875000 && z > -1875000) {
        check(world_seed, x << 4, z << 4, rotation, ylevel, out);
        // printf("%lu %d %d\n", world_seed, x << 4, z << 4);
    }
}

static int32_t ylevels[] = {
    -50, 
    -48, 
    -35, 
    -50, 
    -52, 
    -50, 
    -51,
    -45, 
    -50, 
    -50, 
    -40, 
    -39, 
    -46, 
    -49, 
    -50, 
    -41
};

static int32_t rotations[] = {
    0, 
    0, 
    3, 
    1, 
    3, 
    0, 
    0, 
    2, 
    0, 
    2, 
    0, 
    3, 
    2, 
    0, 
    0, 
    0
};

static int64_t chunk_seeds[] = {
    -6891848762888992262ll, 
    1403213307165593138ll, 
    2795686666713625419ll, 
    2619454913059109429ll, 
    3857584051490099048ll, 
    7827231599277226793ll, 
    8088936849810898404ll, 
    -5114902250324241924ll, 
    -3447419567763214185ll, 
    -2386500757356489706ll, 
    -74704388922236045ll, 
    -2839028705456860351ll,
    3953372026202530989ll, 
    1118422581247290462ll, 
    -8968695556427997921ll, 
    9176787649488563472ll
};


#define SIZEOF(x) (sizeof((x)) / sizeof(*(x)))
 
// static_assert(SIZEOF(rotations) == 16);
// static_assert(SIZEOF(chunk_seeds) == 16);
// static_assert(SIZEOF(ylevels) == 16);

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
    //GPU Params
	uint64_t blocks = 16777216;
	uint64_t threads = 256;
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
            (void)(fread(&data_store, sizeof(data_store), 1, checkpoint_data));
            offsetStart = data_store.offset;
            elapsed_chkpoint = data_store.elapsed_chkpoint;
            fprintf(stderr, "Checkpoint loaded, task time %d ms, seed pos: %llu\n", elapsed_chkpoint, offsetStart);
            fclose(checkpoint_data);
            boinc_end_critical_section();
        }
    #endif
    cudaSetDevice(device);

    auto start = high_resolution_clock::now();
	printf("starting...\n");
    uint64_t checkpointTemp = 0;
    FILE* seedsout = fopen("seeds.txt", "a");
    printf("here: %d %d\n", block_min + offsetStart, block_max);
    for (uint64_t s = (uint64_t)block_min + offsetStart; s < (uint64_t)block_max; s++) {
        //Call GPU kernel

        uint32_t idx = s >> 28;
        uint64_t chunk_seed = chunk_seeds[idx];
        int32_t rot = rotations[idx];
        int32_t ylevel = ylevels[idx];
        uint64_t si = s & ((1 << 28) - 1);
        // printf("chunk_seed: %ld rot: %d ylevel: %d si: %lu\n", chunk_seed, rot, ylevel, si);
        kernel<<<blocks, threads>>>(blocks * threads * si, chunk_seed, rot, ylevel, results);

        // kernel<<<blocks, threads>>>(blocks * threads * s, results);
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
        for (unsigned long long i = 0; i < result_count; i++){
            Result r = results[i];
			fprintf(seedsout,"%lld %d %d\n", r.world_seed, r.x, r.z);
		}
        result_count = 0;
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
