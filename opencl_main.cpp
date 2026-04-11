#define CL_TARGET_OPENCL_VERSION 300

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>

#if __has_include(<CL/opencl.hpp>)
  #include <CL/opencl.hpp>
#elif __has_include(<CL/cl2.hpp>)
  #include <CL/cl2.hpp>
#elif __has_include(<CL/cl.hpp>)
  #include <CL/cl.hpp>
#endif

#include <time.h>
#include <chrono>
using namespace std::chrono;

#ifdef __GNUC__

#include <unistd.h>
#include <sys/time.h>

#endif

#define __STDC_FORMAT_MACROS 1

#include <stdlib.h>
#include <stddef.h>
#include <inttypes.h>

#ifdef BOINC
  #include "boinc_api.h"
  #include "boinc_opencl.h"
#if defined _WIN32 || defined _WIN64
  #include "boinc_win.h"
#endif
#endif

struct checkpoint_vars {
    unsigned long long offset;
    uint64_t elapsed_chkpoint;
};

uint64_t elapsed_chkpoint = 0;

static const char* kernel_source = R"CLC(
typedef ulong u64;
typedef uint  u32;
typedef uchar u8;

#define XRSR_MIX1          0xbf58476d1ce4e5b9UL
#define XRSR_MIX2          0x94d049bb133111ebUL
#define XRSR_SILVER_RATIO  0x6a09e667f3bcc909UL
#define XRSR_GOLDEN_RATIO  0x9e3779b97f4a7c15UL
#define THRESH 26.0f

inline u64 mix64(u64 a) {
    a = (a ^ (a >> 30)) * XRSR_MIX1;
    a = (a ^ (a >> 27)) * XRSR_MIX2;
    return a ^ (a >> 31);
}

inline u64 rotl64(u64 x, u8 b) {
    return (x << b) | (x >> (64 - b));
}

typedef struct {
    u64 lo;
    u64 hi;
} Xoroshiro;

inline void xSetSeed(__private Xoroshiro *xr, u64 value) {
    const u64 XL = 0x9e3779b97f4a7c15UL;
    const u64 XH = 0x6a09e667f3bcc909UL;
    const u64 A  = 0xbf58476d1ce4e5b9UL;
    const u64 B  = 0x94d049bb133111ebUL;

    u64 l = value ^ XH;
    u64 h = l + XL;

    l = (l ^ (l >> 30)) * A;
    h = (h ^ (h >> 30)) * A;

    l = (l ^ (l >> 27)) * B;
    h = (h ^ (h >> 27)) * B;

    l ^= (l >> 31);
    h ^= (h >> 31);

    xr->lo = l;
    xr->hi = h;
}

inline u64 xNextLong(__private Xoroshiro *xr) {
    u64 l = xr->lo;
    u64 h = xr->hi;

    u64 n = rotl64(l + h, 17) + l;

    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);

    return n;
}

inline int xNextInt(__private Xoroshiro *xr, u32 n) {
    u64 r = (xNextLong(xr) & 0xFFFFFFFFUL) * n;
    if ((u32)r < n) {
        while ((u32)r < ((~n + 1U) % n)) {
            r = (xNextLong(xr) & 0xFFFFFFFFUL) * n;
        }
    }
    return (int)(r >> 32);
}

inline float xNextFloat(__private Xoroshiro *xr) {
    return (float)((xNextLong(xr) >> (64 - 24)) * 5.9604645e-8f);
}

typedef struct {
    Xoroshiro internal;
    int num_calls;
} RNG;

inline RNG rng_new() {
    RNG rng;
    rng.num_calls = 0;
    rng.internal.lo = 0;
    rng.internal.hi = 0;
    return rng;
}

inline void rng_set_seed(__private RNG *rng, u64 seed) {
    seed ^= XRSR_SILVER_RATIO;
    rng->internal.lo = mix64(seed);
    rng->internal.hi = mix64(seed + XRSR_GOLDEN_RATIO);
}

inline u64 rng_next(__private RNG *rng, int bits) {
    rng->num_calls++;
    return xNextLong(&rng->internal) >> (64 - bits);
}

inline int rng_next_int(__private RNG *rng, u32 bound) {
    u32 r = (u32)rng_next(rng, 31);
    u32 m = bound - 1;
    if ((bound & m) == 0) {
        r = (u32)((u64)bound * (u64)r >> 31);
    } else {
        for (u32 u = r; (int)(u - (r = u % bound) + m) < 0; u = (u32)rng_next(rng, 31));
    }
    return (int)r;
}

inline float rng_next_float(__private RNG *rng) {
    return xNextFloat(&rng->internal);
}

inline double rng_next_double(__private RNG *rng) {
    int i = (int)rng_next(rng, 26);
    int j = (int)rng_next(rng, 27);
    u64 k = ((u64)i << 27) + (u64)j;
    return (double)k * 1.110223e-16;
}

inline int rng_next_between_inclusive(__private RNG *rng, int i, int j) {
    return rng_next_int(rng, (u32)(j - i + 1)) + i;
}

inline u64 rng_next_long(__private RNG *rng) {
    int i = (int)rng_next(rng, 32);
    int j = (int)rng_next(rng, 32);
    return ((u64)i << 32) + (u64)j;
}

inline u64 rng_set_feature_seed(__private RNG *rng, u64 seed, int a, int b) {
    u64 i = seed + (u64)a + (u64)(10000 * b);
    rng_set_seed(rng, i);
    return i;
}

typedef struct __attribute__((aligned(16))) {
    int dx;
    int dz;
    int height;
    u8 is_valid;
} Offset;

inline Offset offset_new(int dx, int dz, int height) {
    Offset o;
    o.dx = dx;
    o.dz = dz;
    o.height = height;
    o.is_valid = (u8)1;
    return o;
}

inline Offset get_position_standard(__private RNG *rng) {
    int dx = rng_next_int(rng, 16);
    int dz = rng_next_int(rng, 16);

    int i = -144;
    int j = 16;
    int plateau = 0;

    int l = ((j - i) - plateau) / 2;
    int i1 = (j - i) - l;
    int height = i
        + rng_next_between_inclusive(rng, 0, i1)
        + rng_next_between_inclusive(rng, 0, l);

    return offset_new(dx, dz, height);
}

inline Offset get_small_diamond_position(__private RNG *rng, u64 chunk_seed) {
    (void)chunk_seed;
    return get_position_standard(rng);
}

inline Offset get_medium_diamond_position(__private RNG *rng, u64 chunk_seed) {
    (void)chunk_seed;
    int dx = rng_next_int(rng, 16);
    int dz = rng_next_int(rng, 16);
    int i = -64;
    int j = -4;
    int height = rng_next_between_inclusive(rng, i, j);
    return offset_new(dx, dz, height);
}

inline u8 get_large_diamond_position(__private RNG *rng, u64 chunk_seed) {
    (void)rng_set_feature_seed(rng, chunk_seed, 20, 6);
    return (u8)(rng_next_float(rng) < 0.111111f);
}

inline Offset get_buried_diamond_position(__private RNG *rng, u64 chunk_seed) {
    (void)chunk_seed;
    return get_position_standard(rng);
}

inline u8 close_enough(Offset ref, Offset o) {
    int dx = o.dx - ref.dx;
    int dy = o.height - ref.height;
    int dz = o.dz - ref.dz;
    return (u8)(
        (o.height > -55) &&
        (o.height < -6) &&
        (dx * dx + dy * dy + dz * dz <= THRESH)
    );
}

__kernel void seed_kernel(u64 offset, __global u64 *out, volatile __global u32 *result_count) {
    u64 chunk_seed = (u64)get_global_id(0) + offset;

    RNG rng = rng_new();

    if (!get_large_diamond_position(&rng, chunk_seed)) {
        return;
    }

    Offset ref = get_position_standard(&rng);
    Offset o;

    (void)rng_set_feature_seed(&rng, chunk_seed, 18, 6);
    o = get_small_diamond_position(&rng, chunk_seed);
    if (!close_enough(ref, o)) return;

    (void)rng_set_feature_seed(&rng, chunk_seed, 19, 6);
    o = get_medium_diamond_position(&rng, chunk_seed);
    if (!close_enough(ref, o)) return;

    (void)rng_set_feature_seed(&rng, chunk_seed, 21, 6);
    for (int k = 0; k < 4; k++) {
        o = get_buried_diamond_position(&rng, chunk_seed);
        if (!close_enough(ref, o)) return;

        (void)rng_next_float(&rng);
        (void)rng_next_int(&rng, 3);
        (void)rng_next_int(&rng, 3);
        for (int j = 0; j < 8; j++) {
            (void)rng_next_double(&rng);
        }
    }

    u32 idx = atomic_inc(result_count);
    out[idx] = chunk_seed;
}
)CLC";

int main(int argc, char **argv) {
    uint64_t block_min = 0;
    uint64_t block_max = 0;
    uint64_t checked = 0;
    cl_device_id cl_device = 0;
    cl_platform_id platform = 0;
    cl_int err;

    for (int i = 1; i < argc; i += 2) {
		const char *param = argv[i];
		// if (strcmp(param, "-d") == 0 || strcmp(param, "--device") == 0) {
			// device = atoi(argv[i + 1]);
		if (strcmp(param, "-s") == 0 || strcmp(param, "--start") == 0) {
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

    //BOINC
  	#ifdef BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
	    options.normal_thread_priority = true;
        boinc_init_options(&options);
        APP_INIT_DATA aid;
	    boinc_get_init_data(aid);

        int retval = boinc_get_opencl_ids(&cl_device, &platform);
        if (retval != CL_SUCCESS) {
            fprintf(stderr, "Error occurred obtaining opencl_ids from boinc: %d\n", err);
        }
        if (cl_device != nullptr && platform != nullptr) {
            //If BOINC client provided us a device ID
            fprintf(stderr, "boinc gpu %i platform: %i \n", cl_device, platform);
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
        cl::Device device(cl_device);
    #else
        std::vector<cl::Platform> all_platforms;
        err = cl::Platform::get(&all_platforms);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL error: %d\n", err);
            exit(1);
        }

        if (all_platforms.size() < 1) {
            fprintf(stderr, "No OpenCL platform found!\n");
            exit(1);
        } 

        std::vector<cl::Device> all_devices;
        err = all_platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "OpenCL error: %d\n", err);
            exit(1);
        }

        if (all_devices.size() < 1) {
            fprintf(stderr, "No OpenCL device found!\n");
            exit(1);
        }

        cl::Device device = all_devices[0];
    #endif
    // cudaSetDevice(device);
    // opencl setup start
    printf("starting...\n");
    
    cl::Context ctx;
    cl::Program program;

    // cl_device = all_devices[device];
    ctx = cl::Context({device});

    cl::Program::Sources sources;
    sources.push_back({kernel_source, strlen(kernel_source)});
    program = cl::Program(ctx, sources);
    err = program.build({device});
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error: %d\n", err);
        exit(1);
    }

    cl::CommandQueue queue(ctx, device);

    cl::Buffer cl_out(ctx, CL_MEM_READ_WRITE, sizeof(uint64_t) * 512);
    cl::Buffer result_count(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t));

    uint32_t zero32 = 0ull;
    uint32_t h_result_count = 0;
    uint64_t h_out[512];

    queue.enqueueWriteBuffer(result_count, CL_TRUE, 0, sizeof(uint32_t), &zero32);

    uint64_t zero = 0;

    cl::Kernel seed_kernel(program, "seed_kernel");

    seed_kernel.setArg(0, zero);
    seed_kernel.setArg(1, cl_out);
    seed_kernel.setArg(2, result_count);
    // opencl setup end

    cl::NDRange global_size(1ull << 26);
    cl::NDRange local_size(256);
    
    auto start = high_resolution_clock::now();
    uint64_t checkpointTemp = 0;
    FILE* seedsout = fopen("seeds.txt", "a");
    for (uint64_t s = (uint64_t)block_min + offsetStart; s < (uint64_t)block_max; s++) {
        // kernel launch start
        for (uint64_t i = 0; i < 64; i++) {
            uint64_t o = (s * (1ull << 32)) + (i * (1ull << 26));
            seed_kernel.setArg(0, o);
            err = queue.enqueueNDRangeKernel(seed_kernel, cl::NullRange, global_size, local_size);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "OpenCL error: %d\n", err);
                exit(1);
            }
            queue.finish();
        }
        // kernel launch end
    
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

        queue.enqueueReadBuffer(result_count, CL_TRUE, 0, sizeof(uint32_t),      &h_result_count);
        queue.enqueueReadBuffer(cl_out,          CL_TRUE, 0, sizeof(uint64_t) * 512, h_out);
        
        for (int i = 0; i < (int)h_result_count; i++) {
			fprintf(seedsout,"%llu\n", h_out[i]);
            h_out[i] = 0ull;
        }

        queue.enqueueWriteBuffer(result_count, CL_TRUE, 0, sizeof(uint32_t), &zero32);
        seed_kernel.setArg(2, result_count);    

		fflush(seedsout);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    checked = (1ull << 32) * (block_max - block_min);
    fprintf(stderr, "checked = %" PRIu64 "\n", checked);
    fprintf(stderr, "time taken = %f\n", (double)duration.count()/1000.0);

	double seeds_per_second = checked / ((double)duration.count()/1000.0);
	fprintf(stderr, "seeds per second: %f\n", seeds_per_second);

#ifdef BOINC
    boinc_finish(0);
#endif
}
