#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;
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
static inline uint32_t BSWAP32(uint32_t x) {
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
#
static inline ATTR(const, always_inline, artificial)
uint64_t rotl64(uint64_t x, uint8_t b)
{
    return (x << b) | (x >> (64-b));
}

STRUCT(Xoroshiro)
{
    uint64_t lo, hi;
};

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


uint64_t mix64(uint64_t a) {
	a = (a ^ a >> 30) * XRSR_MIX1;
	a = (a ^ a >> 27) * XRSR_MIX2;
	return a ^ a >> 31;
}

RNG rng_new() {
    return {{0}};
}

static inline uint64_t xNextLong(Xoroshiro thread *xr)
{
    uint64_t l = xr->lo;
    uint64_t h = xr->hi;
    uint64_t n = rotl64(l + h, 17) + l;
    h ^= l;
    xr->lo = rotl64(l, 49) ^ h ^ (h << 21);
    xr->hi = rotl64(h, 28);
    return n;
}

static inline float xNextFloat(Xoroshiro thread *xr)
{
    return (xNextLong(xr) >> (64-24)) * 5.9604645E-8F;
}

uint64_t rng_next(RNG thread *rng, int32_t bits) {
    rng->num_calls++;
    return xNextLong(&rng->internal) >> (64 - bits);
}

static float rng_next_float(RNG thread *rng) {
    return xNextFloat(&rng->internal);
}

static int32_t rng_next_int(RNG thread *rng, uint32_t bound) {
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
static int rng_next_between_inclusive(RNG thread *rng, int i, int j) {
    return rng_next_int(rng, j - i + 1) + i;
}

void rng_set_seed(RNG thread *rng, uint64_t seed) {
    seed ^= XRSR_SILVER_RATIO;
    rng->internal.lo = mix64(seed);
    rng->internal.hi = mix64(seed + XRSR_GOLDEN_RATIO);
}

uint64_t rng_set_feature_seed(RNG thread *rng, uint64_t p_190065_, int32_t p_190066_, int32_t p_190067_) {
    uint64_t i = p_190065_ + (uint64_t)p_190066_ + (uint64_t)(10000 * p_190067_);
    //printf("Salt = %" PRIu64 "\n", (uint64_t)p_190066_ + (uint64_t)(10000 * p_190067_));
    rng_set_seed(rng, i);
    return i;
}


typedef struct {
    int dx, dz, height;
    bool is_valid;
} Offset;

Offset offset_new(int dx, int dz, int height) {
    return {dx, dz, height, true};
}

Offset offset_invalid_new() {
    return {-1, -1, -1, false};
}

Offset get_position_standard(RNG thread *rng) {
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

Offset get_small_diamond_position(RNG thread *rng, uint64_t chunk_seed) {
    // uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 18, 6);
    // (void)feature_seed;
    
    return get_position_standard(rng);
}

Offset get_medium_diamond_position(RNG thread *rng, uint64_t chunk_seed) {
    // uint64_t feature_seed = rng_set_feature_seed(rng, chunk_seed, 19, 6);
    // (void)feature_seed;
    
    int dx = rng_next_int(rng, 16);
    int dz = rng_next_int(rng, 16);

    int i = -64;
    int j = -4;

    int height = rng_next_between_inclusive(rng, i, j);

    return offset_new(dx, dz, height);
}

bool get_large_diamond_position(RNG thread *rng, uint64_t chunk_seed) {
    (void)rng_set_feature_seed(rng, chunk_seed, 20, 6);
    return (rng_next_float(rng) < 1.0F / (float)9.0);
    // idiotic branching... how stupid
    // if (!(rng_next_float(rng) < 1.0F / (float)9.0)) {
        // return offset_invalid_new();
    // }
    // return get_position_standard(rng);
}
Offset get_buried_diamond_position(RNG thread *rng, uint64_t chunk_seed) {
    return get_position_standard(rng);
}

float offset_distance_squared(const Offset thread *a, const Offset thread *b) {
    int x1 = a->dx;
    int y1 = a->height;
    int z1 = a->dz;

    int x2 = b->dx;
    int y2 = b->height;
    int z2 = b->dz;

    return ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1));
}

bool in_range(int y) {
    return (y > -55) && (y < -6);
}

#define THRESH 26.0f

struct CrunchResource {
	uint64_t seed;
	uint64_t out[5012];
	atomic_uint result_count;
};
kernel void crunch(struct CrunchResource device *res, uint2 thread_id [[thread_position_in_grid]], uint2 grid_size [[ threads_per_grid ]]) {
    uint64_t chunk_seed = (uint64_t)thread_id.y * (uint64_t)grid_size.x
                        + (uint64_t)thread_id.x
                        + (res->seed);

	//if(thread_id.y * grid_size.x + thread_id.x == 0) metal::os_log_default.log_info("chunk_seed: %lu", chunk_seed);

    RNG rng = rng_new();
    
    if (!get_large_diamond_position(&rng, chunk_seed)) {
        return;
    }

    Offset ref = get_position_standard(&rng); // large

    Offset o;

    (void)rng_set_feature_seed(&rng, chunk_seed, 18, 6);
    o = get_small_diamond_position(&rng, chunk_seed);
    if (!in_range(o.height) || offset_distance_squared(&ref, &o) > THRESH) {
        return;
    }

    (void)rng_set_feature_seed(&rng, chunk_seed, 19, 6);
    o = get_medium_diamond_position(&rng, chunk_seed);
    if (!in_range(o.height) || offset_distance_squared(&ref, &o) > THRESH) {
        return;
    }

    (void)rng_set_feature_seed(&rng, chunk_seed, 21, 6);
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        o = get_buried_diamond_position(&rng, chunk_seed);
        if (!in_range(o.height) || offset_distance_squared(&ref, &o) > THRESH) {
            return;
        }

        rng_next_float(&rng);
        rng_next_int(&rng, 3);
        rng_next_int(&rng, 3);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            rng_next(&rng, 26);
	    rng_next(&rng, 27);
        }
    }
    metal::os_log_default.log_info("passed! %lu", chunk_seed);
    res->out[atomic_fetch_add_explicit(&res->result_count, 1, memory_order_relaxed)] = chunk_seed;
}
