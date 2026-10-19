#pragma once
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_TQP_HD __host__ __device__
#else
#define STRATA_TQP_HD
#endif
namespace strata::tqp {
template<int K> struct Block { uint16_t d; uint8_t qs[8*K]; };
static_assert(sizeof(Block<5>) == 42 && sizeof(Block<6>) == 50 && sizeof(Block<7>) == 58);
// Vendor circular LSB-first stream; 15-bit state, first codebook pair only.
template<int K> STRATA_TQP_HD inline uint32_t state(const uint8_t* qs, int step) {
    constexpr int n = 8*K;
    const int bit = (63-step)*K, byte = bit >> 3, shift = bit & 7;
    uint32_t word = uint32_t(qs[byte]) | (uint32_t(qs[(byte+1)%n]) << 8) | (uint32_t(qs[(byte+2)%n]) << 16);
    return (word >> shift) & 32767u;
}
}
#undef STRATA_TQP_HD
