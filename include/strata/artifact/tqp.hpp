#pragma once

#include <cstdint>

// The packed stream is shared by the CPU reference and CUDA/HIP kernels.
// Keep this header independent of either runtime so both decode the same bits.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_TQP_HD __host__ __device__
#else
#define STRATA_TQP_HD
#endif
namespace strata::tqp {
// TQP5/6/7 store 128 weights as 64 codebook pairs. Each block starts with
// an FP16 scale (stored as raw bits), followed by 64 steps of K bits each.
template <int K> struct Block {
    uint16_t d;
    uint8_t qs[8 * K];
};
static_assert(sizeof(Block<5>) == 42 && sizeof(Block<6>) == 50 && sizeof(Block<7>) == 58);

// Read the 15-bit state for a pair, with step in [0, 64). Steps walk backwards
// through the circular, LSB-first stream. The last two bytes can wrap to its
// beginning; loading three bytes covers every possible bit alignment.
//
// This returns a state, not a codebook index. Callers hash it and use only the
// first codebook pair, unlike the four-weight TQ formats (types 144-146).
template <int K> STRATA_TQP_HD inline uint32_t state(const uint8_t* qs, int step) {
    constexpr int n = 8 * K;
    const int bit = (63 - step) * K, byte = bit >> 3, shift = bit & 7;
    uint32_t word = uint32_t(qs[byte]) | (uint32_t(qs[(byte + 1) % n]) << 8) | (uint32_t(qs[(byte + 2) % n]) << 16);
    return (word >> shift) & 32767u;
}
}  // namespace strata::tqp
#undef STRATA_TQP_HD
