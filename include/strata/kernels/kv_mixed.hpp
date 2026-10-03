#pragma once
#include "strata/kernels/qsa.hpp"
#include <cstdint>
namespace strata::kernels {
inline uint64_t kv_mixed_row_bytes(int bits, int dim) { return bits == 16 ? dim * 2ull : (dim / 32ull) * (2 + bits * 4); }
void kv_mixed_append(uint8_t* k, uint8_t* v, int kb, int vb, const int32_t* table,
 const int32_t* step, int64_t pos, int64_t count, const float* K, const float* V, const QsaShapes& s, void* stream);
void kv_mixed_gather(const uint8_t* k, const uint8_t* v, int kb, int vb, const int32_t* table,
 const int32_t* ids, const int32_t* step, int64_t cap, const QsaShapes& s, uint16_t* K, uint16_t* V, void* stream);
}
#if defined(__CUDACC__) || defined(__HIPCC__)
#include <cuda_fp16.h>
namespace strata::kernels {
__device__ inline float kv_mixed_value(const uint8_t* data, int bits, long long row, int d, int dim = 256) {
 if(bits == 16) return __half2float(((const __half*)data)[row*dim+d]);
 const int stride = 2 + bits*4;
 const uint8_t* b=data+row*(dim/32)*stride+(d/32)*stride;
 const int bit=(d%32)*bits, byte=bit/8, shift=bit%8;
 unsigned code=b[2+byte] >> shift;
 if(shift+bits>8) code |= (unsigned)b[3+byte] << (8-shift);
 const int q=(code & ((1u<<bits)-1))-(1<<(bits-1));
 return q*__half2float(*(const __half*)b);
}
}
#endif
