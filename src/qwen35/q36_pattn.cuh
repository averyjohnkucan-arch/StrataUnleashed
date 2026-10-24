// src/qwen35/q36_pattn.cuh - the q36 engine's fused prompt attention from the int8 cache (--fused-prompt-attn, #PATTN-1).
//
// A prompt chunk (T > 8 rows) used to run attention as FP16 GEMMs over compact FP16 copies of the cache (k_kv_f16),
// with the FP32 scores and the FP16 probabilities of every slice written to and read back from DRAM. This kernel
// does it in one pass: a block takes one KV head and 8 query tokens x its 8 query heads (64 rows, 4 warps of 16
// rows), walks the cache cells in tiles of 32 (int8 cells dequantised into shared memory, swizzled for ldmatrix),
// S = Q K^T on mma.sync m16n8k16 (FP16 operands, FP32 accumulation), online softmax in registers, O += P V (P in
// FP16, FP32 accumulation). Row t sees the cells [0, p0 + t]. A small T has few blocks, so the cells split over
// `nsplit` blocks (partial accumulator, max and sum per split) and `merge` joins them, as flash-decoding does.
// Numerics: the same FP16 operand roundings as the GEMM path (Q, K, V and P in FP16, FP32 sums), another summation
// order and an online softmax: class B, not bit-equal. int8 cache (q8_cell) only; CUDA only (mma.sync).
#pragma once
#if !defined(__HIPCC__)
#include "q36_attn.cuh"
#include <algorithm>

namespace q36pattn {

using q8_cell_t = q36attn::q8_cell;

constexpr int HD = 256;       // head dim
constexpr int G = 8;          // query heads per KV head (the kernel is built for 8)
constexpr int BC = 32;        // cache cells per tile
constexpr int kTok = 8;       // query tokens per block (8 tokens x G heads = 64 rows)
constexpr int kThreads = 128;

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
// (mma.sync m16n8k16 on FP16 and ldmatrix need sm_80: below it the kernel bodies are empty and the engine refuses the flag)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define Q36_PATTN_ARCH 1
#endif
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#ifdef Q36_PATTN_ARCH
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
// S = Q K^T on int8 (--fused-prompt-attn-i8): m16n8k32 s8 x s8 -> s32, four times the FP16 rate on sm_80+ (a zero start each: the
// caller scales every 32-dim block's integer sums by its Q and K scales)
__device__ __forceinline__ void mma16832s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#ifdef Q36_PATTN_ARCH
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
#endif
}
__device__ __forceinline__ uint32_t pack_s8x4(float4 x, float inv) {
    const int a = max(-127, min(127, __float2int_rn(x.x * inv))), b = max(-127, min(127, __float2int_rn(x.y * inv)));
    const int c = max(-127, min(127, __float2int_rn(x.z * inv))), d = max(-127, min(127, __float2int_rn(x.w * inv)));
    return (uint32_t) (a & 0xff) | ((uint32_t) (b & 0xff) << 8) | ((uint32_t) (c & 0xff) << 16) | ((uint32_t) (d & 0xff) << 24);
}
__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
#ifdef Q36_PATTN_ARCH
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}
__device__ __forceinline__ void ldsm4t(uint32_t* r, const void* p) {
#ifdef Q36_PATTN_ARCH
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}

// A cell's 16 int8 values (a uint4) of 32-value block scale d, into two 16-byte shared chunks (8 halves each) of a tile
// row, XOR-swizzled by the row so that ldmatrix's eight rows hit eight different 16-byte banks groups.
__device__ __forceinline__ void put_chunks(__half* tile, int row, int j, uint4 v, float d) {
    uint32_t w[4] = {v.x, v.y, v.z, v.w}, o[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float f0 = (float) (int8_t) (w[i] & 0xff) * d, f1 = (float) (int8_t) ((w[i] >> 8) & 0xff) * d;
        const float f2 = (float) (int8_t) ((w[i] >> 16) & 0xff) * d, f3 = (float) (int8_t) (w[i] >> 24) * d;
        o[2 * i] = pack_h2(f0, f1);
        o[2 * i + 1] = pack_h2(f2, f3);
    }
    __half* base = tile + (size_t) row * HD;
    *reinterpret_cast<uint4*>(base + (((2 * j) ^ (row & 7)) * 8)) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(base + (((2 * j + 1) ^ (row & 7)) * 8)) = make_uint4(o[4], o[5], o[6], o[7]);
}

/// Grid (ceil(T / 8), HKV, nsplit), 128 threads. q: [T][H][HD] FP32 (H = HKV * 8). Row (t, h) attends to the cells
/// [0, p0 + t]. nsplit == 1: out [T][H][HD] FP32 (normalised). else the split's partial (pacc [nsplit][T][H][HD]
/// unnormalised, pm / pl [nsplit][T][H], base-2 max and sum) for `merge`. `split` = cells per split (a multiple of BC).
template <bool I8>
__global__ void __launch_bounds__(kThreads, 2)
pattn(const float* __restrict__ q, const q8_cell_t* __restrict__ kc, const q8_cell_t* __restrict__ vc, int ctx, int H,
      int T, int p0, int split, int nsplit, float scale_log2, float* __restrict__ out, float* __restrict__ pacc,
      float* __restrict__ pm, float* __restrict__ pl) {
#ifdef Q36_PATTN_ARCH
    __shared__ __align__(16) __half Ks[BC * HD];
    __shared__ __align__(16) __half Vs[BC * HD];
    __shared__ float ksc[I8 ? 8 * BC : 1];   // I8: the K cells' 32-block scales, [block][cell]
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, g = lane >> 2, qd = lane & 3;
    const int tok0 = blockIdx.x * kTok, kvh = blockIdx.y, sp = blockIdx.z;
    const int tA = tok0 + 2 * w, tB = tA + 1, head = kvh * G + g;
    const bool vA = tA < T, vB = tB < T;
    // Q as A fragments (rows g -> token A, g + 8 -> token B; FP16 like the GEMM path's q16)
    uint32_t qa[I8 ? 1 : 16][4];
    uint32_t qi[I8 ? 8 : 1][4];
    float sq[I8 ? 8 : 1][2];
    if constexpr (I8) {   // Q as s8 A fragments, one scale per row and 32 dims (the quad's four lanes share a row's block)
        const float* qA = q + ((size_t) tA * H + head) * HD;
        const float* qB = q + ((size_t) tB * H + head) * HD;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const int d = b * 32 + 4 * qd;
            const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 a0 = vA ? *reinterpret_cast<const float4*>(qA + d) : z, a1 = vA ? *reinterpret_cast<const float4*>(qA + d + 16) : z;
            const float4 b0 = vB ? *reinterpret_cast<const float4*>(qB + d) : z, b1 = vB ? *reinterpret_cast<const float4*>(qB + d + 16) : z;
            float ma = fmaxf(fmaxf(fmaxf(fabsf(a0.x), fabsf(a0.y)), fmaxf(fabsf(a0.z), fabsf(a0.w))),
                             fmaxf(fmaxf(fabsf(a1.x), fabsf(a1.y)), fmaxf(fabsf(a1.z), fabsf(a1.w))));
            float mb = fmaxf(fmaxf(fmaxf(fabsf(b0.x), fabsf(b0.y)), fmaxf(fabsf(b0.z), fabsf(b0.w))),
                             fmaxf(fmaxf(fabsf(b1.x), fabsf(b1.y)), fmaxf(fabsf(b1.z), fabsf(b1.w))));
            ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
            ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 2));
            mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
            mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 2));
            const float ia = ma > 0.f ? 127.f / ma : 0.f, ib = mb > 0.f ? 127.f / mb : 0.f;
            qi[b][0] = pack_s8x4(a0, ia);
            qi[b][1] = pack_s8x4(b0, ib);
            qi[b][2] = pack_s8x4(a1, ia);
            qi[b][3] = pack_s8x4(b1, ib);
            sq[b][0] = ma * (1.f / 127.f);
            sq[b][1] = mb * (1.f / 127.f);
        }
    } else {
        const float* qA = q + ((size_t) tA * H + head) * HD;
        const float* qB = q + ((size_t) tB * H + head) * HD;
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
            const int d = ks * 16 + 2 * qd;
            float2 a0 = vA ? *reinterpret_cast<const float2*>(qA + d) : make_float2(0.f, 0.f);
            float2 a1 = vB ? *reinterpret_cast<const float2*>(qB + d) : make_float2(0.f, 0.f);
            float2 a2 = vA ? *reinterpret_cast<const float2*>(qA + d + 8) : make_float2(0.f, 0.f);
            float2 a3 = vB ? *reinterpret_cast<const float2*>(qB + d + 8) : make_float2(0.f, 0.f);
            qa[ks][0] = pack_h2(a0.x, a0.y);
            qa[ks][1] = pack_h2(a1.x, a1.y);
            qa[ks][2] = pack_h2(a2.x, a2.y);
            qa[ks][3] = pack_h2(a3.x, a3.y);
        }
    }
    const int tlast = min(tok0 + kTok - 1, T - 1);
    const int nvis = p0 + tlast + 1;
    const int c_begin = sp * split, c_end = min(c_begin + split, nvis);
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    const q8_cell_t* kbase = kc + (size_t) kvh * ctx;
    const q8_cell_t* vbase = vc + (size_t) kvh * ctx;
    const int visA = vA ? p0 + tA : -1, visB = vB ? p0 + tB : -1;   // last visible cell of each row
    for (int c0 = c_begin; c0 < c_end; c0 += BC) {
        __syncthreads();
        {   // the tile's K and V: thread = (cell tid / 4, 16-byte parts j = i * 4 + tid % 4 of its 256 int8)
            const int cl = tid >> 2, part = tid & 3, cell = c0 + cl;
            const bool ok = cell < c_end;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int j = i * 4 + part;
                uint4 kv = make_uint4(0, 0, 0, 0), vv = kv;
                float kd = 0.f, vd = 0.f;
                if (ok) {
                    kv = *reinterpret_cast<const uint4*>(kbase[cell].q + j * 16);
                    vv = *reinterpret_cast<const uint4*>(vbase[cell].q + j * 16);
                    kd = kbase[cell].d[j >> 1];
                    vd = vbase[cell].d[j >> 1];
                }
                if constexpr (I8) {   // K stays int8 (16-byte chunks, swizzled by the row), its scales go to ksc
                    *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(Ks) + cl * HD + ((j ^ (cl & 7)) * 16)) = kv;
                    if (i < 2) ksc[(part * 2 + i) * BC + cl] = ok ? kbase[cell].d[part * 2 + i] : 0.f;
                } else {
                    put_chunks(Ks, cl, j, kv, kd);
                }
                put_chunks(Vs, cl, j, vv, vd);
            }
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
        if constexpr (I8) {
#pragma unroll
            for (int b = 0; b < 8; ++b) {
#pragma unroll
                for (int pr = 0; pr < 2; ++pr) {
                    const int mm = lane >> 3, cell = pr * 16 + (lane & 7) + ((mm >> 1) << 3), ch = b * 2 + (mm & 1);
                    uint32_t r[4];
                    ldsm4(r, reinterpret_cast<const uint8_t*>(Ks) + cell * HD + ((ch ^ (cell & 7)) * 16));
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        int acc[4];
                        mma16832s8(acc, qi[b], r[2 * e], r[2 * e + 1]);
                        const int nt = 2 * pr + e;
                        const float2 kc2 = *reinterpret_cast<const float2*>(&ksc[b * BC + nt * 8 + 2 * qd]);
                        s[nt][0] = fmaf((float) acc[0], sq[b][0] * kc2.x, s[nt][0]);
                        s[nt][1] = fmaf((float) acc[1], sq[b][0] * kc2.y, s[nt][1]);
                        s[nt][2] = fmaf((float) acc[2], sq[b][1] * kc2.x, s[nt][2]);
                        s[nt][3] = fmaf((float) acc[3], sq[b][1] * kc2.y, s[nt][3]);
                    }
                }
            }
        } else {
#pragma unroll
        for (int ks = 0; ks < 16; ++ks) {
#pragma unroll
            for (int pr = 0; pr < 2; ++pr) {
                const int mm = lane >> 3, cell = pr * 16 + (lane & 7) + ((mm >> 1) << 3), ch = ks * 2 + (mm & 1);
                uint32_t r[4];
                ldsm4(r, Ks + cell * HD + ((ch ^ (cell & 7)) * 8));
                mma16816(s[2 * pr], qa[ks], r[0], r[1]);
                mma16816(s[2 * pr + 1], qa[ks], r[2], r[3]);
            }
        }
        }
        // scale, causal / split mask, online softmax
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int c = c0 + nt * 8 + 2 * qd;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int cc = c + e;
                s[nt][e] = (cc <= visA && cc < c_end) ? s[nt][e] * scale_log2 : -INFINITY;
                s[nt][2 + e] = (cc <= visB && cc < c_end) ? s[nt][2 + e] * scale_log2 : -INFINITY;
                mx[0] = fmaxf(mx[0], s[nt][e]);
                mx[1] = fmaxf(mx[1], s[nt][2 + e]);
            }
        }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            const float mn = fmaxf(m[r], mx[r]);
            const float mu = mn == -INFINITY ? 0.f : mn;
            alpha[r] = exp2f(m[r] - mu);
            m[r] = mn;
            l[r] *= alpha[r];
        }
        uint32_t pa[2][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float muA = m[0] == -INFINITY ? 0.f : m[0], muB = m[1] == -INFINITY ? 0.f : m[1];
            const float pe0 = exp2f(s[nt][0] - muA), pe1 = exp2f(s[nt][1] - muA);
            const float pe2 = exp2f(s[nt][2] - muB), pe3 = exp2f(s[nt][3] - muB);
            l[0] += pe0 + pe1;
            l[1] += pe2 + pe3;
            pa[nt >> 1][(nt & 1) * 2] = pack_h2(pe0, pe1);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_h2(pe2, pe3);
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            o[i][0] *= alpha[0];
            o[i][1] *= alpha[0];
            o[i][2] *= alpha[1];
            o[i][3] *= alpha[1];
        }
        // fragment order: a0 = (row g, cells 2q..), a1 = (row g + 8, same), a2 = (row g, cells + 8), a3 = (row g + 8)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const uint32_t a[4] = {pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3]};
#pragma unroll
            for (int dp = 0; dp < 16; ++dp) {
                const int mm = lane >> 3, cell = kk * 16 + (lane & 7) + ((mm & 1) << 3), ch = dp * 2 + (mm >> 1);
                uint32_t r[4];
                ldsm4t(r, Vs + cell * HD + ((ch ^ (cell & 7)) * 8));
                mma16816(o[2 * dp], a, r[0], r[1]);
                mma16816(o[2 * dp + 1], a, r[2], r[3]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
    const int toks[2] = {tA, tB};
    const bool val[2] = {vA, vB};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (!val[r]) continue;
        const size_t row = (size_t) toks[r] * H + head;
        if (nsplit == 1) {
            const float inv = l[r] > 0.f ? 1.f / l[r] : 0.f;
            float* dst = out + row * HD;
#pragma unroll
            for (int n = 0; n < 32; ++n)
                *reinterpret_cast<float2*>(dst + n * 8 + 2 * qd) = make_float2(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
        } else {
            const size_t prow = (size_t) sp * T * H + row;
            float* dst = pacc + prow * HD;
#pragma unroll
            for (int n = 0; n < 32; ++n)
                *reinterpret_cast<float2*>(dst + n * 8 + 2 * qd) = make_float2(o[n][2 * r], o[n][2 * r + 1]);
            if (qd == 0) {
                pm[prow] = m[r];
                pl[prow] = l[r];
            }
        }
    }
#endif
}

/// out [T][H][HD] = the splits' partials joined (base-2 max / sum as `pattn` writes them). Grid (T * H), block 256.
__global__ void merge(const float* __restrict__ pacc, const float* __restrict__ pm, const float* __restrict__ pl, int rows,
                      int nsplit, float* __restrict__ out) {
    const size_t row = blockIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < nsplit; ++s) M = fmaxf(M, pm[(size_t) s * rows + row]);
    float L = 0.f, acc = 0.f;
    for (int s = 0; s < nsplit; ++s) {
        const float ms = pm[(size_t) s * rows + row];
        if (ms == -INFINITY) continue;
        const float e = exp2f(ms - M);
        L += pl[(size_t) s * rows + row] * e;
        acc += pacc[((size_t) s * rows + row) * HD + threadIdx.x] * e;
    }
    out[row * HD + threadIdx.x] = L > 0.f ? acc / L : 0.f;
}

/// Rows the partial buffers hold (nsplit * T * H of a launch is kept under it): 16 MiB of accumulators.
constexpr size_t kPartialRows = 16384;
inline size_t partial_acc_floats() { return kPartialRows * HD; }

/// The flag needs an sm_80+ device and the 16 / 2 x 256 head layout the kernel is built for (G query heads per KV head).
inline bool supported(int major, int n_head, int n_head_kv, int head_dim) {
    return major >= 8 && head_dim == HD && n_head_kv > 0 && n_head == G * n_head_kv;
}

/// Launch: attention of the T query rows (positions p0 ..) over the int8 cache cells (rows ctx cells apart per KV
/// head). `pacc/pm/pl`: scratch for the splits (partial_acc_floats(), kPartialRows, kPartialRows), may be null when
/// the grid alone fills the card (then no split is used).
inline cudaError_t launch(const float* q, const q8_cell_t* kc, const q8_cell_t* vc, int ctx, int H, int HKV, int T, int p0,
                          float* out, float* pacc, float* pm, float* pl, int n_sm, cudaStream_t st, bool i8 = false) {
    const int P = p0 + T, ntile = (T + kTok - 1) / kTok, blocks = ntile * HKV;
    int nsplit = (2 * n_sm + blocks - 1) / blocks;
    nsplit = std::max(1, std::min(nsplit, P / 512));
    if (!pacc) nsplit = 1;
    while (nsplit > 1 && (size_t) nsplit * T * H > kPartialRows) --nsplit;
    int split = (P + nsplit - 1) / nsplit;
    split = (split + BC - 1) / BC * BC;
    nsplit = (P + split - 1) / split;
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    if (i8)   // S = Q K^T on the int8 tensor cores
        pattn<true><<<dim3(ntile, HKV, nsplit), kThreads, 0, st>>>(q, kc, vc, ctx, H, T, p0, split, nsplit, scale_log2, out, pacc, pm, pl);
    else
        pattn<false><<<dim3(ntile, HKV, nsplit), kThreads, 0, st>>>(q, kc, vc, ctx, H, T, p0, split, nsplit, scale_log2, out, pacc, pm, pl);
    if (nsplit > 1) merge<<<T * H, HD, 0, st>>>(pacc, pm, pl, T * H, nsplit, out);
    return cudaGetLastError();
}

}  // namespace q36pattn
#else
// HIP: no mma.sync kernel; the engine refuses the flag there (the references in q36_engine.cu still have to compile)
#include <cuda_runtime.h>
namespace q36pattn {
constexpr size_t kPartialRows = 16384;
inline size_t partial_acc_floats() { return 0; }
inline bool supported(int, int, int, int) { return false; }
inline cudaError_t launch(const float*, const void*, const void*, int, int, int, int, int, float*, float*, float*, float*, int,
                          cudaStream_t, bool = false) {
    return cudaErrorNotSupported;
}
}  // namespace q36pattn
#endif
