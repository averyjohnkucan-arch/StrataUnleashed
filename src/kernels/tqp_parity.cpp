// Verify pair-trellis decode and dot against independently built vendor GGML.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/cpu/tq_cpu.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <string>
static void check_cuda(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("expected vendor libggml-base path");
        void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!lib)
            throw std::runtime_error(dlerror());
        std::mt19937 rng(20261019);
        int cases = 0;
        double worst_relative_error = 0, total_error = 0, total_reference_l1 = 0;
        constexpr int kColumns = 2560, kRows = 8;
        // Fixed seed, three formats and eight trials: 24 repeatable matrix cases.
        for (int bits_per_step : {5, 6, 7}) {
            auto reference_decode = (void (*)(const void*, float*, int64_t)) dlsym(
                lib, ("dequantize_row_tqp" + std::to_string(bits_per_step)).c_str());
            if (!reference_decode)
                throw std::runtime_error("missing reference codec");
            const int block_bytes = 2 + 8 * bits_per_step, row_bytes = kColumns / 128 * block_bytes;
            for (int trial = 0; trial < 8; ++trial) {
                std::vector<uint8_t> weights(row_bytes * kRows);
                for (auto& x : weights)
                    x = uint8_t(rng());
                for (size_t b = 0; b < weights.size(); b += block_bytes) {
                    __half scale = __float2half(trial == 0 ? 0.f : float((rng() % 100) + 1) / 128);
                    std::memcpy(weights.data() + b, &scale, 2);
                }
                std::vector<float> expected(kColumns * kRows), actual(kColumns * kRows);
                reference_decode(weights.data(), expected.data(), kColumns * kRows);
                // First require exact dequantization. Dot-product tolerances below must not
                // conceal a wrong bitstream, codebook lookup or FP16 conversion.
                void *device_weights, *device_output, *device_activations, *device_q8_1;
                check_cuda(cudaMalloc(&device_weights, weights.size()));
                check_cuda(cudaMalloc(&device_output, actual.size() * 4));
                check_cuda(cudaMalloc(&device_activations, kColumns * 4));
                check_cuda(cudaMalloc(&device_q8_1, kColumns / 32 * 36));
                check_cuda(cudaMemcpy(device_weights, weights.data(), weights.size(), cudaMemcpyHostToDevice));
                strata::kernels::iq_dequant_f32(bits_per_step + 142, device_weights, kColumns * kRows,
                                                (float*) device_output, nullptr);
                check_cuda(cudaMemcpy(actual.data(), device_output, actual.size() * 4, cudaMemcpyDeviceToHost));
                if (std::memcmp(actual.data(), expected.data(), actual.size() * 4))
                    throw std::runtime_error("GPU FP32 dequant differs");
                std::vector<uint16_t> fp16_weights(kColumns * kRows);
                strata::kernels::iq_dequant_f16(bits_per_step + 142, device_weights, kColumns * kRows,
                                                (uint16_t*) device_output, nullptr);
                check_cuda(
                    cudaMemcpy(fp16_weights.data(), device_output, fp16_weights.size() * 2, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < fp16_weights.size(); ++i) {
                    __half e = __float2half(expected[i]);
                    uint16_t bits;
                    std::memcpy(&bits, &e, 2);
                    if (bits != fp16_weights[i])
                        throw std::runtime_error("GPU FP16 dequant differs");
                }
                std::vector<float> x(kColumns);
                for (auto& v : x)
                    v = float(int(rng() % 255) - 127) / 64;
                check_cuda(cudaMemcpy(device_activations, x.data(), kColumns * 4, cudaMemcpyHostToDevice));
                strata::kernels::quantize_q8_1_rows((float*) device_activations, 1, kColumns, device_q8_1, nullptr);
                std::vector<uint8_t> gpu_q8_1(kColumns / 32 * 36), cpu_q8_0(kColumns / 32 * 34);
                check_cuda(cudaMemcpy(gpu_q8_1.data(), device_q8_1, gpu_q8_1.size(), cudaMemcpyDeviceToHost));
                // GPU Q8_1 stores {FP16 scale, FP16 sum, 32 int8 values}; CPU Q8_0 omits
                // the sum. Repack the same quantized values so both see identical inputs.
                for (int b = 0; b < kColumns / 32; ++b) {
                    std::memcpy(cpu_q8_0.data() + b * 34, gpu_q8_1.data() + b * 36, 2);
                    std::memcpy(cpu_q8_0.data() + b * 34 + 2, gpu_q8_1.data() + b * 36 + 4, 32);
                }
                strata::kernels::iq_mmvq(bits_per_step + 142, device_weights, device_q8_1, (float*) device_output,
                                         kColumns, kRows, 1, nullptr);
                std::vector<float> dot(kRows), cpu(kRows);
                check_cuda(cudaMemcpy(dot.data(), device_output, kRows * 4, cudaMemcpyDeviceToHost));
                const void* acts[] = {cpu_q8_0.data()};
                float* outs[] = {cpu.data()};
                strata::kernels::cpu::tq_rows(strata::kernels::cpu::TqIsa::scalar, bits_per_step + 142, weights.data(),
                                              row_bytes, kColumns, acts, 1, outs, 0, kRows);
                double gpu_error = 0, reference_l1 = 0, cpu_error = 0, integer_error = 0, integer_reference_l1 = 0;
                // Compare against two references: the vendor's full-precision dequantized
                // weights, and the GPU's intentionally rounded int8 codebook arithmetic.
                for (int r = 0; r < kRows; ++r) {
                    double sum = 0;
                    for (int b = 0; b < kColumns / 32; ++b) {
                        __half d;
                        std::memcpy(&d, gpu_q8_1.data() + 36 * b, 2);
                        for (int j = 0; j < 32; ++j)
                            sum += expected[r * kColumns + 32 * b + j] * __half2float(d) *
                                   int8_t(gpu_q8_1[36 * b + 4 + j]);
                    }
                    double intsum = 0;
                    for (int b = 0; b < kColumns / 128; ++b) {
                        __half wd;
                        std::memcpy(&wd, weights.data() + r * row_bytes + b * block_bytes, 2);
                        const float scale = __half2float(wd);
                        for (int lane = 0; lane < 4; ++lane) {
                            __half ad;
                            std::memcpy(&ad, gpu_q8_1.data() + 36 * (4 * b + lane), 2);
                            int accum = 0;
                            for (int j = 0; j < 32; ++j) {
                                float value = scale ? expected[r * kColumns + 128 * b + 32 * lane + j] / scale : 0;
                                int code = std::lround(value / (3.984375f / 127.f));
                                accum += code * int8_t(gpu_q8_1[36 * (4 * b + lane) + 4 + j]);
                            }
                            intsum += float(accum) * (scale * (__half2float(ad) * (3.984375f / 127.f)));
                        }
                    }
                    integer_error += std::abs(dot[r] - intsum);
                    integer_reference_l1 += std::abs(intsum);
                    gpu_error += std::abs(dot[r] - sum);
                    cpu_error += std::abs(cpu[r] - sum);
                    reference_l1 += std::abs(sum);
                }
                std::printf("TQP%d trial%d GPUrel %.8f CPUrel %.8f base %.4f\n", bits_per_step, trial,
                            gpu_error / (reference_l1 + 1e-10), cpu_error / (reference_l1 + 1e-10), reference_l1);
                // Per-case gates check each implementation's own arithmetic. Keep the
                // approximation gate aggregated: near-zero reference dots inflate ratios.
                double relative_error = gpu_error / (reference_l1 + 1e-10);
                worst_relative_error = std::max(worst_relative_error, relative_error);
                total_error += gpu_error;
                total_reference_l1 += reference_l1;
                if (integer_error / (integer_reference_l1 + 1e-10) > 1e-5 || cpu_error / (reference_l1 + 1e-10) > .001)
                    throw std::runtime_error("dot disagrees with its exact arithmetic reference");
                check_cuda(cudaFree(device_weights));
                check_cuda(cudaFree(device_output));
                check_cuda(cudaFree(device_activations));
                check_cuda(cudaFree(device_q8_1));
                ++cases;
            }
        }
        if (total_error / (total_reference_l1 + 1e-10) > .03)
            throw std::runtime_error("aggregate int8 approximation exceeds reference tolerance");
        dlclose(lib);
        std::printf("TQP parity passed: %d cases, FP32/FP16 dequant exact, worst GPU dot relative L1 %.8f\n", cases,
                    worst_relative_error);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "TQP parity failed: %s\n", e.what());
        return 1;
    }
}
