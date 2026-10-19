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
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char**argv) {try{
 if(argc!=2)throw std::runtime_error("expected vendor libggml-base path");
 void* lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
 std::mt19937 rng(20261019);int cases=0;double worst=0,totalError=0,totalBase=0;
 constexpr int N=2560,ROWS=8;
 for(int K:{5,6,7}) {
  auto ref=(void(*)(const void*,float*,int64_t))dlsym(lib,("dequantize_row_tqp"+std::to_string(K)).c_str());if(!ref)throw std::runtime_error("missing reference codec");
  const int bytes=2+8*K,rb=N/128*bytes;
  for(int trial=0;trial<8;++trial) {
   std::vector<uint8_t>w(rb*ROWS);for(auto&x:w)x=uint8_t(rng());
   for(size_t b=0;b<w.size();b+=bytes){__half scale=__float2half(trial==0?0.f:float((rng()%100)+1)/128);std::memcpy(w.data()+b,&scale,2);}
   std::vector<float>expected(N*ROWS),actual(N*ROWS);ref(w.data(),expected.data(),N*ROWS);
   void *dw,*dy,*dx,*dq;ck(cudaMalloc(&dw,w.size()));ck(cudaMalloc(&dy,actual.size()*4));ck(cudaMalloc(&dx,N*4));ck(cudaMalloc(&dq,N/32*36));
   ck(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
   strata::kernels::iq_dequant_f32(K+142,dw,N*ROWS,(float*)dy,nullptr);ck(cudaMemcpy(actual.data(),dy,actual.size()*4,cudaMemcpyDeviceToHost));
   if(std::memcmp(actual.data(),expected.data(),actual.size()*4))throw std::runtime_error("GPU FP32 dequant differs");
   std::vector<uint16_t>h(N*ROWS);strata::kernels::iq_dequant_f16(K+142,dw,N*ROWS,(uint16_t*)dy,nullptr);ck(cudaMemcpy(h.data(),dy,h.size()*2,cudaMemcpyDeviceToHost));
   for(size_t i=0;i<h.size();++i){__half e=__float2half(expected[i]);uint16_t bits;std::memcpy(&bits,&e,2);if(bits!=h[i])throw std::runtime_error("GPU FP16 dequant differs");}
   std::vector<float>x(N);for(auto&v:x)v=float(int(rng()%255)-127)/64;
   ck(cudaMemcpy(dx,x.data(),N*4,cudaMemcpyHostToDevice));strata::kernels::quantize_q8_1_rows((float*)dx,1,N,dq,nullptr);
   std::vector<uint8_t>q(N/32*36),q0(N/32*34);ck(cudaMemcpy(q.data(),dq,q.size(),cudaMemcpyDeviceToHost));
   for(int b=0;b<N/32;++b){std::memcpy(q0.data()+b*34,q.data()+b*36,2);std::memcpy(q0.data()+b*34+2,q.data()+b*36+4,32);}
   strata::kernels::iq_mmvq(K+142,dw,dq,(float*)dy,N,ROWS,1,nullptr);std::vector<float>dot(ROWS),cpu(ROWS);ck(cudaMemcpy(dot.data(),dy,ROWS*4,cudaMemcpyDeviceToHost));
   const void*acts[]={q0.data()};float*outs[]={cpu.data()};strata::kernels::cpu::tq_rows(strata::kernels::cpu::TqIsa::scalar,K+142,w.data(),rb,N,acts,1,outs,0,ROWS);
   double err=0,base=0,ce=0,integerError=0,integerBase=0;
   for(int r=0;r<ROWS;++r){double sum=0;for(int b=0;b<N/32;++b){__half d;std::memcpy(&d,q.data()+36*b,2);for(int j=0;j<32;++j)sum+=expected[r*N+32*b+j]*__half2float(d)*int8_t(q[36*b+4+j]);}double intsum=0;
    for(int b=0;b<N/128;++b){__half wd;std::memcpy(&wd,w.data()+r*rb+b*bytes,2);const float scale=__half2float(wd);
     for(int lane=0;lane<4;++lane){__half ad;std::memcpy(&ad,q.data()+36*(4*b+lane),2);int accum=0;
      for(int j=0;j<32;++j){float value=scale?expected[r*N+128*b+32*lane+j]/scale:0;int code=std::lround(value/(3.984375f/127.f));accum+=code*int8_t(q[36*(4*b+lane)+4+j]);}
      intsum+=float(accum)*(scale*(__half2float(ad)*(3.984375f/127.f)));
     }
    }
    integerError+=std::abs(dot[r]-intsum);integerBase+=std::abs(intsum);
    err+=std::abs(dot[r]-sum);ce+=std::abs(cpu[r]-sum);base+=std::abs(sum);}
   std::printf("TQP%d trial%d GPUrel %.8f CPUrel %.8f base %.4f\n",K,trial,err/(base+1e-10),ce/(base+1e-10),base);
   double rel=err/(base+1e-10);worst=std::max(worst,rel);totalError+=err;totalBase+=base;if(integerError/(integerBase+1e-10)>1e-5||ce/(base+1e-10)>.001)throw std::runtime_error("dot disagrees with its exact arithmetic reference");
   ck(cudaFree(dw));ck(cudaFree(dy));ck(cudaFree(dx));ck(cudaFree(dq));++cases;
  }
 }
 if(totalError/(totalBase+1e-10)>.03)throw std::runtime_error("aggregate int8 approximation exceeds reference tolerance");
 dlclose(lib);std::printf("TQP parity passed: %d cases, FP32/FP16 dequant exact, worst GPU dot relative L1 %.8f\n",cases,worst);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"TQP parity failed: %s\n",e.what());return 1;}}
