#include "strata/kernels/kv_mixed.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
using namespace strata::kernels;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* alloc(size_t n){T* p;ck(cudaMalloc((void**)&p,n*sizeof(T)));return p;}
int main(){
 QsaShapes s=qsa_real_shapes(); const int N=137,H=s.n_head_kv,D=s.head_dim,NP=(N+s.page_size-1)/s.page_size;
 std::vector<int32_t> pages(NP),ids(N),steps(kStepCount,0);for(int i=0;i<NP;++i)pages[i]=NP-1-i;
 for(int i=0;i<N;++i)ids[i]=(i*37)%N; steps[kStepWidth]=N;steps[kStepPos]=N;
 auto pt=alloc<int32_t>(NP),di=alloc<int32_t>(N),ds=alloc<int32_t>(kStepCount);
 ck(cudaMemcpy(pt,pages.data(),NP*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(di,ids.data(),N*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(ds,steps.data(),kStepCount*4,cudaMemcpyHostToDevice));
 std::vector<float>x(N*H*D),q(s.n_head*D);
 for(size_t i=0;i<x.size();++i)x[i]=std::sin(i*0.371)*2+std::cos(i*0.121)*0.2;
 for(size_t i=0;i<q.size();++i)q[i]=std::sin(i*0.123)*0.1;
 auto dx=alloc<float>(x.size()),dq=alloc<float>(q.size());ck(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));ck(cudaMemcpy(dq,q.data(),q.size()*4,cudaMemcpyHostToDevice));
 auto go=alloc<uint16_t>(x.size()),vo=alloc<uint16_t>(x.size());auto attn=alloc<float>(q.size()),scratch=alloc<float>(qsa_decode_attn_scratch_floats(N,s));
 int failures=0;
 for(auto pair:std::vector<std::pair<int,int>>{{16,8},{8,5},{5,5},{5,4}}){int kb=pair.first,vb=pair.second;
  auto k=alloc<uint8_t>(NP*s.page_size*H*kv_mixed_row_bytes(kb,D)),v=alloc<uint8_t>(NP*s.page_size*H*kv_mixed_row_bytes(vb,D));
  kv_mixed_append(k,v,kb,vb,pt,nullptr,0,N,dx,dx,s,nullptr);
  kv_mixed_gather(k,v,kb,vb,pt,di,ds,N,s,go,vo,nullptr);ck(cudaDeviceSynchronize());
  std::vector<uint16_t>kg(x.size()),vg(x.size());ck(cudaMemcpy(kg.data(),go,x.size()*2,cudaMemcpyDeviceToHost));ck(cudaMemcpy(vg.data(),vo,x.size()*2,cudaMemcpyDeviceToHost));
  double maxerr=0;
  for(int side=0;side<2;++side){int bits=side?vb:kb;auto& got=side?vg:kg;
   for(int i=0;i<N;++i)for(int h=0;h<H;++h)for(int d=0;d<D;++d){int base=(ids[i]*H+h)*D+(d/32)*32;float want=x[base+d%32];
    if(bits!=16){float a=0;for(int j=0;j<32;++j)a=std::max(a,std::fabs(x[base+j]));int lim=(1<<(bits-1))-1;float sc=strata::fp16_to_fp32(f16_from_f32(a/lim));int c=sc?std::nearbyint(want/sc):0;want=std::clamp(c,-lim,lim)*sc;}
    want=strata::fp16_to_fp32(f16_from_f32(want));maxerr=std::max(maxerr,(double)std::fabs(want-strata::fp16_to_fp32(got[(i*H+h)*D+d])));
   }
  }
  QsaAttnPools pools;pools.page_table=pt;pools.k_mixed=k;pools.v_mixed=v;pools.k_bits=kb;pools.v_bits=vb;
  qsa_decode_attn_step(dq,pools,di,ds,N,s,scratch,attn,nullptr);ck(cudaDeviceSynchronize());std::vector<float>out(q.size());ck(cudaMemcpy(out.data(),attn,q.size()*4,cudaMemcpyDeviceToHost));
  double ae=0;for(int h=0;h<s.n_head;++h){int kh=h/(s.n_head/H);std::vector<double>w(N);double sum=0;
   for(int i=0;i<N;++i){double dot=0;for(int d=0;d<D;++d)dot+=q[h*D+d]*strata::fp16_to_fp32(kg[(i*H+kh)*D+d]);w[i]=std::exp(dot/std::sqrt(D));sum+=w[i];}
   for(int d=0;d<D;++d){double ref=0;for(int i=0;i<N;++i)ref+=w[i]*strata::fp16_to_fp32(vg[(i*H+kh)*D+d]);ae=std::max(ae,std::fabs(out[h*D+d]-ref/sum));}
  }
  // Gather rounds to FP16 while direct attention multiplies scales in FP32.
  bool ok=maxerr==0 && ae<0.001;failures+=!ok;std::printf("K%d/V%d storage max %.3g attention max %.3g %s\n",kb,vb,maxerr,ae,ok?"PASS":"FAIL");cudaFree(k);cudaFree(v);
 }
 return failures?1:0;
}
