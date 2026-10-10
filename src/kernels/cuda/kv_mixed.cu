#include "strata/kernels/kv_mixed.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
namespace strata::kernels {
namespace {
// Quantize once, then write identical bytes to each live destination. Nonresident
// pages have slot -1: only the authoritative host copy (and prefill stage) is written.
__global__ void append(uint8_t* Kd,uint8_t* Vd,int kb,int vb,const int32_t* table,const int32_t* step,
 long long pos,long long count,const float* K,const float* V,int heads,int dim,int ps,
 KvHostPools host,KvHostPools stage) {
 const bool v=blockIdx.z; int bits=v?vb:kb; const float* src=v?V:K;
 const long long tok=blockIdx.y, p=(step?step[kStepPos]:pos)+tok;
 const int group=blockIdx.x, h=group/(dim/32), g=group%(dim/32), lane=threadIdx.x;
 const int slot=table[p/ps];
 const long long row=((long long)slot*heads+h)*ps+p%ps;
 const long long hrow=((p/ps)*heads+h)*ps+p%ps;
 const long long stride=bits==16?dim*2:(dim/32)*(2+bits*4);
 uint8_t* dst[3]={slot>=0?(v?Vd:Kd)+row*stride:nullptr,
                  (v?host.v_mixed:host.k_mixed), (v?stage.v_mixed:stage.k_mixed)};
 for(int j=1;j<3;++j)if(dst[j])dst[j]+=hrow*stride;
 const float x=src[(tok*heads+h)*dim+g*32+lane];
 if(bits==16){
  const __half value=__float2half_rn(x);
  for(int j=0;j<3;++j)if(dst[j])((__half*)dst[j])[g*32+lane]=value;
  return;
 }
 float a=fabsf(x); for(int o=16;o;o>>=1)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,o));
 const int lim=(1<<(bits-1))-1;
 __half hs=__float2half_rn(a/lim); float scale=__half2float(hs);
 int q=scale>0?__float2int_rn(x/scale):0;q=max(-lim,min(lim,q))+(1<<(bits-1));
 __shared__ unsigned codes[32]; codes[lane]=q; __syncthreads();
 for(int target=0;target<3;++target){
  if(!dst[target])continue;
  uint8_t* b=dst[target]+g*(2+bits*4);
  if(lane==0)*(__half*)b=hs;
  for(int j=lane;j<bits*4;j+=32){unsigned val=0;for(int k=0;k<8;++k){int bit=j*8+k;val|=((codes[bit/bits]>>(bit%bits))&1)<<k;}b[2+j]=val;}
 }
}
__global__ void gather(const uint8_t* k,const uint8_t* v,int kb,int vb,const int32_t* table,const int32_t* ids,
 const int32_t* step,int heads,int dim,int ps,uint16_t* K,uint16_t* V){
 const int i=blockIdx.x,h=blockIdx.y,d=threadIdx.x;
 if(i>=step[kStepWidth]||d>=dim)return;
 int p=ids[i];long long row=((long long)table[p/ps]*heads+h)*ps+p%ps;
 long long out=((long long)i*heads+h)*dim+d;
 K[out]=__half_as_ushort(__float2half_rn(kv_mixed_value(k,kb,row,d,dim)));
 V[out]=__half_as_ushort(__float2half_rn(kv_mixed_value(v,vb,row,d,dim)));
}
}
void kv_mixed_append(uint8_t* k,uint8_t* v,int kb,int vb,const int32_t* table,const int32_t* step,
 int64_t pos,int64_t count,const float* K,const float* V,const QsaShapes& s,void* stream,const KvHostPools* host,const KvHostPools* stage){
 if(count<=0)return;
 append<<<dim3(s.n_head_kv*(s.head_dim/32),count,2),32,0,(cudaStream_t)stream>>>(k,v,kb,vb,table,step,pos,count,K,V,s.n_head_kv,s.head_dim,s.page_size,host?*host:KvHostPools{},stage?*stage:KvHostPools{});
 if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("mixed KV append failed");
}
void kv_mixed_gather(const uint8_t* k,const uint8_t* v,int kb,int vb,const int32_t* table,const int32_t* ids,
 const int32_t* step,int64_t cap,const QsaShapes& s,uint16_t* K,uint16_t* V,void* stream){
 gather<<<dim3(cap,s.n_head_kv),s.head_dim,0,(cudaStream_t)stream>>>(k,v,kb,vb,table,ids,step,s.n_head_kv,s.head_dim,s.page_size,K,V);
 if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("mixed KV gather failed");
}
}
