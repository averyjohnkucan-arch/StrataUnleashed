// Test-only C ABI for independent vendor codec comparison.
#include "strata/artifact/tqp.hpp"
#include "../third_party/ggml/ggml-common-tq.h"
#include <cstring>
#include <bit>
static float half(uint16_t h) {
 const unsigned s=unsigned(h&0x8000)<<16;unsigned e=(h>>10)&31,m=h&1023;
 if(!e) { if(!m)return std::bit_cast<float>(s);int shift=0;while(!(m&1024)){m<<=1;++shift;}return std::bit_cast<float>(s|unsigned(113-shift)<<23|(m&1023)<<13); }
 return std::bit_cast<float>(s|(e==31?255:e+112)<<23|m<<13);
}
template<int K> static void decode(const uint8_t* b,float* out) {
 uint16_t h;std::memcpy(&h,b,2);const float d=half(h);
 for(int t=0;t<64;++t) { const auto index=(strata::tqp::state<K>(b+2,t)*0x9e3779b1u)>>21;
  out[2*t]=d*half(tq_lut_f16[2*index]);out[2*t+1]=d*half(tq_lut_f16[2*index+1]); }
}
extern "C" void tqp_probe(int k,const uint8_t* b,float* out) {
 if(k==5)decode<5>(b,out);else if(k==6)decode<6>(b,out);else if(k==7)decode<7>(b,out);
}
