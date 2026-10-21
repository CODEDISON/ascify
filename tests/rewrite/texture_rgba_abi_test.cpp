// Exact old helper bodies, renamed only by the test preprocessor, prove
// bidirectional descriptor rejection. Protected CPU pages detect data loads.
#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#define __NPU_COMPILER_INTERNAL_PURE_SIMT__
#include <ascify/texture_compat.hpp>
namespace ascify_texture_v1 { using ascify::atomic_or_global; }
#undef ASCIFY_TEXTURE_SAMPLING_HPP
#define ascify ascify_texture_v1
#include "fixtures/texture_abi_v1/texture_sampling.hpp"
#undef ascify
namespace ascify_texture_v1 {
using cudaTextureObject_t=unsigned long long;
#include "fixtures/texture_abi_v1/wrappers.inc"
}
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

static unsigned rejects=0;
static void oldReaders(unsigned long long token, uint32_t &error, bool layered) {
  error=0;
  assert(std::isnan(layered?ascify_texture_v1::tex2DLayered<float>(token,0,0,0):
                            ascify_texture_v1::tex2D<float>(token,0,0)));
  assert(error==1);++rejects;error=0;
  assert(std::isnan(layered?ascify_texture_v1::tex2DLayeredFast<float>(token,0,0,0):
                            ascify_texture_v1::tex2DFast<float>(token,0,0)));
  assert(error==1);++rejects;error=0;
  assert((layered?ascify_texture_v1::tex2DLayered<unsigned int>(token,0,0,0):
                   ascify_texture_v1::tex2D<unsigned int>(token,0,0))==0);
  assert(error==1);++rejects;error=0;
  assert((layered?ascify_texture_v1::tex2DLayeredFast<unsigned int>(token,0,0,0):
                   ascify_texture_v1::tex2DFast<unsigned int>(token,0,0))==0);
  assert(error==1);++rejects;
}
static void newReaders(unsigned long long token, uint32_t &error, bool layered) {
  error=0;
  assert(std::isnan(layered?ascify::tex2DLayered<float>(token,0,0,0):ascify::tex2D<float>(token,0,0)));
  assert(error==1);++rejects;error=0;
  assert(std::isnan(layered?ascify::tex2DLayeredFast<float>(token,0,0,0):ascify::tex2DFast<float>(token,0,0)));
  assert(error==1);++rejects;error=0;
  assert((layered?ascify::tex2DLayered<unsigned int>(token,0,0,0):ascify::tex2D<unsigned int>(token,0,0))==0);
  assert(error==1);++rejects;error=0;
  assert((layered?ascify::tex2DLayeredFast<unsigned int>(token,0,0,0):ascify::tex2DFast<unsigned int>(token,0,0))==0);
  assert(error==1);++rejects;
  for(bool fast:{false,true}) {
    error=0;
    const auto bytes=layered?(fast?ascify::tex2DLayeredFast<uchar4>(token,0,0,0):
                                  ascify::tex2DLayered<uchar4>(token,0,0,0)):
                            (fast?ascify::tex2DFast<uchar4>(token,0,0):ascify::tex2D<uchar4>(token,0,0));
    assert(bytes.x==0 && bytes.y==0 && bytes.z==0 && bytes.w==0 && error==1);++rejects;
    error=0;
    const auto floats=layered?(fast?ascify::tex2DLayeredFast<float4>(token,0,0,0):
                                   ascify::tex2DLayered<float4>(token,0,0,0)):
                             (fast?ascify::tex2DFast<float4>(token,0,0):ascify::tex2D<float4>(token,0,0));
    assert(std::isnan(floats.x) && std::isnan(floats.y) && std::isnan(floats.z) &&
           std::isnan(floats.w) && error==1);++rejects;
  }
}
int main() {
  using namespace ascify::detail;
  const size_t page=size_t(sysconf(_SC_PAGESIZE));
  void *inaccessible=mmap(nullptr,page,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
  assert(inaccessible!=MAP_FAILED);
  uint32_t error=0;
  TextureSamplingDescriptor d{};
  d.data=reinterpret_cast<uint64_t>(inaccessible);d.width=d.height=d.layers=1;
  d.pitch_bytes=d.layer_bytes=4;d.error=reinterpret_cast<uint64_t>(&error);
  const auto token=reinterpret_cast<unsigned long long>(&d);
  for(bool layered:{false,true}) {
    d.layered=layered;d.abi_version=TextureDescriptorVersion;
    for(auto kind:{TextureSampleFloat,TextureSampleUnsigned,TextureSampleRGBAElement,TextureSampleRGBANormalized})
      for(auto filter:{TextureFilterPoint,TextureFilterLinear}) {
        d.unsigned_element=kind;d.linear=filter;oldReaders(token,error,layered);
      }
    for(unsigned oldKind:{0U,1U}) for(unsigned oldLinear:{0U,1U}) {
      // v1 allocator zero-initialized its entire upload image, including the
      // padding where v2 now requires its version field.
      d.unsigned_element=oldKind;d.linear=oldLinear;d.abi_version=0;
      newReaders(token,error,layered);
    }
    d.unsigned_element=TextureSampleRGBANormalized;d.linear=TextureFilterLinear;
    for(uint32_t invalidVersion:{0U,1U,0xffffffffU}) {
      d.abi_version=invalidVersion;newReaders(token,error,layered);
    }
    d.abi_version=TextureDescriptorVersion;
    for(uint32_t invalidKind:{0U,1U,0xffffffffU}) {
      d.unsigned_element=invalidKind;newReaders(token,error,layered);
    }
    d.unsigned_element=TextureSampleRGBANormalized;
    for(uint32_t invalidFilter:{0U,1U,0xffffffffU}) {
      d.linear=invalidFilter;newReaders(token,error,layered);
    }
  }
  assert(munmap(inaccessible,page)==0);
  std::printf("texture ABI v1/v2 and corrupted tags: %u actual-wrapper CPU rejections before protected texel loads; no NPU\n",rejects);
}
