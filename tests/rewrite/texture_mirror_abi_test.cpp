// Exact frozen pre-mirror v2 reader bodies must reject mode2 before touching
// PROT_NONE texels. This is a CPU branch/ABI test, not target evidence.
#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#define __NPU_COMPILER_INTERNAL_PURE_SIMT__
#include <ascify/texture_compat.hpp>
namespace ascify_texture_v2 {
using ascify::atomic_or_global;
using cudaTextureObject_t=unsigned long long;
namespace detail {
template<typename T>using textureFetchType=ascify::detail::textureFetchType<T>;
template<typename T>using textureRGBAElementType=ascify::detail::textureRGBAElementType<T>;
template<typename T>using textureRGBAFloatType=ascify::detail::textureRGBAFloatType<T>;
template<typename T>using dependentFalse=ascify::detail::dependentFalse<T>;
}
}
#undef ASCIFY_TEXTURE_SAMPLING_HPP
#define ascify ascify_texture_v2
#include "fixtures/texture_abi_v2/texture_sampling.hpp"
#undef ascify
namespace ascify_texture_v2 {
#include "fixtures/texture_abi_v2/wrappers.inc"
}
#include <cassert>
#include <cmath>
#include <cstdio>
#include <sys/mman.h>
#include <unistd.h>

int main() {
  using namespace ascify::detail;
  const size_t bytes=size_t(sysconf(_SC_PAGESIZE));
  void *protected_data=mmap(nullptr,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
  assert(protected_data!=MAP_FAILED);
  TextureSamplingDescriptor d{};uint32_t error=0;unsigned rejects=0;
  d.data=reinterpret_cast<uint64_t>(protected_data);d.width=d.height=2;
  d.layers=1;d.normalized=1;d.pitch_bytes=8;d.layer_bytes=16;
  d.error=reinterpret_cast<uint64_t>(&error);d.abi_version=TextureDescriptorVersion;
  const auto t=reinterpret_cast<ascify::cudaTextureObject_t>(&d);
  for(bool layered:{false,true})for(bool x_axis:{false,true})for(bool fast:{false,true}) {
    d.layered=layered;d.wrap_x=x_axis?TextureAddressMirror:TextureAddressClamp;
    d.wrap_y=x_axis?TextureAddressClamp:TextureAddressMirror;
    d.linear=TextureFilterLinear;d.unsigned_element=TextureSampleFloat;error=0;
    const auto f=layered?(fast?ascify_texture_v2::tex2DLayeredFast<float>(t,0,0,0):ascify_texture_v2::tex2DLayered<float>(t,0,0,0)):
                          (fast?ascify_texture_v2::tex2DFast<float>(t,0,0):ascify_texture_v2::tex2D<float>(t,0,0));
    assert(std::isnan(f)&&error==1);++rejects;
    d.linear=TextureFilterPoint;d.unsigned_element=TextureSampleUnsigned;error=0;
    const auto u=layered?(fast?ascify_texture_v2::tex2DLayeredFast<unsigned int>(t,0,0,0):ascify_texture_v2::tex2DLayered<unsigned int>(t,0,0,0)):
                          (fast?ascify_texture_v2::tex2DFast<unsigned int>(t,0,0):ascify_texture_v2::tex2D<unsigned int>(t,0,0));
    assert(u==0&&error==1);++rejects;
    d.unsigned_element=TextureSampleRGBAElement;error=0;
    const auto c=layered?(fast?ascify_texture_v2::tex2DLayeredFast<uchar4>(t,0,0,0):ascify_texture_v2::tex2DLayered<uchar4>(t,0,0,0)):
                          (fast?ascify_texture_v2::tex2DFast<uchar4>(t,0,0):ascify_texture_v2::tex2D<uchar4>(t,0,0));
    assert(c.x==0&&c.y==0&&c.z==0&&c.w==0&&error==1);++rejects;
    d.linear=TextureFilterLinear;d.unsigned_element=TextureSampleRGBANormalized;error=0;
    const auto v=layered?(fast?ascify_texture_v2::tex2DLayeredFast<float4>(t,0,0,0):ascify_texture_v2::tex2DLayered<float4>(t,0,0,0)):
                          (fast?ascify_texture_v2::tex2DFast<float4>(t,0,0):ascify_texture_v2::tex2D<float4>(t,0,0));
    assert(std::isnan(v.x)&&std::isnan(v.y)&&std::isnan(v.z)&&std::isnan(v.w)&&error==1);++rejects;
  }
  assert(munmap(protected_data,bytes)==0);
  std::printf("frozen v2 reader mirror-mode CPU rejections=%u before protected texel loads; no NPU\n",rejects);
}
