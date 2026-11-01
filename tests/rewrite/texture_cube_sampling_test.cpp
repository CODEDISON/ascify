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
namespace ascify_texture_v1 {
using ascify::atomic_or_global;
using cudaTextureObject_t=unsigned long long;
namespace detail {
template<typename T>using dependentFalse=ascify::detail::dependentFalse<T>;
}
}
#undef ASCIFY_TEXTURE_SAMPLING_HPP
#define ascify ascify_texture_v1
#include "fixtures/texture_abi_v1/texture_sampling.hpp"
#undef ascify
namespace ascify_texture_v1 {
#include "fixtures/texture_abi_v1/wrappers.inc"
}

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
using namespace ascify::detail;
using Direction=std::array<float,3>;
static unsigned checks=0,rejects=0,legacy_rejects=0;
static Direction inverse(unsigned face,float s,float t) {
  // Independently transcribed original sample's inverse face orientations.
  switch(face) {
    case 0:return {1,-t,-s};case 1:return {-1,-t,s};
    case 2:return {s,1,t};case 3:return {s,-1,-t};
    case 4:return {s,-t,1};default:return {-s,-t,-1};
  }
}
static float reference(const std::vector<float>&data,unsigned n,Direction d) {
  // Double Table6 projection, separate from production FP32 center proof.
  const double a[]={std::fabs(double(d[0])),std::fabs(double(d[1])),std::fabs(double(d[2]))};
  unsigned axis=0;for(unsigned i=1;i<3;++i)if(a[i]>a[axis])axis=i;
  assert(a[axis]>a[(axis+1)%3]&&a[axis]>a[(axis+2)%3]);
  const unsigned face=2*axis+(d[axis]<0);
  const unsigned si[]={2,2,0,0,0,0},ti[]={1,1,2,2,1,1};
  const int ss[]={-1,1,1,1,1,-1},ts[]={-1,-1,1,-1,-1,-1};
  const double u=(ss[face]*double(d[si[face]])/a[axis]+1)*.5;
  const double v=(ts[face]*double(d[ti[face]])/a[axis]+1)*.5;
  const unsigned x=unsigned(std::floor(u*n)),y=unsigned(std::floor(v*n));
  assert(x<n&&y<n);return data[face*n*n+y*n+x];
}
static TextureSamplingDescriptor descriptor(std::vector<float>&data,unsigned n,uint32_t&error) {
  TextureSamplingDescriptor d{};d.data=reinterpret_cast<uint64_t>(data.data());
  d.width=d.height=n;d.layers=6;d.pitch_bytes=n*4;d.layer_bytes=n*n*4;
  d.normalized=1;d.layered=TextureTopologyCube;d.unsigned_element=TextureSampleCubeFloat;
  d.abi_version=TextureDescriptorVersion;d.error=reinterpret_cast<uint64_t>(&error);
  d.linear=TextureFilterLinear;return d;
}
static float fetch(TextureSamplingDescriptor&d,Direction v,bool fast) {
  auto token=reinterpret_cast<ascify::cudaTextureObject_t>(&d);
  return fast?ascify::texCubemapFast<float>(token,v[0],v[1],v[2]):
              ascify::texCubemap<float>(token,v[0],v[1],v[2]);
}
static void reject(TextureSamplingDescriptor&d,Direction v,uint32_t&error) {
  for(bool fast:{false,true}){error=0;assert(std::isnan(fetch(d,v,fast))&&error==1);++rejects;}
}
int main() {
  for(unsigned n:{1U,2U,4U,8U,64U}) {
    std::vector<float> data(6*n*n);
    for(unsigned face=0;face<6;++face)for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x)
      data[face*n*n+y*n+x]=float(100000*face+257*y+3*x);
    uint32_t error=0;auto d=descriptor(data,n,error);
    for(uint32_t mx:{TextureAddressClamp,TextureAddressWrap,TextureAddressMirror})
    for(uint32_t my:{TextureAddressClamp,TextureAddressWrap,TextureAddressMirror})
    for(uint32_t filter:{TextureFilterPoint,TextureFilterLinear})for(bool fast:{false,true}) {
      d.wrap_x=mx;d.wrap_y=my;d.linear=filter;
      for(unsigned face=0;face<6;++face)for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x) {
        const float s=float(2*int(x)+1-int(n))/float(n),t=float(2*int(y)+1-int(n))/float(n);
        const auto v=inverse(face,s,t);
        assert(fetch(d,v,fast)==reference(data,n,v)&&error==0);++checks;
      }
    }
    // Actual original default input/gold, all six faces, no substitute copy.
    if(n==64) {
      for(unsigned i=0;i<data.size();++i)data[i]=float(i);
      for(unsigned face=0;face<6;++face)for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x) {
        const auto v=inverse(face,((x+.5f)/n)*2-1,((y+.5f)/n)*2-1);
        assert(-fetch(d,v,false)==-float(face*n*n+y*n+x));++checks;
      }
    }
  }
  // Upper index bound and exact inverse proof, without allocating huge images.
  for(unsigned n:{1U,2U,64U,1U<<20})for(unsigned i:{0U,n/2,n-1}) {
    int out=-1;const float s=float(2*int(i)+1-int(n))/float(n);
    assert(textureCubeCenterIndex(s,n,out)&&out==int(i));++checks;
    for(float bad:{std::nextafter(s,-INFINITY),std::nextafter(s,INFINITY)}) {
      assert(!textureCubeCenterIndex(bad,n,out));++checks;
    }
  }
  const size_t page=size_t(sysconf(_SC_PAGESIZE));
  void* protected_data=mmap(nullptr,page,PROT_NONE,MAP_PRIVATE|MAP_ANON,-1,0);
  assert(protected_data!=MAP_FAILED);
  std::vector<float> data(24);uint32_t error=0;auto base=descriptor(data,2,error);
  base.data=reinterpret_cast<uint64_t>(protected_data);
  const float nan=std::numeric_limits<float>::quiet_NaN(),inf=INFINITY;
  for(Direction v:std::vector<Direction>{{0,0,0},{-0.f,0,-0.f},{1,1,0},{-1,-1,0},
       {1,0,-1},{0,1,1},{2,0,0},{.5f,0,0},{1,0,0},{1,.25f,.5f},
       {1,std::nextafter(.5f,inf),-.5f},{1,std::nextafter(.5f,-inf),-.5f}}) {
    auto d=base;reject(d,v,error);
  }
  for(unsigned axis=0;axis<3;++axis)for(float bad:{nan,inf,-inf}) {
    Direction v={1,.5f,.5f};v[axis]=bad;auto d=base;reject(d,v,error);
  }
  // Each malformed public header is rejected before touching protected texels.
  for(unsigned mutation=0;mutation<10;++mutation) {
    auto d=base;
    switch(mutation){case 0:d.width=d.height=3;break;case 1:d.height=1;break;
      case 2:d.layers=5;break;case 3:d.normalized=0;break;case 4:d.layered=TextureTopologyLayered;break;
      case 5:d.unsigned_element=TextureSampleFloat;break;case 6:++d.abi_version;break;
      case 7:d.wrap_x=3;break;case 8:d.linear=0;break;case 9:d.width=d.height=(1U<<20)+1;break;}
    reject(d,{1,.5f,.5f},error);
  }
  // Frozen actual old v1/v2 wrappers and current 2D/layered helpers cannot
  // reinterpret Cube as a six-layer ordinary texture, even for valid centers.
  auto d=base;auto token=reinterpret_cast<ascify::cudaTextureObject_t>(&d);
  for(bool layered:{false,true})for(bool fast:{false,true}) {
    error=0;const float old1=layered?(fast?ascify_texture_v1::tex2DLayeredFast<float>(token,.25f,.25f,0):ascify_texture_v1::tex2DLayered<float>(token,.25f,.25f,0)):
      (fast?ascify_texture_v1::tex2DFast<float>(token,.25f,.25f):ascify_texture_v1::tex2D<float>(token,.25f,.25f));
    assert(std::isnan(old1)&&error==1);++legacy_rejects;
    error=0;const float old2=layered?(fast?ascify_texture_v2::tex2DLayeredFast<float>(token,.25f,.25f,0):ascify_texture_v2::tex2DLayered<float>(token,.25f,.25f,0)):
      (fast?ascify_texture_v2::tex2DFast<float>(token,.25f,.25f):ascify_texture_v2::tex2D<float>(token,.25f,.25f));
    assert(std::isnan(old2)&&error==1);++legacy_rejects;
    error=0;const float now=layered?(fast?ascify::tex2DLayeredFast<float>(token,.25f,.25f,0):ascify::tex2DLayered<float>(token,.25f,.25f,0)):
      (fast?ascify::tex2DFast<float>(token,.25f,.25f):ascify::tex2D<float>(token,.25f,.25f));
    assert(std::isnan(now)&&error==1);++legacy_rejects;
    error=0;const auto u1=layered?(fast?ascify_texture_v1::tex2DLayeredFast<unsigned int>(token,0,0,0):ascify_texture_v1::tex2DLayered<unsigned int>(token,0,0,0)):
      (fast?ascify_texture_v1::tex2DFast<unsigned int>(token,0,0):ascify_texture_v1::tex2D<unsigned int>(token,0,0));
    assert(u1==0&&error==1);++legacy_rejects;
    error=0;const auto u2=layered?(fast?ascify_texture_v2::tex2DLayeredFast<unsigned int>(token,0,0,0):ascify_texture_v2::tex2DLayered<unsigned int>(token,0,0,0)):
      (fast?ascify_texture_v2::tex2DFast<unsigned int>(token,0,0):ascify_texture_v2::tex2D<unsigned int>(token,0,0));
    assert(u2==0&&error==1);++legacy_rejects;
    error=0;const auto c=layered?(fast?ascify_texture_v2::tex2DLayeredFast<uchar4>(token,0,0,0):ascify_texture_v2::tex2DLayered<uchar4>(token,0,0,0)):
      (fast?ascify_texture_v2::tex2DFast<uchar4>(token,0,0):ascify_texture_v2::tex2D<uchar4>(token,0,0));
    assert(c.x==0&&c.y==0&&c.z==0&&c.w==0&&error==1);++legacy_rejects;
    error=0;const auto f=layered?(fast?ascify_texture_v2::tex2DLayeredFast<float4>(token,0,0,0):ascify_texture_v2::tex2DLayered<float4>(token,0,0,0)):
      (fast?ascify_texture_v2::tex2DFast<float4>(token,0,0):ascify_texture_v2::tex2D<float4>(token,0,0));
    assert(std::isnan(f.x)&&std::isnan(f.y)&&std::isnan(f.z)&&std::isnan(f.w)&&error==1);++legacy_rejects;
  }
  // Also isolate new readers from old ordinary/layered descriptor kinds.
  for(uint32_t kind:{TextureSampleFloat,TextureSampleUnsigned,TextureSampleRGBAElement,TextureSampleRGBANormalized}) {
    auto bad=base;bad.unsigned_element=kind;reject(bad,{1,.5f,.5f},error);
  }
  assert(munmap(protected_data,page)==0);
  std::vector<float> one(6,1);auto finite=descriptor(one,1,error);
  error=0;assert(fetch(finite,{1,-0.f,0.f},false)==1&&error==0);++checks;
  for(float bad:{nan,inf,-inf}){one[0]=bad;reject(finite,{1,0,0},error);}
  std::printf("cube centers CPU checks=%u sticky_rejections=%u frozen/current_reader_PROT_NONE_rejections=%u; no NPU\n",checks,rejects,legacy_rejects);
}
