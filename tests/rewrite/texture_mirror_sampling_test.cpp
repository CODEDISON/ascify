// Actual public wrappers on CPU/test-double atomic; no CCEC or NPU evidence.
#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#define __NPU_COMPILER_INTERNAL_PURE_SIMT__
#include <ascify/texture_compat.hpp>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using ascify::detail::TextureSamplingDescriptor;
static unsigned comparisons=0, precise_rejections=0;
static double max_error=0;
// Independent double/fmod reference, rather than the product's abs/floor
// triangle. FP32 scaling is explicit: interpolation then uses double weights.
static float effective(float input,unsigned extent,unsigned mode=2) {
  double unit;
  if(mode==0)unit=std::fmin(std::fmax(double(input),0.0),1.0);
  else if(mode==1){unit=std::fmod(double(input),1.0);if(unit<0)unit+=1;}
  else {unit=std::fabs(std::fmod(double(input),2.0));if(unit>1)unit=2-unit;}
  return float(float(unit)*float(extent));
}
static unsigned clamp(long index,unsigned extent) {
  return unsigned(std::max(0L,std::min(index,long(extent)-1)));
}
static unsigned address(long index,unsigned extent,unsigned mode) {
  return mode==1?unsigned((index%long(extent)+long(extent))%long(extent)):clamp(index,extent);
}
static bool grid(float x,float y) {
  const double a=(double(x)-.5)-std::floor(double(x)-.5);
  const double b=(double(y)-.5)-std::floor(double(y)-.5);
  return a*256==std::floor(a*256) && b*256==std::floor(b*256);
}
template<typename Getter>
static double reference(float x,float y,unsigned w,unsigned h,bool linear,Getter get,
                        unsigned mx=2,unsigned my=2) {
  const double xx=double(x)-(linear?.5:0),yy=double(y)-(linear?.5:0);
  const long ix=long(std::floor(xx)),iy=long(std::floor(yy));
  if(!linear)return get(address(ix,w,mx),address(iy,h,my));
  const double a=xx-ix,b=yy-iy;
  double result=0;
  for(unsigned j=0;j<2;++j)for(unsigned i=0;i<2;++i)
    result+=(i?a:1-a)*(j?b:1-b)*get(address(ix+i,w,mx),address(iy+j,h,my));
  return result;
}
static void close(float got,double expected,double tolerance) {
  assert(std::isfinite(got));
  const double error=std::fabs(double(got)-expected);
  assert(error<=tolerance);max_error=std::max(max_error,error);++comparisons;
}
static std::array<float,4> parts(float4 v){return {v.x,v.y,v.z,v.w};}
static std::array<unsigned char,4> parts(uchar4 v){return {v.x,v.y,v.z,v.w};}

int main() {
  using namespace ascify::detail;
  const float limit=std::numeric_limits<float>::max();
  std::vector<float> q={-limit,-0x1p30f,-0x1p24f,-0x1p23f,-13.25f,-3,-2,-1.75f,-1,-.75f,-.125f,
                        -0x1p-149f,-0.0f,0,.125f,.25f,.5f,.75f,1,1.25f,1.5f,1.75f,2,3,13.25f,
                        0x1p23f,0x1p24f,0x1p30f,limit};
  for(float edge:{-2.0f,-1.0f,0.0f,1.0f,2.0f}) {
    q.push_back(std::nextafter(edge,-INFINITY));q.push_back(std::nextafter(edge,INFINITY));
  }
  uint32_t bits=0x12345678;
  for(unsigned i=0;i<12000;++i) {
    bits=bits*1664525U+1013904223U;float x;std::memcpy(&x,&bits,4);
    if(!std::isfinite(x))continue;
    for(unsigned extent:{1U,7U,1U<<20}) {
      const float actual=textureCoordinate(x,extent,true,TextureAddressMirror);
      assert(actual==effective(x,extent) && actual>=0 && actual<=float(extent));++comparisons;
    }
  }
  // Fixed endpoint truth table also distinguishes mirror from wrap/clamp.
  for(const auto &t:std::array<std::pair<float,float>,9>{{{-2,0},{-1,1},{-.25f,.25f},
      {-0.0f,0},{0,0},{.25f,.25f},{1,1},{1.25f,.75f},{2,0}}})
    assert(textureCoordinate(t.first,1,true,TextureAddressMirror)==t.second);

  const unsigned w=7,h=5,pitch=9,layers=2;
  std::vector<float> f(pitch*h*layers,-999.0f);
  std::vector<unsigned int> u(pitch*h*layers,0xdeadbeefU);
  std::vector<unsigned char> rgba(pitch*h*layers*4,0xa5);
  for(unsigned z=0;z<layers;++z)for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x) {
    const unsigned pos=z*pitch*h+y*pitch+x;
    f[pos]=float(z*31+y*7+x)*.125f;u[pos]=0x7fc00000U^(z*9001+y*101+x*19);
    for(unsigned c=0;c<4;++c)rgba[pos*4+c]=(z*71+y*23+x*17+c*47)&255;
  }
  const auto f_before=f;const auto u_before=u;const auto rgba_before=rgba;
  uint32_t error=0;
  TextureSamplingDescriptor d{};
  d.width=w;d.height=h;d.pitch_bytes=pitch*4;d.layer_bytes=pitch*h*4;
  d.normalized=1;d.wrap_x=d.wrap_y=TextureAddressMirror;
  d.abi_version=TextureDescriptorVersion;d.error=reinterpret_cast<uint64_t>(&error);
  const auto token=reinterpret_cast<ascify::cudaTextureObject_t>(&d);
  for(bool layered:{false,true}) {
    d.layered=layered;d.layers=layered?layers:1;
    for(unsigned z=0;z<d.layers;++z)for(bool fast:{false,true})
      for(bool linear:{false,true})for(float x:q)for(float y:q) {
        const float xx=effective(x,w),yy=effective(y,h);
        d.linear=linear?TextureFilterLinear:TextureFilterPoint;
        const bool reject=linear&&!fast&&!grid(xx,yy);
        d.data=reinterpret_cast<uint64_t>(f.data());d.unsigned_element=TextureSampleFloat;error=0;
        const float v=layered?(fast?ascify::tex2DLayeredFast<float>(token,x,y,z):ascify::tex2DLayered<float>(token,x,y,z)):
                               (fast?ascify::tex2DFast<float>(token,x,y):ascify::tex2D<float>(token,x,y));
        if(reject){assert(std::isnan(v)&&error==1);++precise_rejections;}
        else {close(v,reference(xx,yy,w,h,linear,[&](unsigned a,unsigned b){return double(f[z*pitch*h+b*pitch+a]);}),0x1p-17);assert(error==0);}
        d.data=reinterpret_cast<uint64_t>(rgba.data());d.unsigned_element=TextureSampleRGBANormalized;error=0;
        const auto v4=parts(layered?(fast?ascify::tex2DLayeredFast<float4>(token,x,y,z):ascify::tex2DLayered<float4>(token,x,y,z)):
                                      (fast?ascify::tex2DFast<float4>(token,x,y):ascify::tex2D<float4>(token,x,y)));
        for(unsigned c=0;c<4;++c) {
          if(reject)assert(std::isnan(v4[c]));
          else close(v4[c],reference(xx,yy,w,h,linear,[&](unsigned a,unsigned b){return double(rgba[(z*pitch*h+b*pitch+a)*4+c])/255.;}),0x1p-20);
        }
        if(reject){assert(error==1);++precise_rejections;}else assert(error==0);
        if(linear)continue;
        const unsigned pos=z*pitch*h+clamp(long(std::floor(yy)),h)*pitch+clamp(long(std::floor(xx)),w);
        d.data=reinterpret_cast<uint64_t>(u.data());d.unsigned_element=TextureSampleUnsigned;error=0;
        const auto ui=layered?(fast?ascify::tex2DLayeredFast<unsigned int>(token,x,y,z):ascify::tex2DLayered<unsigned int>(token,x,y,z)):
                               (fast?ascify::tex2DFast<unsigned int>(token,x,y):ascify::tex2D<unsigned int>(token,x,y));
        assert(ui==u[pos]&&error==0);++comparisons;
        d.data=reinterpret_cast<uint64_t>(rgba.data());d.unsigned_element=TextureSampleRGBAElement;error=0;
        const auto uv=parts(layered?(fast?ascify::tex2DLayeredFast<uchar4>(token,x,y,z):ascify::tex2DLayered<uchar4>(token,x,y,z)):
                                      (fast?ascify::tex2DFast<uchar4>(token,x,y):ascify::tex2D<uchar4>(token,x,y)));
        for(unsigned c=0;c<4;++c){assert(uv[c]==rgba[pos*4+c]);++comparisons;}assert(error==0);
      }
  }
  // Independent axes: mirror paired with clamp or wrap in either order,
  // including half-texel seams. Keep high-bit U32 and asymmetric RGBA payloads.
  d.layered=0;d.layers=1;
  const std::array<float,11> mixed={-.75f,-.25f,-0.0f,0,.125f,.5f,.75f,1,1.25f,1.75f,2};
  for(auto modes:std::array<std::pair<unsigned,unsigned>,4>{{
      {TextureAddressMirror,TextureAddressClamp},{TextureAddressClamp,TextureAddressMirror},
      {TextureAddressMirror,TextureAddressWrap},{TextureAddressWrap,TextureAddressMirror}}}) {
    d.wrap_x=modes.first;d.wrap_y=modes.second;
    for(float x:mixed)for(float y:mixed)for(bool linear:{false,true}) {
      const float xx=effective(x,w,d.wrap_x),yy=effective(y,h,d.wrap_y);
      d.linear=linear?TextureFilterLinear:TextureFilterPoint;
      d.data=reinterpret_cast<uint64_t>(f.data());d.unsigned_element=TextureSampleFloat;error=0;
      close(ascify::tex2DFast<float>(token,x,y),reference(xx,yy,w,h,linear,
          [&](unsigned a,unsigned b){return double(f[b*pitch+a]);},d.wrap_x,d.wrap_y),0x1p-17);
      assert(error==0);
      d.data=reinterpret_cast<uint64_t>(rgba.data());d.unsigned_element=TextureSampleRGBANormalized;error=0;
      const auto v=parts(ascify::tex2DFast<float4>(token,x,y));
      for(unsigned c=0;c<4;++c)close(v[c],reference(xx,yy,w,h,linear,
          [&](unsigned a,unsigned b){return double(rgba[(b*pitch+a)*4+c])/255.;},d.wrap_x,d.wrap_y),0x1p-20);
      assert(error==0);if(linear)continue;
      const unsigned pos=address(long(std::floor(yy)),h,d.wrap_y)*pitch+address(long(std::floor(xx)),w,d.wrap_x);
      d.data=reinterpret_cast<uint64_t>(u.data());d.unsigned_element=TextureSampleUnsigned;error=0;
      assert(ascify::tex2D<unsigned int>(token,x,y)==u[pos]&&error==0);++comparisons;
      d.data=reinterpret_cast<uint64_t>(rgba.data());d.unsigned_element=TextureSampleRGBAElement;error=0;
      const auto c=parts(ascify::tex2D<uchar4>(token,x,y));
      for(unsigned i=0;i<4;++i){assert(c[i]==rgba[pos*4+i]);++comparisons;}assert(error==0);
    }
  }
  // Rejection stays sticky and precedes any texel load, including mode2 with
  // unnormalized coordinates, mode3, invalidlayer, and nonfinite coordinates.
  d.data=0;d.unsigned_element=TextureSampleFloat;d.linear=TextureFilterPoint;
  d.layered=0;d.layers=1;d.wrap_x=d.wrap_y=TextureAddressMirror;
  auto bad=[&](float x,float y){error=0;assert(std::isnan(ascify::tex2DFast<float>(token,x,y))&&error==1);};
  for(float nonfinite:{INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()})bad(nonfinite,0);
  d.normalized=0;bad(0,0);d.normalized=1;d.wrap_x=3;bad(0,0);d.wrap_x=TextureAddressMirror;
  error=0;assert(std::isnan(ascify::tex2DLayeredFast<float>(token,0,0,-1))&&error==1);
  d.layered=1;error=0;assert(std::isnan(ascify::tex2DLayeredFast<float>(token,0,0,1))&&error==1);
  d.width=0;bad(0,0);
  assert(f==f_before&&u==u_before&&rgba==rgba_before);
  std::printf("mirror CPU comparisons=%u precise_rejections=%u max_abs_error=%.17g; double/fmod and preset reference; no NPU\n",
              comparisons,precise_rejections,max_error);
}
