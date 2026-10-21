// CPU arithmetic and rejection model of the actual helper; no device evidence.
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
static unsigned comparisons = 0;
static double max_point_error = 0, max_linear_error = 0;
static std::array<float,4> components(float4 v) { return {v.x,v.y,v.z,v.w}; }
static std::array<unsigned char,4> components(uchar4 v) { return {v.x,v.y,v.z,v.w}; }

// Independent address reference: double fmod and modulo, not the helper's
// float floor/subtract and single-edge wrap. The explicit float cast records
// the selected FP32 coordinate contract separately from double interpolation.
static double coordinate(float input, unsigned extent, bool normalized, bool wrap) {
  if (!normalized) return std::fmin(std::fmax(double(input),0.0),double(extent));
  double unit = wrap ? std::fmod(double(input),1.0) :
                      std::fmin(std::fmax(double(input),0.0),1.0);
  if (unit < 0) unit += 1.0;
  const float reduced = static_cast<float>(unit);
  return static_cast<float>(double(reduced) * double(extent));
}
static unsigned address(long index, unsigned extent, bool wrap) {
  if (!wrap) return unsigned(std::min(std::max(index,0L),long(extent)-1));
  return unsigned((index % long(extent) + long(extent)) % long(extent));
}
static std::array<double,4> reference(const std::vector<unsigned char> &data,
    unsigned width, unsigned height, unsigned pitch, unsigned layer, double x, double y,
    bool linear, bool wrap) {
  const long ix = long(std::floor(x - (linear ? .5 : 0.0)));
  const long iy = long(std::floor(y - (linear ? .5 : 0.0)));
  const double a = linear ? (x - .5) - ix : 0;
  const double b = linear ? (y - .5) - iy : 0;
  std::array<double,4> result{};
  // Direct four weighted terms in double: independent of product's FP32
  // two-row interpolation order and zero-weight short circuits.
  for (unsigned j=0;j!=2;++j) for(unsigned i=0;i!=2;++i) {
    const double weight=(i?a:1-a)*(j?b:1-b);
    const unsigned offset=layer*pitch*height+address(iy+j,height,wrap)*pitch+
                          address(ix+i,width,wrap)*4;
    for(unsigned c=0;c!=4;++c) result[c]+=weight*double(data[offset+c])/255.0;
  }
  return result;
}
static void compare(float4 actual, const std::array<double,4> &expected, bool linear) {
  const auto got=components(actual);
  for(unsigned c=0;c!=4;++c) {
    assert(std::isfinite(got[c]) && got[c]>=0 && got[c]<=1);
    const double error=std::fabs(double(got[c])-expected[c]);
    if(linear) { assert(error<=0x1p-21); max_linear_error=std::max(max_linear_error,error); }
    else {
      const float rounded=static_cast<float>(expected[c]);
      const double ulp=double(std::nextafter(rounded,INFINITY))-double(rounded);
      assert(std::fabs(double(got[c])-double(rounded))<=ulp);
      if(expected[c]==0 || expected[c]==1) assert(got[c]==expected[c]);
      max_point_error=std::max(max_point_error,error);
    }
    ++comparisons;
  }
}
int main() {
  using namespace ascify::detail;
  const auto channel=ascify::cudaCreateChannelDesc<uchar4>();
  assert(channel.x==8 && channel.y==8 && channel.z==8 && channel.w==8 &&
         channel.f==ascify::cudaChannelFormatKindUnsigned);
  const unsigned width=17,height=13,pitch=84,layers=3;
  std::vector<unsigned char> data(pitch*height*layers,0xa5);
  for(unsigned z=0;z!=layers;++z) for(unsigned y=0;y!=height;++y)
    for(unsigned x=0;x!=width;++x) for(unsigned c=0;c!=4;++c)
      data[z*pitch*height+y*pitch+x*4+c]=(x*19+y*37+z*101+c*71)&255;
  const auto original=data;
  uint32_t error=0;
  TextureSamplingDescriptor d{};
  d.data=reinterpret_cast<uint64_t>(data.data());d.pitch_bytes=pitch;
  d.layer_bytes=pitch*height;d.width=width;d.height=height;d.layers=layers;d.layered=1;
  d.abi_version=TextureDescriptorVersion;d.unsigned_element=TextureSampleRGBANormalized;
  d.error=reinterpret_cast<uint64_t>(&error);
  const auto token=reinterpret_cast<ascify::cudaTextureObject_t>(&d);
  const float limit=std::numeric_limits<float>::max();
  const std::array<float,11> plain={-limit,-19.25f,-0.0f,0.0f,.5f,.75f,1.0f,5.125f,16.5f,17.0f,limit};
  const std::array<float,11> normalized={-limit,-2.25f,-.5f,-0.0f,0.0f,.125f,.5f,.75f,1.0f,1.25f,limit};
  for(bool norm:{false,true}) for(bool wrap:{false,true}) {
    if(wrap&&!norm)continue;
    d.normalized=norm;d.wrap_x=d.wrap_y=wrap;
    for(bool linear:{false,true}) {
      d.linear=linear?TextureFilterLinear:TextureFilterPoint;
      for(unsigned layer=0;layer!=layers;++layer) for(float x:norm?normalized:plain)
        for(float y:norm?normalized:plain) {
          const auto expected=reference(data,width,height,pitch,layer,
              coordinate(x,width,norm,wrap),coordinate(y,height,norm,wrap),linear,wrap);
          compare(ascify::tex2DLayered<float4>(token,x,y,layer),expected,linear);
          compare(ascify::tex2DLayeredFast<float4>(token,x,y,layer),expected,linear);
          assert(error==0);
        }
    }
  }
  // Non-grid FP32 weights belong only to the explicit fast entry point.
  d.normalized=d.wrap_x=d.wrap_y=0;d.linear=TextureFilterLinear;
  compare(ascify::tex2DLayeredFast<float4>(token,.501f,.333f,1),
      reference(data,width,height,pitch,1,.501f,.333f,true,false),true);
  const auto rejected=components(ascify::tex2DLayered<float4>(token,.501f,.333f,1));
  for(float v:rejected)assert(std::isnan(v));assert(error==1);error=0;

  // Exhaust every byte value in every channel, including asymmetric alpha.
  std::vector<unsigned char> every(256*4);
  for(unsigned x=0;x!=256;++x)for(unsigned c=0;c!=4;++c)every[x*4+c]=(x+c*73)&255;
  d.data=reinterpret_cast<uint64_t>(every.data());d.width=256;d.height=d.layers=1;
  d.pitch_bytes=d.layer_bytes=every.size();d.layered=0;d.linear=TextureFilterPoint;
  for(unsigned x=0;x!=256;++x) {
    d.unsigned_element=TextureSampleRGBANormalized;
    compare(ascify::tex2D<float4>(token,x,0),reference(every,256,1,1024,0,x,0,false,false),false);
    compare(ascify::tex2DFast<float4>(token,x,0),reference(every,256,1,1024,0,x,0,false,false),false);
    d.unsigned_element=TextureSampleRGBAElement;
    for(bool fast:{false,true}) {
      const auto got=components(fast?ascify::tex2DFast<uchar4>(token,x,0):ascify::tex2D<uchar4>(token,x,0));
      for(unsigned c=0;c!=4;++c){assert(got[c]==every[x*4+c]);++comparisons;}
    }
  }
  // Predetermined coordinate/texel cases at maximum admitted extent use
  // distinct sparse sentinels, rather than the interpolation reference above.
  const unsigned big=1U<<20;
  std::vector<unsigned char> wide(big*4,0);
  wide[0]=11;wide[(big/2)*4]=33;wide[(big-1)*4]=55;
  d.data=reinterpret_cast<uint64_t>(wide.data());d.width=big;
  d.pitch_bytes=d.layer_bytes=wide.size();d.normalized=1;d.wrap_x=1;d.wrap_y=0;
  struct CoordinateCase { float x; unsigned char red; };
  for(const auto &q:std::array<CoordinateCase,8>{{{-0.0f,11},{0.0f,11},{1.0f,11},
      {-1.0f,11},{.5f,33},{-.5f,33},{limit,11},{-limit,11}}}) {
    const unsigned actual=ascify::tex2D<uchar4>(token,q.x,0).x;
    assert(actual==q.red);++comparisons;
    std::printf("coordinate_case normalized_wrap x=%a width=%u expected_red=%u actual_red=%u\n",
                double(q.x),big,unsigned(q.red),actual);
  }
  assert(ascify::tex2D<uchar4>(token,std::nextafter(1.0f,0.0f),0).x==55);
  assert(ascify::tex2D<uchar4>(token,-0x1p-20f,0).x==55);
  for(float tiny:{-0x1p-149f,std::nextafter(0.0f,-1.0f)}) {
    // Exact rational wrap is 1-2^-149, hence floor(width*wrap)=width-1.
    // FP32 subtraction instead becomes 1 and legacy wrap selects index 0.
    // This known unresolved equivalence is a sticky new-RGBA rejection.
    error=0;
    const auto rejected_tiny=components(ascify::tex2D<uchar4>(token,tiny,0));
    for(auto c:rejected_tiny)assert(c==0);assert(error==1);
    d.unsigned_element=TextureSampleRGBANormalized;error=0;
    const auto rejected_float=components(ascify::tex2DFast<float4>(token,tiny,0));
    for(auto c:rejected_float)assert(std::isnan(c));assert(error==1);
    d.unsigned_element=TextureSampleUnsigned;error=0;
    unsigned expected_first=0;std::memcpy(&expected_first,wide.data(),4);
    assert(ascify::tex2D<unsigned int>(token,tiny,0)==expected_first && error==0);
    std::printf("coordinate_boundary x=%a width=%u exact_rational_index=%u legacy_fp32_index=0 RGBA_rejected=1\n",
                double(tiny),big,big-1);
    d.unsigned_element=TextureSampleRGBAElement;
  }
  d.wrap_x=0;
  assert(ascify::tex2D<uchar4>(token,limit,0).x==55);
  assert(ascify::tex2D<uchar4>(token,-limit,0).x==11);
  assert(ascify::tex2D<uchar4>(token,1.0f,0).x==55);
  d.normalized=0;
  assert(ascify::tex2D<uchar4>(token,limit,0).x==55);
  assert(ascify::tex2D<uchar4>(token,-limit,0).x==11);
  assert(error==0);
  for(float bad:{INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()}) {
    error=0;const auto got=components(ascify::tex2D<uchar4>(token,bad,0));
    for(auto c:got)assert(c==0);assert(error==1);
  }
  d.layered=1;d.layers=1;
  for(auto kind:{TextureSampleRGBAElement,TextureSampleRGBANormalized}) {
    d.unsigned_element=kind;
    for(int layer:{-1,1}) {
      error=0;
      if(kind==TextureSampleRGBAElement) {
        const auto got=components(ascify::tex2DLayered<uchar4>(token,0,0,layer));
        for(auto c:got)assert(c==0);
      } else {
        const auto got=components(ascify::tex2DLayered<float4>(token,0,0,layer));
        for(auto c:got)assert(std::isnan(c));
      }
      assert(error==1);
    }
  }
  d.layered=0;d.unsigned_element=TextureSampleRGBANormalized;error=0;
  const auto mismatch=components(ascify::tex2D<uchar4>(token,0,0));
  for(auto c:mismatch)assert(c==0);assert(error==1);
  d.unsigned_element=TextureSampleRGBAElement;error=0;
  const auto inverse=components(ascify::tex2D<float4>(token,0,0));
  for(auto c:inverse)assert(std::isnan(c));assert(error==1);
  d.linear=TextureFilterLinear;error=0;
  const auto integer_linear=components(ascify::tex2DFast<uchar4>(token,0,0));
  for(auto c:integer_linear)assert(c==0);assert(error==1);
  assert(data==original);
  std::printf("RGBA CPU comparisons=%u max_point_abs=%.17g max_linear_abs=%.17g; "
              "independent effective-coordinate/interpolation and preset large-extent cases; no NPU\n",
              comparisons,max_point_error,max_linear_error);
}
