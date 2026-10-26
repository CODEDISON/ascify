// Host arithmetic/branch checks of the actual public wrappers. ACL and the
// sticky atomic are test doubles; this is not target/device evidence.
#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>

#define __NPU_COMPILER_INTERNAL_PURE_SIMT__
#include <ascify/texture_compat.hpp>

#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>

int main() {
  using ascify::detail::TextureSamplingDescriptor;
  const auto channel = ascify::cudaCreateChannelDesc<unsigned int>();
  assert(channel.x == 32 && channel.y == 0 && channel.z == 0 && channel.w == 0);
  assert(channel.f == ascify::cudaChannelFormatKindUnsigned);
  assert(ascify::detail::textureScalar32Channel(channel));
  auto bad_channel = channel;
  bad_channel.x = 16;
  assert(!ascify::detail::textureScalar32Channel(bad_channel));
  bad_channel = channel;
  bad_channel.y = 32;
  assert(!ascify::detail::textureScalar32Channel(bad_channel));
  bad_channel = channel;
  bad_channel.f = ascify::cudaChannelFormatKindSigned;
  assert(!ascify::detail::textureScalar32Channel(bad_channel));

  // Four columns, two rows, two layers. Padding and sentinels are deliberately
  // different from the texels. Large integers and float NaN/Inf bit patterns
  // detect a numeric cast or an accidental floating-point finiteness check.
  const unsigned int data[] = {
      0U, 1U, 16777217U, 0xffffffffU, 0xdeadbeefU, 0xdeadbeefU,
      0x80000000U, 0x7f800000U, 0x7fc00001U, 0x01000001U, 0xdeadbeefU, 0xdeadbeefU,
      101U, 102U, 103U, 104U, 0xdeadbeefU, 0xdeadbeefU,
      201U, 202U, 203U, 204U, 0xdeadbeefU, 0xdeadbeefU};
  unsigned int original[24];
  std::memcpy(original, data, sizeof(data));
  uint32_t error = 0;
  TextureSamplingDescriptor descriptor{};
  descriptor.data = reinterpret_cast<uint64_t>(data);
  descriptor.pitch_bytes = 6 * sizeof(unsigned int);
  descriptor.layer_bytes = 12 * sizeof(unsigned int);
  descriptor.width = 4;
  descriptor.height = 2;
  descriptor.layers = 1;
  descriptor.unsigned_element = ascify::detail::TextureSampleUnsigned;
  descriptor.abi_version = ascify::detail::TextureDescriptorVersion;
  descriptor.linear = ascify::detail::TextureFilterPoint;
  descriptor.error = reinterpret_cast<uint64_t>(&error);
  const auto token = reinterpret_cast<ascify::cudaTextureObject_t>(&descriptor);

  struct Query { float x, y; unsigned int expected; };
  const Query clamped[] = {
      {0.0f, 0.0f, 0U}, {1.0f, 0.0f, 1U}, {2.0f, 0.0f, 16777217U},
      {3.0f, 0.0f, 0xffffffffU}, {0.0f, 1.0f, 0x80000000U},
      {1.0f, 1.0f, 0x7f800000U}, {2.0f, 1.0f, 0x7fc00001U},
      {3.0f, 1.0f, 0x01000001U}, {-19.0f, 1.5f, 0x80000000U},
      {99.0f, 99.0f, 0x01000001U}, {1.999f, 0.999f, 1U}};
  for (const auto &q : clamped) {
    unsigned char before[sizeof(descriptor)];
    std::memcpy(before, &descriptor, sizeof(descriptor));
    assert(ascify::tex2D<unsigned int>(token, q.x, q.y) == q.expected);
    assert(ascify::tex2DFast<unsigned int>(token, q.x, q.y) == q.expected);
    assert(error == 0);
    assert(std::memcmp(before, &descriptor, sizeof(descriptor)) == 0);
  }

  descriptor.normalized = 1;
  descriptor.wrap_x = descriptor.wrap_y = 1;
  assert(ascify::tex2D<unsigned int>(token, -0.25f, 0.0f) == 0xffffffffU);
  assert(ascify::tex2DFast<unsigned int>(token, 1.5f, -0.5f) == 0x7fc00001U);
  descriptor.wrap_x = descriptor.wrap_y = 0;
  assert(ascify::tex2D<unsigned int>(token, 1.0f, 1.0f) == 0x01000001U);
  descriptor.normalized = 0;
  descriptor.layered = 1;
  descriptor.layers = 2;
  assert(ascify::tex2DLayered<unsigned int>(token, 2.0f, 1.0f, 1) == 203U);
  assert(ascify::tex2DLayeredFast<unsigned int>(token, -1.0f, 0.0f, 1) == 101U);
  assert(error == 0);

  auto rejects = [&](float x, float y, int layer) {
    error = 0;
    assert(ascify::tex2DLayered<unsigned int>(token, x, y, layer) == 0U);
    assert(error == 1);
    error = 0;
    assert(ascify::tex2DLayeredFast<unsigned int>(token, x, y, layer) == 0U);
    assert(error == 1);
  };
  rejects(0.0f, 0.0f, -1);
  rejects(0.0f, 0.0f, 2);
  rejects(std::numeric_limits<float>::infinity(), 0.0f, 0);
  rejects(0.0f, std::numeric_limits<float>::quiet_NaN(), 0);
  descriptor.linear = ascify::detail::TextureFilterLinear;
  rejects(0.0f, 0.0f, 0);
  descriptor.linear = ascify::detail::TextureFilterPoint;
  descriptor.width = 0;  // Existing released-backing invalidation contract.
  rejects(0.0f, 0.0f, 0);
  descriptor.width = 4;
  descriptor.unsigned_element = ascify::detail::TextureSampleFloat;
  rejects(0.0f, 0.0f, 0);
  descriptor.unsigned_element = ascify::detail::TextureSampleUnsigned;
  error = 0;
  assert(std::isnan(ascify::tex2DLayered<float>(token, 0.0f, 0.0f, 0)));
  assert(error == 1);
  error = 0;
  assert(std::isnan(ascify::tex2DLayeredFast<float>(token, 0.0f, 0.0f, 0)));
  assert(error == 1);
  error = 0;
  assert(ascify::tex2D<unsigned int>(token, 0.0f, 0.0f) == 0U);
  assert(error == 1);  // Wrong 2D wrapper for layered storage.
  assert(std::memcmp(data, original, sizeof(data)) == 0);

  // The float path uses the v2 float tag and preserves its numerical behavior.
  const float floats[] = {2.0f, 6.0f, 10.0f, 14.0f};
  descriptor = {};
  descriptor.data = reinterpret_cast<uint64_t>(floats);
  descriptor.pitch_bytes = 2 * sizeof(float);
  descriptor.layer_bytes = 4 * sizeof(float);
  descriptor.width = descriptor.height = 2;
  descriptor.layers = 1;
  descriptor.linear = ascify::detail::TextureFilterLinear;
  descriptor.unsigned_element = ascify::detail::TextureSampleFloat;
  descriptor.abi_version = ascify::detail::TextureDescriptorVersion;
  descriptor.error = reinterpret_cast<uint64_t>(&error);
  error = 0;
  assert(ascify::tex2D<float>(token, 1.0f, 1.0f) == 8.0f);
  assert(ascify::tex2DFast<float>(token, 1.0f, 1.0f) == 8.0f);
  assert(error == 0);
}
