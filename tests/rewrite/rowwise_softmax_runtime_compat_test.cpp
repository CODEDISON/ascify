#include <ascify/target/dav_c310/rowwise_simd_recipes.hpp>
#include <cassert>
#include <cstring>

extern "C" int test_softmax_launches();

int main(int argc, char** argv) {
  assert(argc == 2);
  const bool extended = std::strcmp(argv[1], "extended") == 0;
  namespace simd = ascify::target::dav_c310::rowwise_simd_v1;
  const auto stream = reinterpret_cast<aclrtStream>(0x1000);
  const auto input = reinterpret_cast<const half*>(0x10000);
  auto output = reinterpret_cast<half*>(0x1000000);

  auto result = simd::TrySoftmaxSimd(stream, input, output, 8, 4096);
  assert(result.handled && result.status == ACL_SUCCESS);
  assert(test_softmax_launches() == 1);

  result = simd::TrySoftmaxSimd(stream, input, output, 8, 64);
  assert(result.handled == extended && result.status == ACL_SUCCESS);
  assert(test_softmax_launches() == (extended ? 2 : 1));

  // The new header admits 128. Neither test provider supports this width.
  result = simd::TrySoftmaxSimd(stream, input, output, 8, 128);
  assert(!result.handled && result.status == ACL_SUCCESS);
  assert(test_softmax_launches() == (extended ? 2 : 1));

  // A real failure from a provider that claimed the call is final.
  result = simd::TrySoftmaxSimd(stream, input, output, 9, 64);
  assert(result.handled == extended);
  assert(result.status == (extended ? ACL_ERROR_RT_PARAM_INVALID : ACL_SUCCESS));
  assert(test_softmax_launches() == (extended ? 3 : 1));
}
