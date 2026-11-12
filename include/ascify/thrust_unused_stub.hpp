// Ascify stub for <thrust/...> includes whose translation unit uses no
// thrust entity (proven by the boundary audit: no thrust:: symbol is
// referenced). The real libcudacxx/thrust template surface pulls the full
// cub stack, which this target does not compile; an unused include gets
// out of the way instead. Any actual thrust use fails loudly at the
// missing symbols.
#ifndef ASCIFY_THRUST_UNUSED_STUB_HPP
#define ASCIFY_THRUST_UNUSED_STUB_HPP
#endif  // ASCIFY_THRUST_UNUSED_STUB_HPP
