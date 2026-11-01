// Frozen original host function reference vs named target, including stdout.
// The ACL/SIMT headers below are CPU stubs; no device or CUDA Samples run.
#define __aicore__
#include <ascify/ascify_cuda_compat.hpp>
#include <cassert>
#include <climits>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace original_v130 {
#include "fixtures/sample_sm_cores/v13_0_function.inc"
}
namespace original_v134 {
#include "fixtures/sample_sm_cores/v13_4_function.inc"
}

struct Result { std::vector<int> values; std::string output; };
using Function = int (*)(int,int);
static Result capture(Function function, const std::vector<std::pair<int,int>> &inputs) {
  std::fflush(stdout);
  FILE *temporary=std::tmpfile();assert(temporary);
  const int saved=dup(fileno(stdout));assert(saved>=0);
  assert(dup2(fileno(temporary),fileno(stdout))>=0);
  Result result;
  for(const auto &input:inputs)result.values.push_back(function(input.first,input.second));
  assert(std::fflush(stdout)==0);
  assert(std::fseek(temporary,0,SEEK_SET)==0);
  char bytes[4096];size_t count;
  while((count=std::fread(bytes,1,sizeof(bytes),temporary)))result.output.append(bytes,count);
  assert(!std::ferror(temporary));
  assert(dup2(saved,fileno(stdout))>=0&&close(saved)==0&&std::fclose(temporary)==0);
  return result;
}
int main() {
  std::vector<std::pair<int,int>> inputs;
  // Complete canonical SM range, known/unknown values, and noncanonical
  // minor values whose signed shift/add operations are still defined.
  for(int major=0;major<=20;++major)for(int minor=-16;minor<=31;++minor)
    inputs.push_back({major,minor});
  inputs.push_back({0,INT_MIN});inputs.push_back({0,INT_MAX});
  inputs.push_back({INT_MAX>>4,15});
  unsigned comparisons=0;
  for(auto functions:std::vector<std::pair<Function,Function>>{
      {original_v130::_ConvertSMVer2Cores,ascify::sampleSMVersionToCoresV130},
      {original_v134::_ConvertSMVer2Cores,ascify::sampleSMVersionToCoresV134}}) {
    const auto original=capture(functions.first,inputs),target=capture(functions.second,inputs);
    assert(original.values==target.values&&original.output==target.output);
    comparisons+=unsigned(inputs.size());
  }
  // Same return but deliberately different stdout: merging tables fails this.
  const auto old=capture(ascify::sampleSMVersionToCoresV130,{{10,7}});
  const auto recent=capture(ascify::sampleSMVersionToCoresV134,{{10,7}});
  assert(old.values==std::vector<int>{128}&&recent.values==old.values);
  assert(old.output=="MapSMtoCores for SM 10.7 is undefined.  Default to use 128 Cores/SM\n");
  assert(recent.output.empty());
  // Both explicitly preserve the existing capability-epoch lookup value.
  assert(ascify::sampleSMVersionToCoresV130(7,0)==64);
  assert(ascify::sampleSMVersionToCoresV134(7,0)==64);
  int major=7,minor=0;
  assert(ascify::sampleSMVersionToCoresV130(major++,minor++)==64&&major==8&&minor==1);
  major=7;minor=0;
  assert(ascify::sampleSMVersionToCoresV134(major++,minor++)==64&&major==8&&minor==1);
  std::printf("frozen SM host lookup: %u return/stdout differential cases, version-specific0xa7 and single evaluation passed; no NPU core-count claim\n",comparisons);
}
