/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include <vector>

#include <chrono>
#include <string>
#include <fstream>
#include <map>
#include <set>
#include <list>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#if LLVM_VERSION_MAJOR > 3
#include "clang/Basic/Cuda.h"
#endif
#include "StringUtils.h"

namespace chr = std::chrono;

enum ConvTypes {
  //  driver::ERROR
  // runtime::ERROR
  CONV_ERROR = 0,
  //  driver::INIT
  CONV_INIT,
  //  driver::VERSION
  // runtime::VERSION
  CONV_VERSION,
  //  driver::DEVICE
  //  driver::DEVICE_DEPRECATED
  // runtime::DEVICE
  CONV_DEVICE,
  //  driver::PRIMARY_CONTEXT
  //  driver::CONTEXT
  //  driver::CONTEXT_DEPRECATED
  CONV_CONTEXT,
  //  driver::MODULE
  //  driver::MODULE_DEPRECATED
  CONV_MODULE,
  //  driver::LIBRARY
  CONV_LIBRARY,
  //  driver::MEMORY
  // runtime::MEMORY
  // runtime::MEMORY_DEPRECATED
  CONV_MEMORY,
  //  driver::VIRTUAL_MEMORY
  CONV_VIRTUAL_MEMORY,
  //  driver::ORDERED_MEMORY
  // runtime::ORDERED_MEMORY
  CONV_ORDERED_MEMORY,
  //  driver::MULTICAST
  CONV_MULTICAST,
  //  driver::UNIFIED
  // runtime::UNIFIED
  CONV_UNIFIED,
  //  driver::STREAM
  // runtime::STREAM
  CONV_STREAM,
  //  driver::EVENT
  // runtime::EVENT
  CONV_EVENT,
  //  driver::EXTERNAL_RES
  // runtime::EXTERNAL_RES
  CONV_EXTERNAL_RES,
  //  driver::STREAM_MEMORY
  CONV_STREAM_MEMORY,
  //  driver::EXECUTION
  //  driver::EXECUTION_DEPRECATED
  // runtime::EXECUTION
  // runtime::EXECUTION_REMOVED
  CONV_EXECUTION,
  //  driver::GRAPH
  // runtime::GRAPH
  CONV_GRAPH,
  //  driver::OCCUPANCY
  // runtime::OCCUPANCY
  CONV_OCCUPANCY,
  //  driver::TEXTURE_DEPRECATED
  //  driver::TEXTURE
  // runtime::TEXTURE
  // runtime::TEXTURE_REMOVED
  CONV_TEXTURE,
  //  driver::SURFACE_DEPRECATED
  //  driver::SURFACE
  // runtime::SURFACE
  // runtime::SURFACE_REMOVED
  CONV_SURFACE,
  //  driver::TENSOR
  CONV_TENSOR,
  //  driver::PEER
  // runtime::PEER
  CONV_PEER,
  //  driver::GRAPHICS
  // runtime::GRAPHICS
  CONV_GRAPHICS,
  //  driver::DRIVER_ENTRY_POINT
  // runtime::DRIVER_ENTRY_POINT
  CONV_DRIVER_ENTRY_POINT,
  // runtime::CPP
  CONV_CPP,
  //  driver::COREDUMP
  CONV_COREDUMP,
  //  driver::GREEN_CONTEXT
  CONV_GREEN_CONTEXT,
  //  driver::ERROR_LOG
  // runtime::ERROR_LOG
  CONV_ERROR_LOG,
  // runtime::DRIVER_INTERACT
  CONV_DRIVER_INTERACT,
  //  driver::PROFILER_DEPRECATED
  //  driver::PROFILER
  // runtime::PROFILER
  // runtime::PROFILER_REMOVED
  CONV_PROFILER,
  //  driver::OPENGL
  // runtime::OPENGL
  // runtime::OPENGL_DEPRECATED
  CONV_OPENGL,
  //  driver::D3D9
  // runtime::D3D9
  // runtime::D3D9_DEPRECATED
  CONV_D3D9,
  //  driver::D3D10
  // runtime::D3D10
  // runtime::D3D10_DEPRECATED
  CONV_D3D10,
  //  driver::D3D11
  // runtime::D3D11
  // runtime::D3D11_DEPRECATED
  CONV_D3D11,
  //  driver::VDPAU
  // runtime::VDPAU
  CONV_VDPAU,
  //  driver::EGL
  // runtime::EGL
  CONV_EGL,
  // runtime::THREAD_DEPRECATED
  CONV_THREAD,
  CONV_COMPLEX,
  CONV_LIB_FUNC,
  CONV_LIB_DEVICE_FUNC,
  CONV_DEVICE_FUNC,
  CONV_DEVICE_TYPE,
  CONV_INCLUDE,
  CONV_INCLUDE_CUDA_MAIN_H,
  CONV_INCLUDE_CUDA_MAIN_V2_H,
  CONV_TYPE,
  CONV_LITERAL,
  CONV_NUMERIC_LITERAL,
  CONV_DEFINE,
  CONV_EXTERN_SHARED,
  CONV_KERNEL_LAUNCH,
  CONV_LAST
};
constexpr int NUM_CONV_TYPES = (int) ConvTypes::CONV_LAST;

// TODO: DPP may not need to track all these as it does not have support to it
enum ApiTypes {
  API_DRIVER = 0,
  API_RUNTIME,
  API_COMPLEX,
  API_BLAS,
  API_RAND,
  API_DNN,
  API_FFT,
  API_SPARSE,
  API_SOLVER,
  API_CUB,
  API_RTC,
  API_TENSOR,
  API_FILE,
  API_LAST
};
constexpr int NUM_API_TYPES = (int) ApiTypes::API_LAST;

enum SupportDegree {
  FULL = 0x0,
  HIP_UNSUPPORTED = 0x1,
  UNSUPPORTED = 0x4,
  CUDA_DEPRECATED = 0x8,
  HIP_DEPRECATED = 0x10,
  DEPRECATED = 0x40,
  CUDA_REMOVED = 0x80,
  HIP_REMOVED = 0x100,
  REMOVED = 0x400,
  HIP_EXPERIMENTAL = 0x800,
  HIP_SUPPORTED_V2_ONLY = 0x1000,
  CUDA_OVERLOADED = 0x2000,
  HIP_PARTIALLY_SUPPORTED = 0x4000,
};

enum cudaVersions {
  CUDA_0 = 0, // Unknown version
  CUDA_10 = 1000,
  CUDA_11 = 1010,
  CUDA_20 = 2000,
  CUDA_21 = 2010,
  CUDA_22 = 2020,
  CUDA_23 = 2030,
  CUDA_30 = 3000,
  CUDA_31 = 3010,
  CUDA_32 = 3020,
  CUDA_40 = 4000,
  CUDA_41 = 4010,
  CUDA_42 = 4020,
  CUDA_50 = 5000,
  CUDA_55 = 5050,
  CUDA_60 = 6000,
  CUDA_65 = 6050,
  CUDA_70 = 7000,
  CUDA_75 = 7050,
  CUDA_80 = 8000,
  CUDA_90 = 9000,
  CUDA_91 = 9010,
  CUDA_92 = 9020,
  CUDA_100 = 10000,
  CUDA_101 = 10010,
  CUDA_102 = 10020,
  CUDA_110 = 11000,
  CUDA_111 = 11010,
  CUDA_112 = 11020,
  CUDA_113 = 11030,
  CUDA_114 = 11040,
  CUDA_115 = 11050,
  CUDA_116 = 11060,
  CUDA_117 = 11070,
  CUDA_118 = 11080,
  CUDA_120 = 12000,
  CUDA_121 = 12010,
  CUDA_122 = 12020,
  CUDA_123 = 12030,
  CUDA_124 = 12040,
  CUDA_125 = 12050,
  CUDA_126 = 12060,
  CUDA_128 = 12080,
  CUDA_129 = 12090,
  CUDA_130 = 13000,
  CUDA_131 = 13010,
  CUDA_LATEST = CUDA_129,
  CUDA_PARTIALLY_SUPPORTED = CUDA_130,
  CUDNN_10 = 100,
  CUDNN_20 = 200,
  CUDNN_30 = 300,
  CUDNN_40 = 400,
  CUDNN_50 = 500,
  CUDNN_51 = 510,
  CUDNN_60 = 600,
  CUDNN_704 = 704,
  CUDNN_705 = 705,
  CUDNN_712 = 712,
  CUDNN_713 = 713,
  CUDNN_714 = 714,
  CUDNN_721 = 721,
  CUDNN_730 = 730,
  CUDNN_731 = 731,
  CUDNN_741 = 741,
  CUDNN_742 = 742,
  CUDNN_750 = 750,
  CUDNN_751 = 751,
  CUDNN_760 = 760,
  CUDNN_761 = 761,
  CUDNN_762 = 762,
  CUDNN_763 = 763,
  CUDNN_764 = 764,
  CUDNN_765 = 765,
  CUDNN_801 = 801,
  CUDNN_802 = 802,
  CUDNN_803 = 803,
  CUDNN_804 = 804,
  CUDNN_805 = 805,
  CUDNN_810 = 810,
  CUDNN_811 = 811,
  CUDNN_820 = 820,
  CUDNN_830 = 830,
  CUDNN_840 = 840,
  CUDNN_850 = 850,
  CUDNN_860 = 860,
  CUDNN_870 = 870,
  CUDNN_880 = 880,
  CUDNN_881 = 881,
  CUDNN_890 = 890,
  CUDNN_891 = 891,
  CUDNN_892 = 892,
  CUDNN_893 = 893,
  CUDNN_894 = 894,
  CUDNN_895 = 895,
  CUDNN_896 = 896,
  CUDNN_897 = 897,
  CUDNN_900 = 900,
  CUDNN_910 = 910,
  CUDNN_920 = 920,
  CUDNN_930 = 930,
  CUDNN_940 = 940,
  CUDNN_950 = 950,
  CUDNN_960 = 960,
  CUDNN_970 = 970,
  CUDNN_980 = 980,
  CUDNN_990 = 990,
  CUDNN_9100 = 9100,
  CUDNN_9110 = 9110,
  CUDNN_9120 = 9120,
  CUDNN_9130 = 9130,
  CUDNN_9140 = 9140,
  CUDNN_9150 = 9150,
  CUDNN_9170 = 9170,
  CUDNN_9180 = 9180,
  CUDNN_9200 = 9200,
  CUDNN_LATEST = CUDNN_9200,
  CUTENSOR_1010 = 100010,
  CUTENSOR_1100 = 100100,
  CUTENSOR_1200 = 100200,
  CUTENSOR_1210 = 100210,
  CUTENSOR_1220 = 100220,
  CUTENSOR_1300 = 100300,
  CUTENSOR_1310 = 100310,
  CUTENSOR_1320 = 100320,
  CUTENSOR_1330 = 100330,
  CUTENSOR_1400 = 100400,
  CUTENSOR_1500 = 100500,
  CUTENSOR_1600 = 100600,
  CUTENSOR_1610 = 100610,
  CUTENSOR_1620 = 100620,
  CUTENSOR_1700 = 100700,
  CUTENSOR_2000 = 200000,
  CUTENSOR_2010 = 200010,
  CUTENSOR_2020 = 200020,
  CUTENSOR_2021 = 200021,
  CUTENSOR_2109 = 200109,
  CUTENSOR_2200 = 200200,
  CUTENSOR_2300 = 200300,
  CUTENSOR_2400 = 200400,
  CUTENSOR_2500 = 200500,
  CUTENSOR_LATEST = CUTENSOR_2500,
  CUFILE_1000 = 1000000,
  CUFILE_1001 = 1000001,
  CUFILE_1002 = 1000002,
  CUFILE_1010 = 1000010,
  CUFILE_1011 = 1000011,
  CUFILE_1020 = 1000020,
  CUFILE_1021 = 1000021,
  CUFILE_1030 = 1000030,
  CUFILE_1031 = 1000031,
  CUFILE_1040 = 1000040,
  CUFILE_1050 = 1000050,
  CUFILE_1051 = 1000051,
  CUFILE_1060 = 1000060,
  CUFILE_1061 = 1000061,
  CUFILE_1070 = 1000070,
  CUFILE_1071 = 1000071,
  CUFILE_1072 = 1000072,
  CUFILE_1080 = 1000080,
  CUFILE_1081 = 1000081,
  CUFILE_1090 = 1000090,
  CUFILE_1091 = 1000091,
  CUFILE_1100 = 1000100,
  CUFILE_1101 = 1000101,
  CUFILE_1110 = 1000110,
  CUFILE_1111 = 1000111,
  CUFILE_1130 = 1000130,
  CUFILE_1131 = 1000131,
  CUFILE_1140 = 1000140,
  CUFILE_1141 = 1000141,
  CUFILE_1150 = 1000150,
  CUFILE_1151 = 1000151,
  CUFILE_1160 = 1000160,
  CUFILE_1161 = 1000161,
  CUFILE_1170 = 1000170,
  CUFILE_LATEST = CUFILE_1170,
};

enum dppVersions {
  DPP_0 = 0, // Unknown version
  DPP_1050 = 1050,
  DPP_LATEST = 8000,
};

struct cudaAPIversions {
  cudaVersions appeared = CUDA_0;
  cudaVersions deprecated = CUDA_0;
  cudaVersions removed = CUDA_0;
};

struct dppAPIversions {
  dppVersions appeared = DPP_0;
  dppVersions deprecated = DPP_0;
  dppVersions removed = DPP_0;
  dppVersions experimental = DPP_0;
};

typedef std::list<dppVersions> dppAPIChangedVersions;
typedef std::list<cudaVersions> cudaAPIChangedVersions;
typedef std::list<cudaVersions> cudaAPIUnsupportedVersions;

// The names of various fields in in the statistics reports.
extern const char *counterNames[NUM_CONV_TYPES];
extern const char *counterTypes[NUM_CONV_TYPES];
extern const char *apiNames[NUM_API_TYPES];
extern const char *apiTypes[NUM_API_TYPES];

extern const std::vector<cudaVersions> CUDA_114_cuFile_version;
extern const std::vector<cudaVersions> CUDA_115_cuFile_version;
extern const std::vector<cudaVersions> CUDA_116_cuFile_version;
extern const std::vector<cudaVersions> CUDA_117_cuFile_version;
extern const std::vector<cudaVersions> CUDA_118_cuFile_version;
extern const std::vector<cudaVersions> CUDA_120_cuFile_version;
extern const std::vector<cudaVersions> CUDA_121_cuFile_version;
extern const std::vector<cudaVersions> CUDA_122_cuFile_version;
extern const std::vector<cudaVersions> CUDA_123_cuFile_version;
extern const std::vector<cudaVersions> CUDA_124_cuFile_version;
extern const std::vector<cudaVersions> CUDA_125_cuFile_version;
extern const std::vector<cudaVersions> CUDA_126_cuFile_version;
extern const std::vector<cudaVersions> CUDA_128_cuFile_version;
extern const std::vector<cudaVersions> CUDA_129_cuFile_version;
extern const std::vector<cudaVersions> CUDA_130_cuFile_version;
extern const std::vector<cudaVersions> CUDA_131_cuFile_version;
extern const std::vector<cudaVersions> Empty_vector;

struct dppCounter {
  llvm::StringRef dppName;
  ConvTypes type = CONV_ERROR;
  ApiTypes apiType = API_DRIVER;
  unsigned int apiSection = 1;
  unsigned int supportDegree = FULL;
};

/**
  * Tracks a set of named counters, as well as counters for each of the type enums defined above.
  */
class StatCounter {
private:
  // Each thing we track is either "supported" or "unsupported"...
  std::map<std::string, int> counters;
  int apiCounters[NUM_API_TYPES] = {};
  int convTypeCounters[NUM_CONV_TYPES] = {};

public:
  void incrementCounter(const dppCounter &counter, const std::string &name);
  // Add the counters from `other` onto the counters of this object.
  void add(const StatCounter &other);
  int getConvSum();
  void print(std::ostream *csv, llvm::raw_ostream *printOut, const std::string &prefix);
};

/**
  * Tracks the statistics for a single input file.
  */
class Statistics {
  friend class AscifyAction;
  StatCounter supported;
  StatCounter unsupported;
  std::string fileName;
  std::set<int> touchedLinesSet = {};
  unsigned touchedLines = 0;
  unsigned totalLines = 0;
  unsigned touchedBytes = 0;
  unsigned totalBytes = 0;
  chr::steady_clock::time_point startTime;
  chr::steady_clock::time_point completionTime;
  // CUDA Toolkit version provided by clang at runtime and converted to ASCIFY cudaVersions enum.
  static cudaVersions cudaVersionUsedByClang;
  static unsigned cudaVersion;

public:
  Statistics(const std::string &name);
  void incrementCounter(const dppCounter &counter, const std::string &name);
  // Add the counters from `other` onto the counters of this object.
  void add(const Statistics &other);
  void lineTouched(unsigned int lineNumber);
  void bytesChanged(unsigned int bytes);
  // Set the completion timestamp to now.
  void markCompletion();

public:
  /**
    * Pretty-print the statistics stored in this object.
    *
    * @param csv Pointer to an output stream for the CSV to write. If null, no CSV is written
    * @param printOut Pointer to an output stream to print human-readable textual stats to. If null, no
    *                 such stats are produced.
    */
  void print(std::ostream *csv, llvm::raw_ostream *printOut, bool skipHeader = false);
  // Print aggregated statistics for all registered counters.
  static void printAggregate(std::ostream *csv, llvm::raw_ostream *printOut);
  // The Statistics for each input file.
  static std::map<std::string, Statistics> stats;
  // The Statistics objects for the currently-being-processed input file.
  static Statistics *currentStatistics;
  // Aggregate statistics over all entries in `stats` and return the resulting Statistics object.
  static Statistics getAggregate();
  /**
    * Convenient global entry point for updating the "active" Statistics. Since we operate single-threadedly
    * processing one file at a time, this allows us to simply expose the stats for the current file globally,
    * simplifying things.
    */
  static Statistics &current();
  /**
    * Set the active Statistics object to the named one, creating it if necessary, and write the completion
    * timestamp into the currently active one.
    */
  static void setActive(const std::string &name);
  // Check whether the counter is HIP_EXPERIMENTAL or not.
  static bool isHipExperimental(const dppCounter &counter);
  // Check whether the counter is HIP_PARTIALLY_SUPPORTED or not.
  static bool isHipPartiallySupported(const dppCounter &counter);
  // Check whether the counter is HIP_UNSUPPORTED or not.
  static bool isHipUnsupported(const dppCounter &counter);
  // Check whether the counter is HIP_UNSUPPORTED/UNSUPPORTED or not.
  static bool isUnsupported(const dppCounter &counter);
  // Check whether the counter is CUDA_DEPRECATED or not.
  static bool isCudaDeprecated(const dppCounter &counter);
  // Check whether the counter is HIP_DEPRECATED or not.
  static bool isHipDeprecated(const dppCounter &counter);
  // Check whether the counter is DEPRECATED or not.
  static bool isDeprecated(const dppCounter &counter);
  // Check whether the counter is CUDA_REMOVED or not.
  static bool isCudaRemoved(const dppCounter &counter);
  // Check whether the counter is HIP_REMOVED or not.
  static bool isHipRemoved(const dppCounter &counter);
  // Check whether the counter is REMOVED or not.
  static bool isRemoved(const dppCounter &counter);
  // Check whether the counter is HIP_SUPPORTED_V2_ONLY or not.
  static bool isHipSupportedV2Only(const dppCounter &counter);
  // Check whether the counter is CUDA_OVERLOADED or not.
  static bool isCudaOverloaded(const dppCounter &counter);
  // Get string CUDA version.
  static std::string getCudaVersion(const cudaVersions &ver);
  // Get string DPP version.
  static std::string getDppVersion(const dppVersions &ver);
  // Get CUDA version by cuFile version.
  static cudaVersions getCudaVersionForCuFileVersion(const cudaVersions &ver);
  // Get cuFile versions by CUDA version.
  static const std::vector<cudaVersions> &getCuFileVersionsForCudaVersion(const cudaVersions &ver);
  // Set this flag in case of hipification errors.
  bool hasErrors = false;
#if LLVM_VERSION_MAJOR > 3
  // Converts CUDA version used by clang to CUDA version ASCIFY type.
  static cudaVersions convertCudaToolkitVersion(const clang::CudaVersion &ver);
#endif
  // Get CUDA version used by clang in the ASCIFY CUDA version format.
  static cudaVersions getCudaVersionUsedByClang() { return cudaVersionUsedByClang; }
  // Get actual CUDA version.
  static unsigned getCudaVersion() { return cudaVersion; }
  // Detects actual CUDA version based on the provided cuda.h file.
  static void setCudaVersion(llvm::StringRef cuda_h_file);
};
