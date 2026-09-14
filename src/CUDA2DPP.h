/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "Statistics.h"

const std::string sDPP_version = Statistics::getDppVersion(DPP_LATEST);

// Maps CUDA header names to DPP header names
extern const std::map<llvm::StringRef, dppCounter> CUDA_INCLUDE_MAP;
// Maps the names of CUDA DRIVER API types to the corresponding DPP types
extern const std::map<llvm::StringRef, dppCounter> CUDA_DRIVER_TYPE_NAME_MAP;
// Maps the names of CUDA DRIVER API functions to the corresponding DPP functions
extern const std::map<llvm::StringRef, dppCounter> CUDA_DRIVER_FUNCTION_MAP;
// Maps the names of CUDA RUNTIME API types to the corresponding DPP types
extern const std::map<llvm::StringRef, dppCounter> CUDA_RUNTIME_TYPE_NAME_MAP;
// Maps the names of CUDA Complex API types to the corresponding DPP types
extern const std::map<llvm::StringRef, dppCounter> CUDA_COMPLEX_TYPE_NAME_MAP;
// Maps the names of CUDA Complex API functions to the corresponding target functions
extern const std::map<llvm::StringRef, dppCounter> CUDA_COMPLEX_FUNCTION_MAP;
// Maps the names of CUDA RUNTIME API functions to the corresponding DPP functions
extern const std::map<llvm::StringRef, dppCounter> CUDA_RUNTIME_FUNCTION_MAP;
// Maps the names of CUDA Device types to the corresponding DPP types
extern const std::map<llvm::StringRef, dppCounter> CUDA_DEVICE_TYPE_NAME_MAP;
// Maps the names of CUDA Device functions to the corresponding DPP functions
extern const std::map<llvm::StringRef, dppCounter> CUDA_DEVICE_FUNCTION_MAP;
// Maps the names of cuFile API types to the corresponding DPP types
extern const std::map<llvm::StringRef, dppCounter> CUDA_FILE_TYPE_NAME_MAP;
// Maps the names of cuFile API functions to the corresponding DPP types
extern const std::map<llvm::StringRef, dppCounter> CUDA_FILE_FUNCTION_MAP;

/**
  * The union of all the above maps, except includes.
  *
  * This should be used rarely, but is still needed to convert macro definitions (which can
  * contain any combination of the above things). AST walkers can usually get away with just
  * looking in the lookup table for the type of element they are processing, however, saving
  * a great deal of time.
  */
const std::map<llvm::StringRef, dppCounter> &CUDA_RENAMES_MAP();


namespace runtime {
  enum CUDA_RUNTIME_API_SECTIONS {
    DEVICE = 1,
    MEMORY = 10,
    STREAM = 20,
    EVENT = 30,
    EXECUTION = 40,
    ERROR_HANDLING = 50,
    VERSION = 60,
    OCCUPANCY = 70,
    DEVICE_MGMT = 80,
  };
}