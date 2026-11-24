cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING
    "dav-3510 ASC architecture")
if(NOT CMAKE_ASC_ARCHITECTURES STREQUAL "dav-3510")
  message(FATAL_ERROR
      "the rowwise SIMD runtime supports only dav-3510/950PR")
endif()

find_package(ASC REQUIRED)
project(ascify950_rowwise_simd_runtime_v1 LANGUAGES ASC CXX)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ASCIFY950_RUNTIME_ABI_VERSION 1 CACHE STRING
    "rowwise SIMD ABI version")
if(NOT ASCIFY950_RUNTIME_ABI_VERSION STREQUAL "1")
  message(FATAL_ERROR "only rowwise SIMD ABI version 1 is available")
endif()

set(ASCIFY950_RUNTIME_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(ASCIFY950_PUBLIC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../include")
set(ASCIFY950_CANN_LIBRARY_DIRS
    "$ENV{ASCEND_HOME_PATH}/lib64"
    "$ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64")
set(ASCIFY950_ASC_OPTIONS
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    $<$<COMPILE_LANGUAGE:ASC>:-O2>)
set(ASCIFY950_ASC_RUNTIME_LIBRARIES
    ascendc_runtime
    ascendcl
    runtime
    register
    error_manager
    profapi
    ge_common_base
    unified_dlog
    mmpa
    dl
    ascend_dump
    c_sec
    stdc++
    m)

function(ascify950_add_rowwise_dso target output source)
  add_library(${target} SHARED "${source}")
  target_include_directories(${target} PRIVATE
      "${ASCIFY950_RUNTIME_SOURCE_DIR}"
      "${ASCIFY950_PUBLIC_INCLUDE_DIR}")
  target_compile_options(${target} PRIVATE ${ASCIFY950_ASC_OPTIONS})
  target_link_directories(${target} PRIVATE ${ASCIFY950_CANN_LIBRARY_DIRS})
  target_link_libraries(${target} PRIVATE ${ASCIFY950_ASC_RUNTIME_LIBRARIES})
  set_target_properties(${target} PROPERTIES
      OUTPUT_NAME "${output}"
      VERSION 1.0.0
      SOVERSION 1
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      BUILD_RPATH
          "$ENV{ASCEND_HOME_PATH}/lib64;$ENV{ASCEND_HOME_PATH}/x86_64-linux/lib64"
      BUILD_WITH_INSTALL_RPATH FALSE
      INSTALL_RPATH ""
      INSTALL_RPATH_USE_LINK_PATH FALSE)
endfunction()

# Each launch remains in a separate ELF/device-registration domain.
ascify950_add_rowwise_dso(
    softmax_reg_recompute
    ascify950_softmax_reg_recompute_v1
    "${ASCIFY950_RUNTIME_SOURCE_DIR}/softmax_reg_recompute_entry.asc")
ascify950_add_rowwise_dso(
    rmsnorm_reg_cached
    ascify950_rmsnorm_reg_cached_v1
    "${ASCIFY950_RUNTIME_SOURCE_DIR}/rmsnorm_reg_cached_entry.asc")
ascify950_add_rowwise_dso(
    rmsnorm_reg_plain_rowbatch
    ascify950_rmsnorm_reg_plain_rowbatch_v1
    "${ASCIFY950_RUNTIME_SOURCE_DIR}/rmsnorm_reg_plain_rowbatch_entry.asc")
ascify950_add_rowwise_dso(
    layernorm_reg_cached
    ascify950_layernorm_reg_cached_v1
    "${ASCIFY950_RUNTIME_SOURCE_DIR}/layernorm_reg_cached_entry.asc")

set_target_properties(softmax_reg_recompute PROPERTIES
    EXPORT_NAME softmax_reg_recompute_v1)
set_target_properties(rmsnorm_reg_cached PROPERTIES
    EXPORT_NAME rmsnorm_reg_cached_v1)
set_target_properties(rmsnorm_reg_plain_rowbatch PROPERTIES
    EXPORT_NAME rmsnorm_reg_plain_rowbatch_v1)
set_target_properties(layernorm_reg_cached PROPERTIES
    EXPORT_NAME layernorm_reg_cached_v1)

add_library(ascify950_rowwise_simd_v1 INTERFACE)
add_library(Ascify950::rowwise_simd_v1 ALIAS ascify950_rowwise_simd_v1)
set_target_properties(ascify950_rowwise_simd_v1 PROPERTIES
    EXPORT_NAME rowwise_simd_v1)
target_include_directories(ascify950_rowwise_simd_v1 INTERFACE
    $<BUILD_INTERFACE:${ASCIFY950_PUBLIC_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(ascify950_rowwise_simd_v1 INTERFACE
    softmax_reg_recompute
    rmsnorm_reg_cached
    rmsnorm_reg_plain_rowbatch
    layernorm_reg_cached
    ${CMAKE_DL_LIBS})

install(TARGETS
    softmax_reg_recompute
    rmsnorm_reg_cached
    rmsnorm_reg_plain_rowbatch
    layernorm_reg_cached
    ascify950_rowwise_simd_v1
    EXPORT Ascify950RowwiseSimdV1Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES
    "${ASCIFY950_PUBLIC_INCLUDE_DIR}/ascify/target/dav_c310/rowwise_simd_abi.h"
    "${ASCIFY950_PUBLIC_INCLUDE_DIR}/ascify/target/dav_c310/rowwise_hybrid_registry_v1.hpp"
    "${ASCIFY950_PUBLIC_INCLUDE_DIR}/ascify/target/dav_c310/rowwise_simd_selectors_v1.hpp"
    "${ASCIFY950_PUBLIC_INCLUDE_DIR}/ascify/target/dav_c310/rowwise_simd_recipes.hpp"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ascify/target/dav_c310)
install(EXPORT Ascify950RowwiseSimdV1Targets
    FILE Ascify950RowwiseSimdV1Targets.cmake
    NAMESPACE Ascify950::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Ascify950RowwiseSimdV1)
