cmake_minimum_required(VERSION 3.16.8)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are unsupported. Use cmake -S . -B build.")
endif()

project(ascify-clang LANGUAGES C CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/dist" CACHE PATH
    "Ascify installation prefix" FORCE)
endif()
include(GNUInstallDirs)

option(ASCIFY_CLANG_TESTS "Build ascify-clang and register release checks" OFF)
option(ASCIFY_CLANG_TESTS_ONLY "Register release checks without building ascify-clang" OFF)
option(ASCIFY_INSTALL_CLANG_HEADERS "Install matching Clang resource headers" ON)
option(ADDRESS_SANITIZER "Build ascify-clang with AddressSanitizer" OFF)

if(NOT ASCIFY_CLANG_TESTS_ONLY)
  find_package(LLVM REQUIRED CONFIG)
  find_package(Clang REQUIRED CONFIG)
  find_package(Threads REQUIRED)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}: ${LLVM_DIR}")
  message(STATUS "Found Clang: ${Clang_DIR}")
  message(STATUS "Ascify compiler: ${CMAKE_CXX_COMPILER}")

  # Keep core LLVM libraries in LLVM_LINK_COMPONENTS so AddLLVM respects
  # distributions built with LLVM_LINK_LLVM_DYLIB=ON.
  set(LLVM_LINK_COMPONENTS
    ProfileData Support MCParser MC BitReader Option Core)
  set(ASCIFY_ADDL_LINK_LIBS)
  if(LLVM_PACKAGE_VERSION VERSION_GREATER "6.0.1")
    list(APPEND ASCIFY_ADDL_LINK_LIBS clangToolingInclusions)
  endif()
  if(LLVM_PACKAGE_VERSION VERSION_GREATER "9.0.1")
    list(APPEND LLVM_LINK_COMPONENTS FrontendOpenMP)
  endif()
  if(LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL "15.0.0")
    list(APPEND LLVM_LINK_COMPONENTS WindowsDriver)
    list(APPEND ASCIFY_ADDL_LINK_LIBS clangSupport)
  endif()

  add_llvm_executable(ascify-clang
    src/ArgParse.cpp
    src/AscifyAction.cpp
    src/CUDA2DPP.cpp
    src/CUDA2DPP_Device_API.cpp
    src/CUDA2DPP_Runtime_API_functions.cpp
    src/CUDA2DPP_Runtime_API_types.cpp
    src/DavC310TargetRecipe.cpp
    src/DeviceMemoryLowering.cpp
    src/FrontendCompatibility.cpp
    src/ImplicitCudaHeaders.cpp
    src/LLVMCompat.cpp
    src/LocalHeader.cpp
    src/MigrationReceipt.cpp
    src/MigrationReceiptIO.cpp
    src/Statistics.cpp
    src/StringUtils.cpp
    src/UniformBlockReduction.cpp
    src/main.cpp)
  target_compile_features(ascify-clang PRIVATE cxx_std_17)
  set_target_properties(ascify-clang PROPERTIES CXX_EXTENSIONS OFF)
  target_include_directories(ascify-clang SYSTEM PRIVATE
    ${CLANG_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
  target_include_directories(ascify-clang PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  separate_arguments(ASCIFY_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
  target_compile_options(ascify-clang PRIVATE ${ASCIFY_LLVM_DEFINITIONS})
  target_link_directories(ascify-clang PRIVATE ${LLVM_LIBRARY_DIRS})
  target_link_libraries(ascify-clang PRIVATE
    clangASTMatchers clangFrontend clangTooling clangParse clangSerialization
    clangSema clangEdit clangFormat clangLex clangAnalysis clangDriver clangAST
    clangToolingCore clangRewrite clangBasic ${ASCIFY_ADDL_LINK_LIBS}
    Threads::Threads)

  if(MSVC)
    target_link_libraries(ascify-clang PRIVATE version)
    target_compile_options(ascify-clang PRIVATE /GR- /EHs- /EHc- /MP)
  else()
    target_compile_options(ascify-clang PRIVATE
      -fno-rtti -fvisibility-inlines-hidden)
  endif()
  if(ADDRESS_SANITIZER)
    if(MSVC)
      message(FATAL_ERROR "ADDRESS_SANITIZER currently requires Clang or GCC")
    endif()
    target_compile_options(ascify-clang PRIVATE -fsanitize=address)
    target_link_options(ascify-clang PRIVATE -fsanitize=address)
  endif()
  if(NOT "${EXTRA_CFLAGS}" STREQUAL "")
    separate_arguments(ASCIFY_EXTRA_CFLAGS NATIVE_COMMAND "${EXTRA_CFLAGS}")
    target_compile_options(ascify-clang PRIVATE ${ASCIFY_EXTRA_CFLAGS})
  endif()

  if(LLVM_PACKAGE_VERSION VERSION_GREATER_EQUAL "16.0.0")
    set(LIB_CLANG_RES "${LLVM_VERSION_MAJOR}")
  else()
    set(LIB_CLANG_RES "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
  endif()

  include(cmake/AscifyClangResources.cmake)
  ascify_find_clang_resources()

  # GNUInstallDirs allows non-default binary/libexec directories. Resource
  # headers are private compiler data, separate from the public Ascify headers.
  file(RELATIVE_PATH ASCIFY_CLANG_RESOURCE_INSTALL_FROM_BINDIR
    "${CMAKE_INSTALL_FULL_BINDIR}"
    "${CMAKE_INSTALL_FULL_LIBEXECDIR}/ascify/clang/${LIB_CLANG_RES}")
  file(RELATIVE_PATH ASCIFY_FRONTEND_COMPAT_INSTALL_FROM_BINDIR
    "${CMAKE_INSTALL_FULL_BINDIR}"
    "${CMAKE_INSTALL_FULL_LIBEXECDIR}/ascify/frontend-compat")
  target_compile_definitions(ascify-clang PRIVATE
    LIB_CLANG_RES=${LIB_CLANG_RES}
    ASCIFY_CLANG_RESOURCE_INSTALL_FROM_BINDIR="${ASCIFY_CLANG_RESOURCE_INSTALL_FROM_BINDIR}"
    ASCIFY_CLANG_RESOURCE_BUILD_DIR="${ASCIFY_CLANG_RESOURCE_DIRECTORY}"
    ASCIFY_EXECUTABLE_BUILD_DIR="$<TARGET_FILE_DIR:ascify-clang>"
    ASCIFY_FRONTEND_COMPAT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/frontend_compat"
    ASCIFY_FRONTEND_COMPAT_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    ASCIFY_FRONTEND_COMPAT_INSTALL_FROM_BINDIR="${ASCIFY_FRONTEND_COMPAT_INSTALL_FROM_BINDIR}")

  install(TARGETS ascify-clang RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT ascify-compat-headers)
  install(FILES acl_cub/aclcub.hpp
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/acl_cub"
    COMPONENT ascify-compat-headers)
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/frontend_compat/
    DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/ascify/frontend-compat"
    COMPONENT ascify-frontend-compat)
  install(FILES LICENSE THIRD_PARTY_NOTICES.md
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/ascify")
  install(DIRECTORY LICENSES/
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/ascify")

  if(ASCIFY_INSTALL_CLANG_HEADERS)
    install(DIRECTORY "${ASCIFY_CLANG_RESOURCE_DIRECTORY}/include/"
      DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/ascify/clang/${LIB_CLANG_RES}/include"
      COMPONENT clang-resource-headers
      FILES_MATCHING
      PATTERN "*.h" PATTERN "*.modulemap"
      PATTERN "algorithm" PATTERN "complex" PATTERN "new"
      PATTERN "ppc_wrappers" EXCLUDE
      PATTERN "omp-tools.h" EXCLUDE PATTERN "omp.h" EXCLUDE
      PATTERN "ompt.h" EXCLUDE PATTERN "openmp_wrappers" EXCLUDE)
  endif()

  if(UNIX)
    if(APPLE)
      set(ASCIFY_INSTALL_ORIGIN "@loader_path")
    else()
      set(ASCIFY_INSTALL_ORIGIN "$ORIGIN")
    endif()
    file(RELATIVE_PATH ASCIFY_INSTALL_LIB_FROM_BINDIR
      "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
    set_target_properties(ascify-clang PROPERTIES
      INSTALL_RPATH "${ASCIFY_INSTALL_ORIGIN}/${ASCIFY_INSTALL_LIB_FROM_BINDIR}")
  endif()
endif()

if(ASCIFY_CLANG_TESTS OR ASCIFY_CLANG_TESTS_ONLY)

  message(STATUS "Registering Ascify release checks")

  include(CTest)
  enable_testing()

  find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
  find_program(ASCIFY_TEST_SHELL sh)
  if(NOT ASCIFY_TEST_SHELL)
    message(FATAL_ERROR
      "Can't find a POSIX sh executable required by tests/run_release_checks.sh")
  endif()

  set(ASCIFY_TEST_BINARY "" CACHE FILEPATH
    "Existing ascify-clang binary used with ASCIFY_CLANG_TESTS_ONLY")
  set(ASCIFY_TEST_CUDA_PATH "" CACHE PATH
    "CUDA parsing root used by translated-fixture release checks")
  set(ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY "" CACHE PATH
    "Clang resource directory used by translated-fixture release checks")

  if(NOT ASCIFY_CLANG_TESTS_ONLY AND
     ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY STREQUAL "")
    set(ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY
      "${ASCIFY_CLANG_RESOURCE_DIRECTORY}")
  endif()

  set(ASCIFY_RELEASE_CHECK_ENV
    "PYTHON=${Python3_EXECUTABLE}"
    "ASCIFY_BINARY="
    "ASCIFY_CUDA_PATH="
    "ASCIFY_CLANG_RESOURCE_DIRECTORY=")
  if(ASCIFY_CLANG_TESTS_ONLY AND NOT ASCIFY_TEST_BINARY STREQUAL "")
    if(ASCIFY_TEST_CUDA_PATH STREQUAL "" OR
       ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY STREQUAL "")
      message(FATAL_ERROR
        "ASCIFY_TEST_BINARY requires ASCIFY_TEST_CUDA_PATH and "
        "ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY")
    endif()
    list(APPEND ASCIFY_RELEASE_CHECK_ENV
      "ASCIFY_BINARY=${ASCIFY_TEST_BINARY}")
  elseif(NOT ASCIFY_CLANG_TESTS_ONLY AND
         NOT ASCIFY_TEST_CUDA_PATH STREQUAL "")
    list(APPEND ASCIFY_RELEASE_CHECK_ENV
      "ASCIFY_BINARY=$<TARGET_FILE:ascify-clang>")
  endif()
  if(NOT ASCIFY_TEST_CUDA_PATH STREQUAL "")
    list(APPEND ASCIFY_RELEASE_CHECK_ENV
      "ASCIFY_CUDA_PATH=${ASCIFY_TEST_CUDA_PATH}")
  endif()
  if(NOT ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY STREQUAL "")
    list(APPEND ASCIFY_RELEASE_CHECK_ENV
      "ASCIFY_CLANG_RESOURCE_DIRECTORY=${ASCIFY_TEST_CLANG_RESOURCE_DIRECTORY}")
  endif()

  add_test(
    NAME ascify-release-checks
    COMMAND
      ${CMAKE_COMMAND} -E env ${ASCIFY_RELEASE_CHECK_ENV}
      ${ASCIFY_TEST_SHELL}
      ${CMAKE_CURRENT_LIST_DIR}/tests/run_release_checks.sh)
  set_tests_properties(ascify-release-checks PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

  if(NOT ASCIFY_CLANG_TESTS_ONLY)
    add_test(NAME ascify-cli-surface
      COMMAND "${CMAKE_COMMAND}" -E env
        "ASCIFY_BINARY=$<TARGET_FILE:ascify-clang>"
        "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/engineering/test_cli_surface.py")
  endif()

  add_custom_target(test-ascify
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      -R "^ascify-(release-checks|cli-surface)$"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL)
  if(NOT ASCIFY_CLANG_TESTS_ONLY)
    add_dependencies(test-ascify ascify-clang)
  endif()

  add_custom_target(test-ascify-clang)
  add_dependencies(test-ascify-clang test-ascify)
  set_target_properties(test-ascify test-ascify-clang PROPERTIES
    FOLDER "Tests")

endif() # if(ASCIFY_CLANG_TESTS OR ASCIFY_CLANG_TESTS_ONLY)
