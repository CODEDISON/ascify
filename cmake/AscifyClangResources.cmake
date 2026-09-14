# Resolve the resource headers from the LLVM/Clang development package used to
# build Ascify. An explicit path is authoritative; never replace an invalid one.
function(ascify_find_clang_resources)
  set(ASCIFY_CLANG_RESOURCE_DIRECTORY "" CACHE PATH
    "Matching Clang resource directory (contains include/__clang_cuda_runtime_wrapper.h)")
  if(NOT ASCIFY_CLANG_RESOURCE_DIRECTORY STREQUAL "")
    set(candidates "${ASCIFY_CLANG_RESOURCE_DIRECTORY}")
  else()
    set(candidates)
    foreach(clang_name clang "clang-${LLVM_VERSION_MAJOR}")
      set(clang_executable "${LLVM_TOOLS_BINARY_DIR}/${clang_name}${CMAKE_EXECUTABLE_SUFFIX}")
      if(EXISTS "${clang_executable}")
        execute_process(COMMAND "${clang_executable}" -print-resource-dir
          RESULT_VARIABLE resource_result
          OUTPUT_VARIABLE resource_directory
          OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(resource_result EQUAL 0 AND NOT resource_directory STREQUAL "")
          list(APPEND candidates "${resource_directory}")
        endif()
      endif()
    endforeach()
    foreach(library_directory IN LISTS LLVM_LIBRARY_DIRS)
      list(APPEND candidates "${library_directory}/clang/${LIB_CLANG_RES}")
    endforeach()
  endif()

  foreach(candidate IN LISTS candidates)
    if(EXISTS "${candidate}/include/__clang_cuda_runtime_wrapper.h" AND
       NOT IS_DIRECTORY "${candidate}/include/__clang_cuda_runtime_wrapper.h" AND
       EXISTS "${candidate}/include/cuda_wrappers/algorithm" AND
       NOT IS_DIRECTORY "${candidate}/include/cuda_wrappers/algorithm")
      get_filename_component(resource_directory "${candidate}" REALPATH)
      set(ASCIFY_CLANG_RESOURCE_DIRECTORY "${resource_directory}" PARENT_SCOPE)
      message(STATUS "Ascify Clang resource directory: ${resource_directory}")
      return()
    endif()
  endforeach()

  if(ASCIFY_INSTALL_CLANG_HEADERS OR NOT ASCIFY_CLANG_RESOURCE_DIRECTORY STREQUAL "")
    message(FATAL_ERROR
      "Matching Clang resource headers were not found. Set "
      "ASCIFY_CLANG_RESOURCE_DIRECTORY to the directory containing "
      "include/__clang_cuda_runtime_wrapper.h and include/cuda_wrappers/algorithm.")
  endif()
  message(STATUS "Clang resource headers unavailable; supply --clang-resource-directory at runtime")
endfunction()
