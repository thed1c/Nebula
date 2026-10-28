# Impulse portable: Impulse for a machine without an NVIDIA card (Halo on a Radeon, an Intel card or the CPU).
#
#   cmake -S impulse/portable -B build-portable -DCMAKE_BUILD_TYPE=Release        (Vulkan's SDK on the PATH for glslc)
#   cmake --build build-portable --target impulse-server
#
# Impulse's GGUF engines (gemma4, qwen35/lfm2 MoE and dense) are built on CUDA; a card without CUDA runs a GGUF through
# llama.cpp's own server path instead. This is that path, built the way Impulse builds it: the same llama.cpp commit
# (b6b003d), the same hooks (impulse/llama-hooks: the LFM JSON/tool parser, penalties in O(last_n)), the same
# LLAMA_API_KEY and --lora-scaled. No CUDA: ggml's CPU backend in every x86-64 variant, picked at run time, and Vulkan
# (AMD, Intel and NVIDIA cards alike), both as loadable backends beside the binary. Output in bin/: impulse-server
# (impulse-server.exe) and the ggml backend libraries it loads.
cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
project(impulse_portable VERSION 0.1.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(IMPULSE_LLAMA_COMMIT b6b003d2cb29647d968302eb2db8da6f66303b3e)
set(IMPULSE_LLAMA_DIR "" CACHE PATH "llama.cpp b6b003d source tree (empty: download it)")
option(IMPULSE_PORTABLE_VULKAN "the Vulkan backend (AMD, Intel and NVIDIA cards)" ON)
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(cfg Release RelWithDebInfo Debug MinSizeRel)
  string(TOUPPER ${cfg} CFG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

if(IMPULSE_LLAMA_DIR)
  set(LLAMA_DIR ${IMPULSE_LLAMA_DIR})
else()
  include(FetchContent)
  FetchContent_Declare(impulse_llamacpp URL https://github.com/ggml-org/llama.cpp/archive/${IMPULSE_LLAMA_COMMIT}.zip)
  FetchContent_GetProperties(impulse_llamacpp)
  if(NOT impulse_llamacpp_POPULATED)
    FetchContent_Populate(impulse_llamacpp)
  endif()
  set(LLAMA_DIR ${impulse_llamacpp_SOURCE_DIR})
endif()
if(NOT EXISTS ${LLAMA_DIR}/tools/server/server.cpp)
  message(FATAL_ERROR "no llama.cpp source at ${LLAMA_DIR}")
endif()
# Flint (ggml-org/llama.cpp#30928, after b6b003d), as in Impulse's own build (../CMakeLists.txt): the arch, its tokenizer
# and graph from patches/llama.cpp-b6b003d-flint.patch, applied to a private copy of the tree in the build folder. A tree
# that already has src/models/flint.cpp is used as it is.
if(NOT EXISTS ${LLAMA_DIR}/src/models/flint.cpp)
  set(_flint_dir ${CMAKE_BINARY_DIR}/llama.cpp-flint-src)
  set(_flint_patch ${ROOT}/patches/llama.cpp-b6b003d-flint.patch)
  file(SHA256 ${_flint_patch} _flint_sha)
  if(NOT EXISTS ${_flint_dir}/.impulse-flint-${_flint_sha})
    file(REMOVE_RECURSE ${_flint_dir})
    file(COPY ${LLAMA_DIR}/ DESTINATION ${_flint_dir} PATTERN ".git" EXCLUDE)
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} apply --whitespace=nowarn ${_flint_patch} WORKING_DIRECTORY ${_flint_dir} RESULT_VARIABLE _flint_rc)
    if(NOT _flint_rc EQUAL 0)
      message(FATAL_ERROR "impulse portable: the Flint patch does not apply to ${LLAMA_DIR}")
    endif()
    file(WRITE ${_flint_dir}/.impulse-flint-${_flint_sha} "")
    message(STATUS "impulse portable: llama.cpp with Flint support at ${_flint_dir}")
  endif()
  set(LLAMA_DIR ${_flint_dir})
endif()

# the libraries sit beside the binary and are found there (no wrapper, no LD_LIBRARY_PATH)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
# ggml: no CUDA; the CPU in every variant and Vulkan, each a library loaded at run time (ggml's backend registry)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
set(GGML_VULKAN ${IMPULSE_PORTABLE_VULKAN} CACHE BOOL "" FORCE)
foreach(o LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_UI LLAMA_USE_PREBUILT_UI LLAMA_OPENSSL LLAMA_CURL LLAMA_TOOLS_INSTALL)
  set(${o} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(o LLAMA_BUILD_COMMON LLAMA_BUILD_TOOLS LLAMA_BUILD_SERVER)
  set(${o} ON CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
# the server code finds its precompiled header and includes from CMAKE_SOURCE_DIR (llama.cpp's root when it is built
# alone, this folder here): no precompiled header, and llama.cpp's root as an include folder (as impulse/CMakeLists.txt)
foreach(t server-context llama-server-impl llama-server)
  if(TARGET ${t})
    set_property(TARGET ${t} PROPERTY PRECOMPILE_HEADERS "")
    target_include_directories(${t} PRIVATE ${LLAMA_DIR})
  endif()
endforeach()
include(${ROOT}/llama-hooks/lfm-json-tools.cmake)
include(${ROOT}/llama-hooks/penalties-olastn.cmake)

# llama.cpp's server under Impulse's name: one more copy of its target, named impulse-server
add_custom_target(impulse-server ALL
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:llama-server> ${CMAKE_BINARY_DIR}/bin/impulse-server${CMAKE_EXECUTABLE_SUFFIX}
  DEPENDS llama-server
  COMMENT "impulse-server (portable): llama.cpp's server path, CPU + Vulkan")
