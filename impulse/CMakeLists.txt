# Impulse: one build for the Gemma 4 and Qwen3.6 engines and their server, on Linux and Windows.
#
#   cmake -S impulse -B build-impulse -DCMAKE_BUILD_TYPE=Release          (Windows: from a "x64 Native Tools" prompt, -G Ninja)
#   cmake --build build-impulse --config Release
#
# It fetches llama.cpp at the commit every number in impulse/ was checked against (b6b003d, as a zip: no git needed),
# appends Impulse's one addition to it (impulse/llama-hooks: llama.cpp's sampler run on a logits vector, which the
# server needs because the forward pass is the engine's, not a llama_context), and builds the parts of llama.cpp the
# server links - ggml with CUDA, llama, common, the server's request handling, mtmd for images - statically into one
# binary, impulse-server, which serves either model. ggml is linked statically on purpose: the CPU expert pool
# calls ggml-cpu's own dot products and SwiGLU so its rows round exactly as llama.cpp's CPU path does, and those are
# internal symbols a shared ggml does not export on Windows.
#
# Options:
#   -DIMPULSE_LLAMA_DIR=<path>   use a llama.cpp b6b003d source tree already on disk (the setup's download)
#   -DIMPULSE_PORTABLE=ON        a binary for other PCs: ggml-cpu at an AVX2 baseline instead of this CPU's features
#   -DCMAKE_CUDA_ARCHITECTURES=...   default 75-real;80-real;86-real;89-real;90-real;100-real;120a-real (RTX 20 through
#                                    RTX 50, A100, H100, B200; CUDA 13; the V100 (70) needs a CUDA 12 build of its own)
#   -DIMPULSE_TOOLS=ON           also the test drivers (impulse-gemma4, impulse-qwen35, logit-cmp) and route-oracle
#   -DIMPULSE_FLASHNEXT=ON       also Strata's engine from flashnext/ (One Engine P1): libimpulse_fx (the engine as one
#                                shared library impulse-server loads for a Flash-Next pack) and the standalone `strata`,
#                                flashnext/ built as its own project with Strata's options, its ggml from the same
#                                llama.cpp checkout (see below). -DIMPULSE_FX_CUDA_ARCHITECTURES, -DIMPULSE_FX_JOBS.
cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)   # FetchContent_Populate of a declared zip (deprecated from CMake 3.30)
endif()
project(impulse VERSION 0.1.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(IMPULSE_LLAMA_COMMIT b6b003d2cb29647d968302eb2db8da6f66303b3e)
set(IMPULSE_LLAMA_DIR "" CACHE PATH "llama.cpp b6b003d source tree (empty: download it)")
option(IMPULSE_PORTABLE "ggml-cpu at an AVX2 baseline (binaries for other PCs)" OFF)
option(IMPULSE_TOOLS "also build the test drivers and route-oracle" OFF)
option(IMPULSE_FLASHNEXT "also build Strata's engine (flashnext/) as libraries and the standalone strata" OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;100-real;120a-real")
endif()
set(ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
foreach(cfg Release RelWithDebInfo Debug MinSizeRel)
  string(TOUPPER ${cfg} CFG)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG} ${CMAKE_BINARY_DIR}/bin)
endforeach()

# ---- llama.cpp b6b003d
if(IMPULSE_LLAMA_DIR)
  set(LLAMA_DIR ${IMPULSE_LLAMA_DIR})
else()
  include(FetchContent)
  FetchContent_Declare(impulse_llamacpp
    URL https://github.com/ggml-org/llama.cpp/archive/${IMPULSE_LLAMA_COMMIT}.zip)
  FetchContent_GetProperties(impulse_llamacpp)
  if(NOT impulse_llamacpp_POPULATED)
    FetchContent_Populate(impulse_llamacpp)
  endif()
  set(LLAMA_DIR ${impulse_llamacpp_SOURCE_DIR})
endif()
if(NOT EXISTS ${LLAMA_DIR}/common/sampling.cpp)
  message(FATAL_ERROR "no llama.cpp source at ${LLAMA_DIR}")
endif()
# the hook, appended once (a tree that already has it - the patched checkout of impulse/patches - is left as it is)
foreach(pair "sampling.cpp|sampling_logits.cpp.inc" "sampling.h|sampling_logits.h.inc")
  string(REPLACE "|" ";" pair "${pair}")
  list(GET pair 0 target)
  list(GET pair 1 hook)
  file(READ ${LLAMA_DIR}/common/${target} text)
  string(FIND "${text}" "common_sampler_sample_logits" found)
  if(found EQUAL -1)
    file(READ ${ROOT}/llama-hooks/${hook} add)
    file(APPEND ${LLAMA_DIR}/common/${target} "${add}")
    message(STATUS "impulse: added common_sampler_sample_logits to common/${target}")
  endif()
endforeach()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA_FA_QUANTS "q8_0-q8_0;q4_0-q4_0;f16-f16;q8_0-q4_0" CACHE STRING "" FORCE)   # q4 KV; q8_0 K + q4_0 V (Dense)
if(IMPULSE_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  foreach(f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${f} ON CACHE BOOL "" FORCE)
  endforeach()
endif()
foreach(o LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_APP LLAMA_BUILD_UI LLAMA_USE_PREBUILT_UI LLAMA_OPENSSL LLAMA_TOOLS_INSTALL)
  set(${o} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(o LLAMA_BUILD_COMMON LLAMA_BUILD_TOOLS LLAMA_BUILD_SERVER)
  set(${o} ON CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
include(${ROOT}/llama-hooks/lfm-json-tools.cmake)
include(${ROOT}/llama-hooks/stable-topk.cmake)
include(${ROOT}/llama-hooks/penalties-olastn.cmake)
# the server code finds its precompiled header and includes from CMAKE_SOURCE_DIR (llama.cpp's root when it is built alone,
# this folder here): no precompiled header, and llama.cpp's root as an include folder
foreach(t server-context llama-server-impl)
  set_property(TARGET ${t} PROPERTY PRECOMPILE_HEADERS "")
  target_include_directories(${t} PRIVATE ${LLAMA_DIR})
endforeach()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# ---- Strata's CPU expert pool, with the same per-file ISA flags as Strata's own build. One copy since One Engine P0: the
# sources are flashnext/src/kernels/cpu and flashnext/include (Strata's tree, with Impulse's additions merged in;
# third_party/strata/UPSTREAM says where each file went). Built here as Impulse's own library - Impulse's flags (the
# AVX-VNNI copies off, C++17) - and by flashnext/CMakeLists.txt as strata_kernels_cpu with Strata's.
set(SK ${ROOT}/flashnext/src/kernels/cpu)
set(STRATA_INC ${ROOT}/flashnext/include)
add_library(impulse_pool STATIC ${SK}/pool.cpp ${SK}/expert.cpp ${SK}/expert_layout.cpp ${SK}/q2_avx2.cpp ${SK}/native_expert.cpp
            ${SK}/iq_avx512.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp ${SK}/kq_avx1.cpp ${SK}/q2_avx1.cpp ${SK}/iq_avx1.cpp
            ${SK}/nvfp4_avx512.cpp ${SK}/nvfp4_avx512bw.cpp ${SK}/nvfp4_avx2.cpp)
target_include_directories(impulse_pool PUBLIC ${STRATA_INC})
target_include_directories(impulse_pool PRIVATE ${ROOT}/flashnext/third_party/ggml)
target_compile_definitions(impulse_pool PUBLIC STRATA_NATIVE_EXPERTS=1)
if(MSVC)
  set_source_files_properties(${SK}/expert.cpp ${SK}/iq_avx512.cpp ${SK}/nvfp4_avx512.cpp ${SK}/nvfp4_avx512bw.cpp
                              PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp ${SK}/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(${SK}/kq_avx1.cpp ${SK}/q2_avx1.cpp ${SK}/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX")
else()
  set_source_files_properties(${SK}/expert.cpp ${SK}/iq_avx512.cpp ${SK}/nvfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  # the NVFP4 rows for AVX-512 without VNNI/VBMI (Skylake-SP, the test server's Gold 6140): no VNNI, no VBMI flags
  set_source_files_properties(${SK}/nvfp4_avx512bw.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mf16c;-mfma")
  set_source_files_properties(${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp ${SK}/kq_avx2.cpp ${SK}/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${SK}/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-mavx")
  set_source_files_properties(${SK}/q2_avx1.cpp ${SK}/iq_avx1.cpp PROPERTIES COMPILE_OPTIONS "-msse4.2;-mavx;-mno-avx2;-mno-fma;-mno-f16c")
endif()
# the AVX-VNNI copies stay off: the engines' checks were made with the plain AVX2 rows
set_property(SOURCE ${SK}/q2_avx2.cpp ${SK}/iq_avx2.cpp APPEND PROPERTY COMPILE_DEFINITIONS STRATA_AVXVNNI=0)
target_link_libraries(impulse_pool PUBLIC ggml-cpu ggml-base Threads::Threads)

# ---- Strata's engine (flashnext/): libimpulse_fx (One Engine P1), the engine as one shared library that impulse-server
# loads for a Flash-Next pack (src/fx/fx_server.cpp), and the standalone `strata`.  flashnext/ is built as its own
# project (an external project, its own configure), with Strata's own build options and its own ggml from the same
# llama.cpp checkout: the engine is built exactly as the reference engine (STRATA_PORTABLE: ggml-cpu at the AVX2
# baseline, no OpenMP), only position-independent with hidden visibility, and its copies of the CPU pool, the MMQ
# prompt kernels (ggml-cuda's sources with -use_fast_math) and ggml stay inside the library.  impulse-server's own
# ggml, ggml-cuda and CPU pool (the GGUF engines') keep their flags and are never linked together with those copies
# (P0 found 10 MMQ symbols and the pool's 112 defined twice, compiled with different flags).  Outputs in bin/:
# libimpulse_fx.so (impulse_fx.dll), strata, libstrata_kq.so; -DIMPULSE_TOOLS=ON adds Strata's tests (STRATA_BUILD_TESTS).
if(IMPULSE_FLASHNEXT)
  include(ExternalProject)
  set(IMPULSE_FX_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING
      "CUDA architectures of the Flash-Next engine (default: this build's; the reference engine is built for 120)")
  option(IMPULSE_FX_PORTABLE "the Flash-Next engine's ggml-cpu at the AVX2 baseline (STRATA_PORTABLE, as the reference engine)" ON)
  set(IMPULSE_FX_JOBS "" CACHE STRING "parallel jobs for the Flash-Next engine's build (empty: the build tool's default)")
  string(REPLACE ";" "|" _fx_archs "${IMPULSE_FX_CUDA_ARCHITECTURES}")
  set(_fx_args -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CUDA_ARCHITECTURES=${_fx_archs} -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_FX_LIBRARY=ON
      -DSTRATA_GGML_DIR=${LLAMA_DIR} -DSTRATA_PORTABLE=${IMPULSE_FX_PORTABLE} -DSTRATA_BUILD_TESTS=${IMPULSE_TOOLS}
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/bin)
  if(CMAKE_CUDA_COMPILER)
    list(APPEND _fx_args -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER})
  endif()
  set(_fx_build ${CMAKE_COMMAND} --build <BINARY_DIR> --config ${CMAKE_BUILD_TYPE})
  if(IMPULSE_FX_JOBS)
    list(APPEND _fx_build -j ${IMPULSE_FX_JOBS})
  endif()
  ExternalProject_Add(impulse_flashnext
    SOURCE_DIR ${ROOT}/flashnext
    BINARY_DIR ${CMAKE_BINARY_DIR}/flashnext
    LIST_SEPARATOR |
    CMAKE_ARGS ${_fx_args}
    BUILD_COMMAND ${_fx_build}
    BUILD_ALWAYS ON
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/bin/${CMAKE_SHARED_LIBRARY_PREFIX}impulse_fx${CMAKE_SHARED_LIBRARY_SUFFIX}
    USES_TERMINAL_BUILD ON)
  # impulse-fx (One Engine P1c): the server front for a pack, its own project with ggml's CPU backend only - no CUDA
  # code in the process that runs the engine but the engine's own (src/fx/CMakeLists.txt); impulse-server execs it
  set(_front_args -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DIMPULSE_LLAMA_DIR=${LLAMA_DIR} -DIMPULSE_VERSION=${PROJECT_VERSION} -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${CMAKE_BINARY_DIR}/bin)
  ExternalProject_Add(impulse_fx_front
    SOURCE_DIR ${ROOT}/src/fx
    BINARY_DIR ${CMAKE_BINARY_DIR}/fx-front
    CMAKE_ARGS ${_front_args}
    BUILD_COMMAND ${_fx_build}
    BUILD_ALWAYS ON
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/bin/impulse-fx${CMAKE_EXECUTABLE_SUFFIX}
    USES_TERMINAL_BUILD ON)
endif()

# ---- the engines
if(MSVC)
  set(LITE_FLAGS /bigobj /utf-8)
else()
  set(LITE_FLAGS -ffp-contract=off)   # the pool op's combine must round like ggml's mul/add
endif()
foreach(e g4 q35)
  if(e STREQUAL g4)
    set(src ${ROOT}/src/gemma4/g4_engine.cpp ${ROOT}/src/gemma4/g4_tsplit.cpp)
  else()
    set(src ${ROOT}/src/qwen35/q35_engine.cpp ${ROOT}/src/qwen35/q35_tsplit.cpp ${ROOT}/src/lite/lfm_lora.cpp)
  endif()
  add_library(${e}_engine STATIC ${src})
  target_compile_options(${e}_engine PRIVATE ${LITE_FLAGS})
  target_include_directories(${e}_engine PUBLIC ${LLAMA_DIR}/ggml/include ${STRATA_INC})
  target_include_directories(${e}_engine PRIVATE ${LLAMA_DIR}/ggml/src)   # ggml-backend-impl.h: the pinned host buffer type (src/lite/pinned_host.hpp)
  target_link_libraries(${e}_engine PUBLIC impulse_pool ggml ggml-base ggml-cpu ggml-cuda CUDA::cudart Threads::Threads)
endforeach()

# ---- the server: both engines in one binary; the GGUF's architecture picks one
set(SERVER_INC ${LLAMA_DIR}/tools/server ${LLAMA_DIR}/tools/mtmd ${LLAMA_DIR}/common ${LLAMA_DIR}/include ${LLAMA_DIR}/vendor
               ${LLAMA_DIR}/vendor/cpp-httplib ${LLAMA_DIR})
# a Flash-Next pack goes to impulse-fx (src/fx/: exec, the same arguments), the GGUF engines stay in this binary
add_executable(impulse-server ${ROOT}/src/gemma4/g4_server.cpp ${ROOT}/src/qwen35/q35_server.cpp)
target_compile_definitions(impulse-server PRIVATE IMPULSE_COMBINED=1 IMPULSE_VERSION="${PROJECT_VERSION}")
target_compile_options(impulse-server PRIVATE ${LITE_FLAGS})
target_include_directories(impulse-server PRIVATE ${SERVER_INC} ${ROOT}/src/gemma4 ${ROOT}/src/qwen35)
target_link_libraries(impulse-server PRIVATE g4_engine q35_engine server-context mtmd llama-common llama cpp-httplib Threads::Threads ${CMAKE_DL_LIBS})
if(IMPULSE_FLASHNEXT)
  add_dependencies(impulse-server impulse_flashnext impulse_fx_front)
endif()

if(IMPULSE_TOOLS)
  add_executable(impulse-gemma4 ${ROOT}/src/gemma4/g4_cli.cpp)
  target_link_libraries(impulse-gemma4 PRIVATE g4_engine)
  add_executable(g4-seqs tests/g4_seqs.cpp)
  target_include_directories(g4-seqs PRIVATE ${ROOT}/src/gemma4)
  target_link_libraries(g4-seqs PRIVATE g4_engine)
  add_executable(q35-seqs tests/q35_seqs.cpp)
  target_include_directories(q35-seqs PRIVATE ${ROOT}/src/qwen35)
  target_link_libraries(q35-seqs PRIVATE q35_engine)
  add_executable(impulse-qwen35 ${ROOT}/src/qwen35/q35_cli.cpp)
  target_link_libraries(impulse-qwen35 PRIVATE q35_engine)
  add_executable(lfm25-chat tests/lfm25_chat.cpp)
  target_include_directories(lfm25-chat PRIVATE ${LLAMA_DIR}/src)
  target_link_libraries(lfm25-chat PRIVATE llama-common llama Threads::Threads)
  add_executable(lfm25-state tests/lfm25_state.cpp)
  target_include_directories(lfm25-state PRIVATE ${ROOT}/src/qwen35)
  target_link_libraries(lfm25-state PRIVATE q35_engine)
  add_executable(logit-cmp ${ROOT}/src/gemma4/logit_cmp.cpp)
  add_executable(api-key-test tests/api_key_test.cpp)   # impulse-server's --api-key check (no GPU, no model)
  # numa-always: the per-socket split of the CPU expert pool on made-up layouts (no GPU, no model)
  add_executable(numa-pool-test tests/numa_pool_test.cpp)
  target_include_directories(numa-pool-test PRIVATE ${STRATA_INC})
  add_executable(pool-tasks-test tests/pool_tasks_test.cpp)
  target_link_libraries(pool-tasks-test PRIVATE impulse_pool)
  # the NVFP4 CPU rows (VNNI, AVX-512BW, AVX2) against ggml-cpu's dot, and the BW twin against VNNI bit for bit (no GPU)
  add_executable(nvfp4-parity tests/nvfp4_parity.cpp)
  target_link_libraries(nvfp4-parity PRIVATE impulse_pool)
  # the CPU pool's per-job second-level scales (NVFP4's ffn_*_exps.scale) against llama.cpp's scale ops, every format (no GPU)
  add_executable(pool-scales-test tests/pool_scales_test.cpp)
  target_link_libraries(pool-scales-test PRIVATE impulse_pool)
  # numa-always: the per-socket placement of the CPU experts (src/lite/numa_place.hpp) on a made-up layout (no GPU)
  add_executable(numa-place-test tests/numa_place_test.cpp)
  target_link_libraries(numa-place-test PRIVATE impulse_pool)
  # One Engine P1 (no GPU, no model): which engine a -m path selects (a pack folder or a dense GGUF beside pack/ -> fx,
  # a GGUF alone -> its architecture's), and a mock libimpulse_fx speaking the serve protocol for the server's tests
  add_executable(fx-dispatch-test tests/fx_dispatch_test.cpp)
  target_include_directories(fx-dispatch-test PRIVATE ${ROOT}/src ${LLAMA_DIR}/ggml/include)
  target_link_libraries(fx-dispatch-test PRIVATE ggml-base)
  add_library(impulse_fx_mock SHARED tests/fx_mock_engine.cpp)
  set_target_properties(impulse_fx_mock PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  target_link_libraries(impulse_fx_mock PRIVATE Threads::Threads)
  # "parallel": N (no GPU, no model): impulse-fx's batch-slot scheduler (src/fx/fx_slots.hpp) against a fake engine
  add_executable(fx-slots-test tests/fx_slots_test.cpp)
  target_include_directories(fx-slots-test PRIVATE ${ROOT}/src)
  target_link_libraries(fx-slots-test PRIVATE Threads::Threads)
  # (fx-tokenize, gate P1a's tokenizer driver, is built CPU-only with impulse-fx: src/fx/CMakeLists.txt)
  foreach(t route-oracle route-log)
    add_executable(${t} ${CMAKE_CURRENT_SOURCE_DIR}/tools/${t}/route-log.cpp)
    target_include_directories(${t} PRIVATE ${LLAMA_DIR}/common ${LLAMA_DIR}/include ${LLAMA_DIR}/vendor)
    target_link_libraries(${t} PRIVATE llama-common llama Threads::Threads)
  endforeach()
endif()
