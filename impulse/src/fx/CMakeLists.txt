# impulse-fx: the Flash-Next engine's server front (One Engine P1c), a project of its own with NO CUDA in it.
#
# impulse-server hands a Flash-Next pack to this binary (exec, the same arguments: src/gemma4/g4_server.cpp's main).
# It holds the HTTP / OpenAI layer (llama.cpp's server code), the dispatch and llama.cpp's tokenizer and chat template,
# all built from the same llama.cpp checkout as impulse-server but with ggml's CPU backend only (GGML_CUDA OFF), and
# loads the engine, libimpulse_fx, at run time.  So the process that runs a pack holds exactly the CUDA code the
# standalone `strata` holds: gate P1b (the test server, 2026-10-25) found impulse-server's own ggml-cuda kernels loaded into every
# context under the engine's CUDA_MODULE_LOADING=EAGER, ~110 MiB of VRAM per card the engine's cache did not get.
# impulse-server (the GGUF engines, ggml-cuda) is not changed by any of this.
#
# Built by impulse/CMakeLists.txt with IMPULSE_FLASHNEXT=ON as an external project (-DIMPULSE_LLAMA_DIR=<the checkout>);
# outputs impulse-fx and fx-tokenize (gate P1a's tokenizer driver, CPU only) beside impulse-server.
cmake_minimum_required(VERSION 3.24)
project(impulse_fx_front VERSION 0.1.1 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(IMPULSE_LLAMA_DIR "" CACHE PATH "llama.cpp b6b003d source tree (impulse/CMakeLists.txt's)")
set(IMPULSE_VERSION "${PROJECT_VERSION}" CACHE STRING "impulse's version string")
set(LLAMA_DIR ${IMPULSE_LLAMA_DIR})
set(IMPULSE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
if(NOT EXISTS ${LLAMA_DIR}/common/sampling.cpp)
  message(FATAL_ERROR "impulse-fx: no llama.cpp source at '${LLAMA_DIR}' (-DIMPULSE_LLAMA_DIR)")
endif()
# impulse's one addition to the checkout (impulse/CMakeLists.txt appends it too; appended once)
foreach(pair "sampling.cpp|sampling_logits.cpp.inc" "sampling.h|sampling_logits.h.inc")
  string(REPLACE "|" ";" pair "${pair}")
  list(GET pair 0 target)
  list(GET pair 1 hook)
  file(READ ${LLAMA_DIR}/common/${target} text)
  string(FIND "${text}" "common_sampler_sample_logits" found)
  if(found EQUAL -1)
    file(READ ${IMPULSE}/llama-hooks/${hook} add)
    file(APPEND ${LLAMA_DIR}/common/${target} "${add}")
  endif()
endforeach()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)            # no CUDA in the front: the engine brings its own
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)          # ggml computes nothing here (the tokenizer, the template)
foreach(o LLAMA_BUILD_TESTS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_APP LLAMA_BUILD_UI LLAMA_USE_PREBUILT_UI LLAMA_OPENSSL LLAMA_TOOLS_INSTALL)
  set(${o} OFF CACHE BOOL "" FORCE)
endforeach()
foreach(o LLAMA_BUILD_COMMON LLAMA_BUILD_TOOLS LLAMA_BUILD_SERVER)
  set(${o} ON CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
# impulse-server's request parsing is the same as this one's (its private source copies of llama.cpp files)
include(${IMPULSE}/llama-hooks/lfm-json-tools.cmake)
include(${IMPULSE}/llama-hooks/stable-topk.cmake)        # (CUDA only: nothing here)
include(${IMPULSE}/llama-hooks/penalties-olastn.cmake)
include(${IMPULSE}/llama-hooks/fx-jinja-python.cmake)   # Python's float repr and str.strip() in the Jinja engine (impulse-fx only)
foreach(t server-context llama-server-impl)
  if(TARGET ${t})
    set_property(TARGET ${t} PROPERTY PRECOMPILE_HEADERS "")
    target_include_directories(${t} PRIVATE ${LLAMA_DIR})
  endif()
endforeach()
find_package(Threads REQUIRED)

set(SERVER_INC ${LLAMA_DIR}/tools/server ${LLAMA_DIR}/tools/mtmd ${LLAMA_DIR}/common ${LLAMA_DIR}/include ${LLAMA_DIR}/vendor
               ${LLAMA_DIR}/vendor/cpp-httplib ${LLAMA_DIR} ${LLAMA_DIR}/ggml/include)
if(MSVC)
  set(FX_FLAGS /bigobj /utf-8)
endif()
add_executable(impulse-fx ${CMAKE_CURRENT_SOURCE_DIR}/fx_server.cpp ${CMAKE_CURRENT_SOURCE_DIR}/fx_main.cpp
               ${CMAKE_CURRENT_SOURCE_DIR}/fx_front.cpp ${CMAKE_CURRENT_SOURCE_DIR}/fx_vision.cpp
               ${LLAMA_DIR}/vendor/hash/sha256/sha256.c)
target_compile_definitions(impulse-fx PRIVATE IMPULSE_VERSION="${IMPULSE_VERSION}")
target_compile_options(impulse-fx PRIVATE ${FX_FLAGS})
target_include_directories(impulse-fx PRIVATE ${SERVER_INC} ${IMPULSE}/src ${LLAMA_DIR}/vendor/hash)
target_link_libraries(impulse-fx PRIVATE server-context mtmd llama-common llama cpp-httplib Threads::Threads ${CMAKE_DL_LIBS})

add_executable(fx-tokenize ${IMPULSE}/tests/fx_tokenize.cpp)
target_include_directories(fx-tokenize PRIVATE ${LLAMA_DIR}/common ${LLAMA_DIR}/include)
target_link_libraries(fx-tokenize PRIVATE llama-common llama Threads::Threads)
