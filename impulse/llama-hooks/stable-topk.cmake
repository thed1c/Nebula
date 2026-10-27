# Deterministic ggml_top_k on CUDA (Qwen3.8 Flash-Next's attention indexer). With CCCL >= 3.2 ggml-cuda picks CUB's
# DeviceTopK, which does not promise which of several equal scores it returns (determinism::not_guaranteed); the indexer
# scores every cell of a block alike, so its picks, and with them the logits, changed from run to run (llama.cpp too).
# This build keeps ggml-cuda's other path, argsort + copy (CUB's radix sort, ties in index order; one launch for all rows
# instead of one per row). A private copy of top-k.cu replaces the checkout's in the ggml-cuda target; the shared llama.cpp
# checkout is not changed. -DIMPULSE_STABLE_TOPK=OFF keeps llama.cpp's choice.
option(IMPULSE_STABLE_TOPK "ggml_top_k through argsort on CUDA (deterministic ties)" ON)
if(IMPULSE_STABLE_TOPK AND TARGET ggml-cuda)
  set(_tk_upstream "${LLAMA_DIR}/ggml/src/ggml-cuda/top-k.cu")
  file(READ "${_tk_upstream}" _tk_source)
  set(_tk_anchor "#endif      // GGML_CUDA_USE_CUB\n")
  string(FIND "${_tk_source}" "${_tk_anchor}" _tk_at)
  if(_tk_at EQUAL -1)
    message(FATAL_ERROR "stable top-k hook does not match the pinned llama.cpp source")
  endif()
  string(REPLACE "${_tk_anchor}" "${_tk_anchor}#undef CUB_TOP_K_AVAILABLE   // impulse: deterministic ties (llama-hooks/stable-topk.cmake)\n" _tk_source "${_tk_source}")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/llama-hooks")
  set(_tk_generated "${CMAKE_CURRENT_BINARY_DIR}/llama-hooks/top-k.cu")
  file(WRITE "${_tk_generated}" "${_tk_source}")
  get_target_property(_tk_sources ggml-cuda SOURCES)
  list(FIND _tk_sources "${_tk_upstream}" _tk_idx)
  if(_tk_idx EQUAL -1)
    message(FATAL_ERROR "stable top-k hook: top-k.cu is not a ggml-cuda source")
  endif()
  list(REMOVE_ITEM _tk_sources "${_tk_upstream}")
  list(APPEND _tk_sources "${_tk_generated}")
  set_property(TARGET ggml-cuda PROPERTY SOURCES "${_tk_sources}")
  target_include_directories(ggml-cuda PRIVATE "${LLAMA_DIR}/ggml/src/ggml-cuda")
  message(STATUS "impulse: ggml_top_k on CUDA through argsort (deterministic ties)")
endif()
