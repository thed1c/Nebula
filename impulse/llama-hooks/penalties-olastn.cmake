# The repeat/frequency/presence penalties in O(last_n) instead of O(n_vocab). llama.cpp's penalties sampler looks every
# candidate up in its token-count map, so a 128K-vocabulary model (LFM2.5) pays ~1.5 ms a token for a penalty that touches
# at most last_n tokens. When the candidate array is still in token-id order (data[t].id == t for every counted token, as
# the sampler chain receives it from the logits), this applies the same float operations to the counted tokens only;
# otherwise it falls back to llama.cpp's scan. Same arithmetic per token, so the same bits. A private copy of
# llama-sampler.cpp replaces the checkout's in the llama target; the shared checkout is not changed.
# -DIMPULSE_PENALTIES_OLASTN=OFF keeps llama.cpp's scan.
option(IMPULSE_PENALTIES_OLASTN "penalties sampler in O(last_n) (same results)" ON)
if(IMPULSE_PENALTIES_OLASTN AND TARGET llama)
  set(_pn_upstream "${LLAMA_DIR}/src/llama-sampler.cpp")
  file(READ "${_pn_upstream}" _pn_source)
  set(_pn_anchor "    // Apply frequency and presence penalties to the cur_p\n    for (size_t i = 0; i < cur_p->size; ++i) {\n")
  string(FIND "${_pn_source}" "${_pn_anchor}" _pn_at)
  if(_pn_at EQUAL -1)
    message(FATAL_ERROR "penalties hook does not match the pinned llama.cpp source")
  endif()
  file(READ "${CMAKE_CURRENT_LIST_DIR}/penalties_olastn.inc" _pn_add)
  string(REPLACE "${_pn_anchor}" "${_pn_add}${_pn_anchor}" _pn_source "${_pn_source}")
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/llama-hooks")
  set(_pn_generated "${CMAKE_CURRENT_BINARY_DIR}/llama-hooks/llama-sampler.cpp")
  file(WRITE "${_pn_generated}" "${_pn_source}")
  get_target_property(_pn_sources llama SOURCES)
  list(FIND _pn_sources "${_pn_upstream}" _pn_idx)
  if(_pn_idx EQUAL -1)
    list(FIND _pn_sources "llama-sampler.cpp" _pn_idx)
    if(_pn_idx EQUAL -1)
      message(FATAL_ERROR "penalties hook: llama-sampler.cpp is not a llama source")
    endif()
    list(REMOVE_ITEM _pn_sources "llama-sampler.cpp")
  else()
    list(REMOVE_ITEM _pn_sources "${_pn_upstream}")
  endif()
  list(APPEND _pn_sources "${_pn_generated}")
  set_property(TARGET llama PROPERTY SOURCES "${_pn_sources}")
  # as upstream marks its core sources (llama.cpp src/CMakeLists.txt): never merged into the unity batches
  set_source_files_properties("${_pn_generated}" TARGET_DIRECTORY llama PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
  target_include_directories(llama PRIVATE "${LLAMA_DIR}/src")
  message(STATUS "impulse: penalties sampler in O(last_n)")
endif()
