// Impulse: pinned host memory for the scheduler's CPU-side buffers, allocated once and kept.
//
// ggml's scheduler puts a graph's inputs (token ids, positions, KV indices, the attention mask) in the buffer type given
// for the CPU backend; both engines give it the card's pinned host type, so the copies to the card are DMA. ggml-cuda's
// host type calls cudaMallocHost for every buffer and cudaFreeHost when it goes, and a chunked prompt read makes the
// scheduler's buffers grow and then drops them (end_read makes a new scheduler): the Nsight profile of 2026-10-16 saw
// three cudaMallocHost (121 ms each) and three cudaFreeHost (38 ms each) in one 15K-token read on an RTX 5070, 476 ms of
// a 2.04 s read. This buffer type hands out pinned regions from a pool it keeps for the life of the process: a region
// freed by the scheduler is kept and handed out again to the next request it fits; the pool grows only when a request
// is bigger than every free region (the smaller free ones go then). What lands in the regions and how it is read is
// unchanged: the buffers are the same host buffers (ggml_backend_cpu_buffer_from_ptr), so the outputs are the same bits.
//
// GGML_CUDA_NO_PINNED set, or cudaMallocHost failing: pageable CPU buffers, as ggml-cuda does.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"   // the buffer type and buffer structs (ggml/src)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace lite {

struct PinnedHostPool {
    struct Region { void * ptr; size_t cap; bool used; };
    std::mutex mu;
    std::vector<Region> regions;
    size_t total = 0, largest = 0;    // bytes pinned now; the biggest region ever made
    uint64_t allocs = 0, reuses = 0, grows = 0;

    static PinnedHostPool & get() { static PinnedHostPool p; return p; }

    // a region of at least `size` bytes: a free one that fits (the smallest), else a new one (the free ones that did not
    // fit are released first, so the pool only grows). A new region is at least twice the biggest so far: a chunked read's
    // mask grows with every chunk (n x n_kv), and without the headroom every chunk of a long read would pin a new region
    // (30 of them in a 120K read in 4K chunks); with it a read grows the pool a handful of times.
    void * take(size_t size) {
        std::lock_guard<std::mutex> lk(mu);
        allocs++;
        int best = -1;
        for (int i = 0; i < (int) regions.size(); ++i)
            if (!regions[i].used && regions[i].cap >= size && (best < 0 || regions[i].cap < regions[best].cap)) best = i;
        if (best >= 0) { regions[best].used = true; reuses++; return regions[best].ptr; }
        for (int i = (int) regions.size() - 1; i >= 0; --i)
            if (!regions[i].used) { cudaFreeHost(regions[i].ptr); total -= regions[i].cap; regions.erase(regions.begin() + i); }
        void * p = nullptr;
        size_t cap = std::max(size, 2 * largest);
        cap = (cap + ((size_t) 1 << 20) - 1) & ~(((size_t) 1 << 20) - 1);   // whole MiB
        if (cudaMallocHost(&p, cap) != cudaSuccess) {   // no room for the headroom: the request alone, else pageable
            (void) cudaGetLastError();
            cap = (size + ((size_t) 1 << 20) - 1) & ~(((size_t) 1 << 20) - 1);
            if (cudaMallocHost(&p, cap) != cudaSuccess) { (void) cudaGetLastError(); return nullptr; }
        }
        regions.push_back({ p, cap, true });
        total += cap; grows++; largest = std::max(largest, cap);
        if (getenv("IMPULSE_PINNED_LOG"))
            fprintf(stderr, "impulse: pinned host pool %.0f MiB (+%.0f MiB region, %llu regions)\n",
                    total / 1048576.0, cap / 1048576.0, (unsigned long long) regions.size());
        return p;
    }
    void give(void * ptr) {
        std::lock_guard<std::mutex> lk(mu);
        for (Region & r : regions) if (r.ptr == ptr) { r.used = false; return; }
    }
};

namespace detail {
inline const char * pinned_host_buft_name(ggml_backend_buffer_type_t) { return "CUDA_Host_Pool"; }
inline void pinned_host_buffer_free(ggml_backend_buffer_t buffer) { PinnedHostPool::get().give(buffer->context); }
inline ggml_backend_buffer_t pinned_host_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = getenv("GGML_CUDA_NO_PINNED") ? nullptr : PinnedHostPool::get().take(size);
    if (!ptr) return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);   // pageable, as ggml-cuda does
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = pinned_host_buffer_free;
    return buffer;
}
inline size_t pinned_host_buft_alignment(ggml_backend_buffer_type_t) { return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type()); }
inline size_t pinned_host_buft_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) { return ggml_backend_buft_get_alloc_size(ggml_backend_cpu_buffer_type(), t); }
inline bool pinned_host_buft_is_host(ggml_backend_buffer_type_t) { return true; }
} // namespace detail

// the buffer type for a scheduler's CPU backend: pinned host memory from the kept pool (the CPU backend's device, as the
// buffers are host buffers the CPU backend owns)
inline ggml_backend_buffer_type_t pinned_host_buffer_type(ggml_backend_t cpu) {
    static ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ detail::pinned_host_buft_name,
            /* .alloc_buffer   = */ detail::pinned_host_buft_alloc,
            /* .get_alignment  = */ detail::pinned_host_buft_alignment,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ detail::pinned_host_buft_alloc_size,
            /* .is_host        = */ detail::pinned_host_buft_is_host,
        },
        /* .device  = */ ggml_backend_get_device(cpu),
        /* .context = */ nullptr,
    };
    return &buft;
}

} // namespace lite
