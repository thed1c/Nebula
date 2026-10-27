// include/strata/core/async_pool.hpp - numa-always STRATA_ASYNC_POOL=1 (opt-in; with STRATA_DEFER_CPU=1 it pays).
//
// The verify window's host loop used to run a layer's CPU experts itself: plan and publish the GPU's share, run the pool,
// raise the layer's flag, and only then wait for the next layer's doorbell.  With expert deferral the GPU reaches the next
// layer while the CPU still works, but the host could not plan that layer until the pool was done.  Here the host plans,
// builds the jobs (in one of two buffers by layer parity) and hands them to a runner thread, which runs the pool and raises
// the layer's flag; the host goes straight back to the next doorbell.  The runner serves layers in order, so the flags
// (monotonic "at least" counters the GPU spins on) are never raised ahead of a layer whose rows are not written.
#pragma once

#include <cstdint>

namespace strata::kernels::cpu { class ExpertPool; struct NativeFmt; struct ExpertJobMulti; }

namespace strata::core {

/// The verify loop's request for one layer: where the CPU rows' flag is and the value that means "this layer's rows are
/// written".  The dispatch sets `taken` when it handed the layer to the runner (which then raises the flag).
struct AsyncFlagRequest {
    volatile uint32_t* flag = nullptr;
    uint32_t want = 0;
    bool taken = false;
};
/// numa-always STRATA_DEFER_WMAX: for a deferred layer, the CPU entries whose router weight is >= `threshold` are run first
/// and *early_flag = want is raised (the GPU combines them on time); the rest are run after and joined one layer late.
struct DeferSplitRequest {
    const float* weights = nullptr;   ///< this layer's routing weights, n_tok x k (the doorbell's host copy)
    float threshold = 0.0f;
    volatile uint32_t* early_flag = nullptr;
    uint32_t want = 0;
};
DeferSplitRequest*& defer_split_request();
/// Set by the verify loop around one pool call on its thread (nullptr: the ordered path).
AsyncFlagRequest*& async_flag_request();
/// Whether STRATA_ASYNC_POOL=1.
bool async_pool_enabled();
/// Queues a layer's pool work: run the native (or Strata-form) multi-token jobs, then raise *flag = want.
void async_pool_submit(strata::kernels::cpu::ExpertPool* pool, const strata::kernels::cpu::NativeFmt* fmt,
                       strata::kernels::cpu::ExpertJobMulti* jobs, int njobs, volatile uint32_t* flag, uint32_t want);
/// Blocks until every queued layer is done (the verify loop calls it before raising a flag itself and at a window's end).
void async_pool_drain();
/// Milliseconds the runner spent in the pool (cumulative; the decode timing's CPU figure in async mode).
double async_pool_ms();

}  // namespace strata::core
