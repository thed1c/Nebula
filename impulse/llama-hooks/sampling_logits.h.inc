
// ---- Impulse (impulse/llama-hooks): appended at configure time by impulse/CMakeLists.txt ----
// Impulse: common_sampler_sample on a logits vector of n_vocab floats instead of a llama_context row
// (the same reasoning budget, grammar check and resampling, and sampler chain)
llama_token common_sampler_sample_logits(struct common_sampler * gsmpl, const float * logits, int n_vocab, bool grammar_first = false);
