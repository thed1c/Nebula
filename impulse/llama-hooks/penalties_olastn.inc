    // impulse (llama-hooks/penalties-olastn.cmake): the counted tokens only, when the array is in token-id order
    {
        bool direct = true;
        for (const auto & kv : ctx->token_count) {
            const llama_token t = kv.first;
            if (t < 0 || (size_t) t >= cur_p->size || cur_p->data[t].id != t) { direct = false; break; }
        }
        if (direct) {
            for (const auto & kv : ctx->token_count) {
                llama_token_data & d = cur_p->data[kv.first];
                const int count = kv.second;
                assert(count > 0 && count <= ctx->penalty_last_n);
                if (d.logit <= 0) {
                    d.logit *= ctx->penalty_repeat;
                } else {
                    d.logit /= ctx->penalty_repeat;
                }
                d.logit -= float(count) * ctx->penalty_freq + float(count > 0) * ctx->penalty_present;
            }
            cur_p->sorted = false;
            return;
        }
    }
