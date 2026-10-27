// fx-tokenize: llama.cpp's tokenizer (a vocab-only load of a GGUF: the dense GGUF of a Flash-Next pack, or a GGUF holding
// only its metadata, tools/gguf_meta_only.py) on a batch of texts, for the tokenizer parity check of One Engine P1 (gate
// P1a, tests/fx_tokenizer_parity.py).  No GPU, no weights.
//
//   fx-tokenize MODEL.gguf IN OUT
//   IN:  records of [u8 parse_special][u32 little-endian byte length][the UTF-8 bytes]
//   OUT: one line per record, the ids separated by spaces (impulse-server's tokenization of a prompt: add_special true,
//        as tokenize_input_prompts does; the GGUF says add_bos_token false)
#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: fx-tokenize MODEL.gguf IN OUT\n"); return 2; }
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "cannot load the vocabulary of %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::ifstream in(argv[2], std::ios::binary);
    std::FILE * out = std::fopen(argv[3], "wb");
    if (!in || !out) { std::fprintf(stderr, "cannot open %s or %s\n", argv[2], argv[3]); return 1; }
    size_t n = 0;
    for (;;) {
        unsigned char ps = 0;
        uint32_t len = 0;
        if (!in.read((char *) &ps, 1)) break;
        if (!in.read((char *) &len, 4)) { std::fprintf(stderr, "a truncated record\n"); return 1; }
        std::string text(len, '\0');
        if (len && !in.read(text.data(), len)) { std::fprintf(stderr, "a truncated record\n"); return 1; }
        const std::vector<llama_token> ids = common_tokenize(vocab, text, /*add_special*/ true, /*parse_special*/ ps != 0);
        for (size_t i = 0; i < ids.size(); ++i) std::fprintf(out, i ? " %d" : "%d", ids[i]);
        std::fputc('\n', out);
        ++n;
    }
    std::fclose(out);
    std::fprintf(stderr, "fx-tokenize: %zu texts, vocabulary %d tokens\n", n, llama_vocab_n_tokens(vocab));
    llama_model_free(model);
    return 0;
}
