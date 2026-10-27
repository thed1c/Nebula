// impulse-server's API keys (src/lite/api_key.hpp): no key = open; a wrong or missing key = refused; the right key (Bearer or
// X-Api-Key, any of several, from --api-key or --api-key-file) = let through; /health, /v1/health and "/" open; constant-time compare.
#include "../src/lite/api_key.hpp"
#include <cstdio>
#include <fstream>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    lite::ApiKeys none;
    CHECK(!none.enabled());
    CHECK(none.accepts("", ""));                              // no key: unchanged behaviour
    lite::ApiKeys k;
    k.add_csv("s3cret,,other");
    CHECK(k.keys.size() == 2);
    CHECK(k.accepts("Bearer s3cret", ""));
    CHECK(k.accepts("Bearer other", ""));
    CHECK(k.accepts("s3cret", ""));                           // llama-server takes the key without "Bearer " too
    CHECK(k.accepts("", "s3cret"));                           // X-Api-Key
    CHECK(!k.accepts("", ""));
    CHECK(!k.accepts("Bearer s3cre", ""));
    CHECK(!k.accepts("Bearer s3cret2", ""));
    CHECK(!k.accepts("Bearer S3CRET", ""));
    CHECK(!k.accepts("Basic s3cret", ""));
    CHECK(lite::ApiKeys::public_path("/health") && lite::ApiKeys::public_path("/v1/health") && lite::ApiKeys::public_path("/"));
    CHECK(!lite::ApiKeys::public_path("/v1/chat/completions") && !lite::ApiKeys::public_path("/lora-adapters") &&
          !lite::ApiKeys::public_path("/completion") && !lite::ApiKeys::public_path("/v1/models"));
    const char * path = "api_key_test.keys";
    { std::ofstream f(path); f << "# a comment\nfilekey\r\n\nsecond\n"; }
    lite::ApiKeys fk;
    CHECK(fk.add_file(path).empty());
    CHECK(fk.keys.size() == 2 && fk.accepts("Bearer filekey", "") && fk.accepts("Bearer second", "") && !fk.accepts("Bearer # a comment", ""));
    CHECK(!fk.add_file("/nonexistent/keys").empty());
    std::remove(path);
    CHECK(lite::ApiKeys::same("abc", "abc") && !lite::ApiKeys::same("abc", "abd") && !lite::ApiKeys::same("abc", "abcd") && !lite::ApiKeys::same("", "a"));
    if (fails) { fprintf(stderr, "%d failure(s)\n", fails); return 1; }
    printf("api_key_test: all checks passed\n");
    return 0;
}
