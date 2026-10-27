// src/net/crypto.hpp - the transport's crypto pieces (internal): HKDF-SHA256, HMAC-SHA256, random bytes, and one direction's
// AES-256-GCM frame state.  Shared by the TCP channel (transport.cpp) and the RDMA channel (rdma.cpp).
#pragma once

#include "strata/net/transport.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(STRATA_NET_CRYPTO)
#include <openssl/core_names.h>
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>
#include <openssl/params.h>
#include <openssl/rand.h>
#endif

namespace strata::net::detail {

constexpr size_t kKey = 32, kTag = 16;

#if defined(STRATA_NET_CRYPTO)
inline bool hkdf(const std::vector<uint8_t>& ikm, const uint8_t* salt, size_t salt_n, const char* info, uint8_t* out, size_t n) {
    EVP_KDF* kdf = EVP_KDF_fetch(nullptr, "HKDF", nullptr);
    if (kdf == nullptr) return false;
    EVP_KDF_CTX* ctx = EVP_KDF_CTX_new(kdf);
    EVP_KDF_free(kdf);
    if (ctx == nullptr) return false;
    OSSL_PARAM p[5];
    char digest[] = "SHA256";
    p[0] = OSSL_PARAM_construct_utf8_string(OSSL_KDF_PARAM_DIGEST, digest, 0);
    p[1] = OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_KEY, (void*) ikm.data(), ikm.size());
    p[2] = OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_SALT, (void*) salt, salt_n);
    p[3] = OSSL_PARAM_construct_octet_string(OSSL_KDF_PARAM_INFO, (void*) info, std::strlen(info));
    p[4] = OSSL_PARAM_construct_end();
    const bool ok = EVP_KDF_derive(ctx, out, n, p) == 1;
    EVP_KDF_CTX_free(ctx);
    return ok;
}
inline bool hmac256(const uint8_t* key, size_t key_n, const std::vector<uint8_t>& msg, uint8_t out[32]) {
    unsigned int n = 32;
    return HMAC(EVP_sha256(), key, (int) key_n, msg.data(), msg.size(), out, &n) != nullptr && n == 32;
}
#endif

inline bool random_bytes(uint8_t* p, size_t n) {
#if defined(STRATA_NET_CRYPTO)
    return RAND_bytes(p, (int) n) == 1;
#else
    std::ifstream f("/dev/urandom", std::ios::binary);
    return f.read((char*) p, (std::streamsize) n).good();
#endif
}

// one direction's AES-256-GCM state: the key and the frame counter (the nonce)
struct Gcm {
#if defined(STRATA_NET_CRYPTO)
    EVP_CIPHER_CTX* ctx = nullptr;
#endif
    bool on = false, enc = true;
    uint64_t counter = 0;
    ~Gcm() {
#if defined(STRATA_NET_CRYPTO)
        if (ctx != nullptr) EVP_CIPHER_CTX_free(ctx);
#endif
    }
    bool init(const uint8_t* key, bool encrypt) {
#if defined(STRATA_NET_CRYPTO)
        ctx = EVP_CIPHER_CTX_new();
        if (ctx == nullptr) return false;
        enc = encrypt;
        const int r = encrypt ? EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr)
                              : EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr);
        if (r != 1 || EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) != 1) return false;
        if ((encrypt ? EVP_EncryptInit_ex(ctx, nullptr, nullptr, key, nullptr) : EVP_DecryptInit_ex(ctx, nullptr, nullptr, key, nullptr)) != 1)
            return false;
        on = true;
        return true;
#else
        (void) key; (void) encrypt;
        return false;
#endif
    }
    void iv(uint8_t out[12]) const {
        std::memset(out, 0, 12);
        for (int i = 0; i < 8; ++i) out[4 + i] = (uint8_t) (counter >> (8 * i));
    }
#if defined(STRATA_NET_CRYPTO)
    // pieces -> ciphertext + tag into `out`
    bool seal(const uint8_t* aad, size_t aad_n, const Iov* pieces, int n, std::vector<uint8_t>& out) {
        uint8_t nonce[12];
        iv(nonce);
        int len = 0;
        if (EVP_EncryptInit_ex(ctx, nullptr, nullptr, nullptr, nonce) != 1 || EVP_EncryptUpdate(ctx, nullptr, &len, aad, (int) aad_n) != 1)
            return false;
        size_t total = 0;
        for (int i = 0; i < n; ++i) total += pieces[i].size;
        out.resize(total + kTag);
        size_t off = 0;
        for (int i = 0; i < n; ++i) {
            const uint8_t* p = (const uint8_t*) pieces[i].data;
            size_t left = pieces[i].size;
            while (left > 0) {   // EVP takes an int length
                const int step = (int) std::min<size_t>(left, 1u << 30);
                if (EVP_EncryptUpdate(ctx, out.data() + off, &len, p, step) != 1) return false;
                off += (size_t) len; p += step; left -= (size_t) step;
            }
        }
        if (EVP_EncryptFinal_ex(ctx, out.data() + off, &len) != 1) return false;
        off += (size_t) len;
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, (int) kTag, out.data() + off) != 1) return false;
        ++counter;
        return off == total;
    }
    bool open(const uint8_t* aad, size_t aad_n, uint8_t* data, size_t n, const uint8_t* tag) {
        uint8_t nonce[12];
        iv(nonce);
        int len = 0;
        if (EVP_DecryptInit_ex(ctx, nullptr, nullptr, nullptr, nonce) != 1 || EVP_DecryptUpdate(ctx, nullptr, &len, aad, (int) aad_n) != 1)
            return false;
        size_t off = 0;
        while (off < n) {
            const int step = (int) std::min<size_t>(n - off, 1u << 30);
            if (EVP_DecryptUpdate(ctx, data + off, &len, data + off, step) != 1) return false;
            off += (size_t) step;
        }
        if (EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, (int) kTag, (void*) tag) != 1) return false;
        const bool ok = EVP_DecryptFinal_ex(ctx, data + n, &len) == 1;
        ++counter;
        return ok;
    }
#endif
};


}  // namespace strata::net::detail
