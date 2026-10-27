// src/fx/fx_vision.hpp - the image encoder for impulse-fx: serve/server.py's Vision (One Engine P2).
//
// `strata-vision` (llama.cpp's mtmd and the model's mmproj; the reference engine runs it on the CPU) as a resident child process:
// `ENC <image> <out>` per picture (both names relative to its working directory, #480), `OK <n tokens>` or `ERR ...`
// back; the embeddings file is cached by the picture's normalized bytes' hash (the first 32 hex digits of its SHA-256,
// 64 kept).  The picture: a data: URL or a local file (server.py's Vision.load); JPEG, PNG, BMP and GIF go in as they
// are, any other format (and a JPEG whose EXIF orientation is not 1) is converted to PNG by Pillow exactly as
// server.py's Vision.normalize does - here through `--fx-image-python` (a Python with Pillow; the reference engine's image has
// one), byte for byte the same PNG, so the same hash and the same embeddings.  An http(s) image URL is refused (no
// caller sends one; server.py downloads it).
//
// A request's pictures go to the engine as one file (`GENI`): their embeddings files one after the other, on /dev/shm
// when it has room (#874), deleted when the request ends.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace fxv {

struct Config {
    std::string exe, mmproj, model;
    bool gpu = false;
    int threads = 0, max_tokens = 0, min_tokens = 0;
    std::string python;           // a Python with Pillow, for the formats the encoder cannot read
};

class Vision {
public:
    explicit Vision(const Config & c);
    ~Vision();
    std::string start();                                                  // "" or why it did not start
    // -> (embeddings file, image tokens) for one source; throws fxf::BadRequest (the picture) / runtime_error (encoder)
    std::pair<std::string, int> encode(const std::string & source, const std::vector<std::string> & keep = {});
    std::string dir() const { return dir_; }
    bool alive() const;
    // the request's pictures in one file for GENI (combined_embeddings_path + write_temporary); "" on failure
    std::string combine(const std::vector<std::string> & files);
    static std::string load(const std::string & source);                   // Vision.load: the picture's bytes
    std::string normalize(const std::string & data) const;                // Vision.normalize
    static bool exif_rotated(const std::string & jpeg);                   // _exif_rotated (without Pillow)
private:
    Config cfg_;
    std::string dir_;
    std::mutex mu_;
    int pid_ = -1, in_ = -1, out_ = -1;
    std::string rbuf_;
    bool stopped_ = true;
    std::vector<std::pair<std::string, std::pair<std::string, int>>> cache_;   // key -> (file, n), oldest first
    std::string readline(double timeout_s, const std::string & what);
    void kill_child();
};

std::string sha256_hex(const std::string & data);

}  // namespace fxv
