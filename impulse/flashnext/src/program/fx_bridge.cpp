// src/program/fx_bridge.cpp - libimpulse_fx's whole interface (One Engine P1, step 1: the bridge).
//
// impulse-server loads this library (dlopen, RTLD_LOCAL) for a Flash-Next pack and runs the engine's --serve loop in
// its own process, on a thread of its own: impulse_fx_run is strata_main with the serve loop's input and output on two
// pipes instead of stdin / stdout (strata_set_serve_io), so the requests and the lines are exactly the protocol
// serve/server.py speaks (GEN / GENI / BGEN / STOP / SAVE / RESTORE / VRAM / QUIT in; READY, INFO, T, PP, RESUME,
// DONE, ERR ... out).  Nothing of the engine changes: the same arguments as `strata`, the same code.
//
// The handles are the OS's (a descriptor on Linux / macOS, a HANDLE on Windows), not C runtime descriptors: this library
// may carry a C runtime of its own (Windows), whose descriptor table is not the caller's.
#include "strata/program/entry.hpp"

#include <cstdint>
#include <cstdio>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#define IMPULSE_FX_API extern "C" __declspec(dllexport)
#else
#include <unistd.h>
#define IMPULSE_FX_API extern "C" __attribute__((visibility("default")))
#endif

// the interface's version: impulse-server refuses a library whose version it does not know
IMPULSE_FX_API int impulse_fx_abi(void) { return 1; }

// Runs the engine: `argv` as `strata`'s (argv[0] a name, then e.g. --serve --pack ... --native ...), its requests read
// from `in_handle` and its serve lines written to `out_handle`.  Returns strata_main's exit code once the serve loop
// ends (QUIT, or the input closed); the output is closed then, so a reader sees its end.  Once per process: the engine
// keeps process-wide state (CUDA, the expert pool, its threads).
IMPULSE_FX_API int impulse_fx_run(int argc, char** argv, std::intptr_t in_handle, std::intptr_t out_handle) {
#if defined(_WIN32)
    const int in_fd = _open_osfhandle(in_handle, _O_RDONLY | _O_BINARY);
    const int out_fd = _open_osfhandle(out_handle, _O_WRONLY | _O_BINARY);
#else
    const int in_fd = (int) in_handle;
    const int out_fd = (int) out_handle;
#endif
    if (in_fd < 0 || out_fd < 0) return 3;
#if defined(_WIN32)
    std::FILE* out = _fdopen(out_fd, "wb");
#else
    std::FILE* out = fdopen(out_fd, "w");
#endif
    if (out == nullptr) return 3;
    strata_set_serve_io(in_fd, out);
    const int rc = strata_main(argc, argv);
    std::fflush(out);
    strata_set_serve_io(0, nullptr);
    std::fclose(out);
    return rc;
}
