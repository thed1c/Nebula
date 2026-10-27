// include/strata/program/entry.hpp - the engine's entry as a function (One Engine P0).
//
// src/program/generate.cpp built with STRATA_ENGINE_LIBRARY (the strata_generate library) names its main strata_main;
// the standalone `strata` is src/program/strata_main.cpp, which only calls it, and another program (impulse-server's
// fx engine, P1) links strata_generate and calls it the same way.  Same arguments, same exit code as `strata`.
#pragma once

#include <cstdio>

int strata_main(int argc, char** argv);

// One Engine P1: where --serve reads its requests and writes its lines (the protocol serve/server.py speaks).  The
// standalone `strata` leaves both alone: stdin (descriptor 0) and stdout.  libimpulse_fx (src/program/fx_bridge.cpp)
// sets them to its pipes before strata_main, so the engine runs inside impulse-server without touching the process's
// own stdin / stdout.  `out` == nullptr: stdout.
void strata_set_serve_io(int in_fd, std::FILE* out);
