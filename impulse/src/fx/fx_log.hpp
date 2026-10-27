// impulse-fx's engine log: the run config's "log", as serve/server.py's StrataEngine keeps it.
//
// server.py opens the log in append mode ("a": never truncated, across restarts), writes one line
// "[strata] <YYYY-mm-dd HH:MM:SS> engine started: <exe> --serve <args>" and gives the engine process the log as its
// stderr; the vision helper's stderr goes there too.  Its own lines go to stdout, not to the log.
//
// Here the engine runs on a thread of this process, so its stderr is this process's: open(log) - dup2 onto fd 2 sends
// the engine's stderr, the vision helper's (it inherits fd 2) and llama.cpp's to the log, as with server.py.  The
// front's own lines (fx::say) go to the log too AND to the console (the stderr this process was started with), so the
// container's output still shows what the front says, as server.py's stdout did, and the log also holds them (the P3
// cut-over greps "impulse fx" in /sandbox/engine.log).  Without "log" nothing changes: everything on stderr.
#pragma once

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fx {

inline int & console_fd() { static int fd = -1; return fd; }   // the original stderr once the log has fd 2; else -1

inline void write_fd(int fd, const char * p, size_t n) {
    while (n > 0) {
#if defined(_WIN32)
        const int w = _write(fd, p, (unsigned) n);
#else
        const ssize_t w = ::write(fd, p, n);
#endif
        if (w <= 0) return;
        p += w; n -= (size_t) w;
    }
}

// the front's own line: stderr (the log, when there is one) and the console
inline void say(const char * fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::string big;
    if (n >= (int) sizeof buf) {   // a long line (an engine line echoed): formatted again, whole
        big.resize((size_t) n + 1);
        va_start(ap, fmt);
        vsnprintf(&big[0], big.size(), fmt, ap);
        va_end(ap);
        big.resize((size_t) n);
    }
    if (n < 0) return;
    const char * p = big.empty() ? buf : big.c_str();
    fflush(stderr);
    write_fd(2, p, (size_t) n);
    if (console_fd() >= 0) write_fd(console_fd(), p, (size_t) n);
}

// "log": fd 2 becomes the log (appended, created if missing), the console is kept for say().  "" or why it failed.
inline std::string open_engine_log(const std::string & path) {
    fflush(stderr);
#if defined(_WIN32)
    const int fd = _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_APPEND | _O_BINARY, _S_IREAD | _S_IWRITE);
    if (fd < 0) return "cannot open the engine log " + path + ": " + std::strerror(errno);
    const int con = _dup(2);
    if (con < 0 || _dup2(fd, 2) != 0) { _close(fd); return "cannot send stderr to the engine log " + path; }
    _close(fd);
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) return "cannot open the engine log " + path + ": " + std::strerror(errno);
    const int con = fcntl(2, F_DUPFD_CLOEXEC, 3);
    if (con < 0 || dup2(fd, 2) < 0) { ::close(fd); return "cannot send stderr to the engine log " + path; }
    ::close(fd);
#endif
    console_fd() = con;
    return "";
}

// server.py's "[strata] <time> engine started: <exe> --serve <args>" line (#779: a crash report has the time): the log only
inline void log_engine_started(const std::string & exe, const std::string & args) {
    if (console_fd() < 0) return;
    char t[32];
    const std::time_t now = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    std::strftime(t, sizeof t, "%Y-%m-%d %H:%M:%S", &tmv);
    const std::string line = "[strata] " + std::string(t) + " engine started: " + exe + " " + args + "\n";
    fflush(stderr);
    write_fd(2, line.data(), line.size());
}

}  // namespace fx
