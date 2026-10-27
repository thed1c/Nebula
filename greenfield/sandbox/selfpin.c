// nebula selfpin (Nebula, for engines in OpenShell sandboxes): inside a sandbox, sched_setaffinity is allowed only on
// "this thread" written as pid 0; a thread's own id (what pthread_setaffinity_np(pthread_self()) passes) or another
// thread's id is refused (EPERM). Engines such as Strata pin each worker from inside it with its own id. This preload
// retries those calls in the allowed form: a thread's own id as 0, and another thread's through a real-time signal
// whose handler, in that thread, pins it with 0. Nothing else changes; threads are created as usual.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define SIG (SIGRTMIN + 7)
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static cpu_set_t want; static size_t want_size;
static atomic_int done, installed; static int result, result_errno;

static void handler(int sig) {
    (void)sig;
    int e = errno;
    result = (int)syscall(SYS_sched_setaffinity, 0, want_size, &want); result_errno = errno;
    atomic_store(&done, 1);
    errno = e;
}

static void install(void) {
    if (atomic_exchange(&installed, 1)) return;
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_handler = handler; sa.sa_flags = SA_RESTART; sigemptyset(&sa.sa_mask);
    sigaction(SIG, &sa, NULL);
}

static int self_pin(size_t size, const cpu_set_t *mask) {          // 0, or an error number
    return syscall(SYS_sched_setaffinity, 0, size, mask) == 0 ? 0 : errno;
}

static int other_pin(int by_pthread, pthread_t th, pid_t tid, size_t size, const cpu_set_t *mask) {
    install();
    pthread_mutex_lock(&lock);
    if (size > sizeof want) size = sizeof want;
    memset(&want, 0, sizeof want); memcpy(&want, mask, size); want_size = size;
    atomic_store(&done, 0);
    int err = by_pthread ? pthread_kill(th, SIG) : (syscall(SYS_tgkill, getpid(), tid, SIG) == 0 ? 0 : errno);
    if (!err) {
        for (int i = 0; i < 400 && !atomic_load(&done); i++) { struct timespec ts = {0, 500000}; nanosleep(&ts, NULL); }
        err = atomic_load(&done) ? (result == 0 ? 0 : result_errno) : EAGAIN;   // a thread blocking signals: EAGAIN
    }
    pthread_mutex_unlock(&lock);
    return err;
}

int sched_setaffinity(pid_t pid, size_t size, const cpu_set_t *mask) {
    if (syscall(SYS_sched_setaffinity, pid, size, mask) == 0) return 0;
    if (errno != EPERM) return -1;
    int err = (pid == 0 || pid == (pid_t)syscall(SYS_gettid)) ? self_pin(size, mask) : other_pin(0, 0, pid, size, mask);
    if (err) { errno = err; return -1; }
    return 0;
}

int pthread_setaffinity_np(pthread_t th, size_t size, const cpu_set_t *mask) {
    static int (*real)(pthread_t, size_t, const cpu_set_t *);
    if (!real) real = dlsym(RTLD_NEXT, "pthread_setaffinity_np");
    int r = real(th, size, mask);
    if (r != EPERM) return r;
    return pthread_equal(th, pthread_self()) ? self_pin(size, mask) : other_pin(1, th, 0, size, mask);
}
