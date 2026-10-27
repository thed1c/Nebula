// impulse-fx's entry (src/fx/CMakeLists.txt): impulse-server execs this binary for a Flash-Next pack.
namespace fxsrv { int run(int argc, char ** argv); }
int main(int argc, char ** argv) { return fxsrv::run(argc, argv); }
