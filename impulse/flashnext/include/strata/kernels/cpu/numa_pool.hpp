// include/strata/kernels/cpu/numa_pool.hpp - when the CPU expert pool splits per NUMA node (0.2.2, from upstream Strata;
// numa-always: the layout is found inside sandboxes too).
//
// Pure decisions, no threads and no globals, so a test can feed them a made-up machine:
//   * numa_cpu_node_count_at(dir)  how many NUMA nodes with CPUs a sysfs node folder lists (Linux: /sys/devices/system/node;
//                                  Windows reads GetNumaHighestNodeNumber in pool.cpp; every other system: 1)
//   * pick_numa_pool(...)          whether the split is tried without STRATA_NUMA_SPLIT, and the line that says so
//   * decide_numa_split(...)       the split's plan for a pool whose threads sit on given cores
//
// numa-always (this fork): a sandbox that hides /sys or shows one node no longer turns the split off by itself.  Where each
// CPU's node comes from, first that answers:
//   1. IMPULSE_NUMA_NODES (or STRATA_NUMA_NODES): the launcher states the layout, "0=0-17,36-53;1=18-35,54-71"
//      (parse_numa_nodes; Nebula writes it for an engine whose CPUs span two sockets)
//   2. sysfs: /sys/devices/system/cpu/cpu<N>/node<M> (the machine's own layout, when the sandbox shows it)
//   3. getcpu(): a short-lived thread pinned to each allowed CPU asks the kernel which node it is on; the kernel answers
//      from its own per-CPU data, so no file has to be visible (resolve_numa_layout's `probe`)
// The machine's node count is the larger of the sysfs node folder's and the nodes the layout found.
//
// One node is exactly the code path before the split existed: pick_numa_pool answers `Unchanged` with no log line and the pool
// never reads anything else from here.  STRATA_NUMA_POOL=0 is the off switch of the default; STRATA_NUMA_SPLIT=1 / test keep
// their meaning from the contributor's branch (#3201).
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace strata::kernels::cpu {

/// rows stay 50/50, so a slice with under half the other's threads would be the straggler of every batch
inline bool numa_slices_balanced(int a, int b) { return std::min(a, b) * 2 >= std::max(a, b); }

/// A cpulist ("0-17,36-53"): the CPUs in it, or false when the text is not one.
inline bool parse_cpulist(const std::string& text, std::vector<int>& out) {
    out.clear();
    size_t i = 0;
    auto skip_space = [&] { while (i < text.size() && std::isspace((unsigned char) text[i])) ++i; };
    auto number = [&](long& v) {
        skip_space();
        const size_t start = i;
        v = 0;
        while (i < text.size() && std::isdigit((unsigned char) text[i]) && v < 1000000) v = v * 10 + (text[i++] - '0');
        return i > start && v < 1000000;
    };
    skip_space();
    if (i == text.size()) return false;
    while (i < text.size()) {
        long lo = 0, hi = 0;
        if (!number(lo)) return false;
        hi = lo;
        skip_space();
        if (i < text.size() && text[i] == '-') {
            ++i;
            if (!number(hi) || hi < lo) return false;
        }
        for (long c = lo; c <= hi; ++c) out.push_back((int) c);
        skip_space();
        if (i < text.size()) {
            if (text[i] != ',') return false;
            ++i;
        }
    }
    return !out.empty();
}

/// IMPULSE_NUMA_NODES / STRATA_NUMA_NODES: "<node>=<cpulist>" entries separated by ';' ("0=0-17,36-53;1=18-35,54-71";
/// ':' also works for '=', and "node0" for "0").  `cpu_node` gets cpu -> node.  An error names what is wrong ("" = ok).
inline std::string parse_numa_nodes(const std::string& text, std::map<int, int>& cpu_node) {
    cpu_node.clear();
    size_t at = 0;
    while (at <= text.size()) {
        const size_t end = std::min(text.find(';', at), text.size());
        std::string item = text.substr(at, end - at);
        at = end + 1;
        item.erase(0, item.find_first_not_of(" \t\n"));
        item.erase(item.find_last_not_of(" \t\n") + 1);
        if (item.empty()) {
            if (end == text.size()) break;
            continue;
        }
        const size_t eq = item.find_first_of("=:");
        if (eq == std::string::npos) return "'" + item + "' has no '=' (use node=cpus, such as 0=0-17,36-53)";
        std::string id = item.substr(0, eq);
        id.erase(id.find_last_not_of(" \t") + 1);
        if (id.compare(0, 4, "node") == 0) id.erase(0, 4);
        if (id.empty() || id.size() > 4 || !std::all_of(id.begin(), id.end(), [](unsigned char c) { return std::isdigit(c) != 0; }))
            return "'" + item + "': the node is not a number";
        const int node = std::stoi(id);
        std::vector<int> cpus;
        if (!parse_cpulist(item.substr(eq + 1), cpus)) return "'" + item + "': the CPUs are not a list such as 0-17,36-53";
        for (int c : cpus) {
            const auto [it, fresh] = cpu_node.emplace(c, node);
            if (!fresh && it->second != node)
                return "CPU " + std::to_string(c) + " is listed on node " + std::to_string(it->second) + " and node " +
                       std::to_string(node);
        }
        if (end == text.size()) break;
    }
    if (cpu_node.empty()) return "no node listed";
    return "";
}

/// Where the pool's NUMA layout came from, and each CPU's node.
struct NumaLayout {
    std::map<int, int> cpu_node;   ///< the CPUs that have a known node
    std::string source;            ///< "IMPULSE_NUMA_NODES", "sysfs", "getcpu", "sysfs + getcpu" or "" (nothing known)
    std::string error;             ///< a layout setting that could not be read (the pool then reads the machine)
    int machine_nodes = 0;         ///< NUMA nodes with CPUs on the machine (sysfs folder, else the nodes found)
    int node_of(int cpu) const {
        const auto it = cpu_node.find(cpu);
        return it == cpu_node.end() ? -1 : it->second;
    }
    /// The nodes the CPUs in `cpus` are on (unknown CPUs are not counted).
    std::set<int> nodes_of(const std::vector<int>& cpus) const {
        std::set<int> s;
        for (int c : cpus)
            if (node_of(c) >= 0) s.insert(node_of(c));
        return s;
    }
};

/// The layout for the CPUs this process may use (`allowed`).  `env` is IMPULSE_NUMA_NODES (else STRATA_NUMA_NODES; nullptr or
/// "" = unset); `sysfs_node(cpu)` reads /sys/devices/system/cpu/cpu<N>/node<M> (-1 = none); `sysfs_nodes` is the node folder's
/// count (numa_cpu_node_count_at, 0 = hidden); `probe(cpu)` is getcpu() on a thread pinned to that CPU (-1 = could not pin).
/// The probe runs only for allowed CPUs sysfs does not place: a machine whose sysfs is visible reads exactly what it read
/// before, and the probe never changes a thread of the engine (it runs on a thread of its own).
inline NumaLayout resolve_numa_layout(const char* env, const std::vector<int>& allowed, int sysfs_nodes,
                                      const std::function<int(int)>& sysfs_node, const std::function<int(int)>& probe) {
    NumaLayout L;
    if (env != nullptr && *env != '\0') {
        L.error = parse_numa_nodes(env, L.cpu_node);
        if (L.error.empty()) {
            L.source = "IMPULSE_NUMA_NODES";
            std::set<int> nodes;
            for (const auto& kv : L.cpu_node) nodes.insert(kv.second);
            L.machine_nodes = (std::max)((int) nodes.size(), sysfs_nodes);
            return L;
        }
        L.cpu_node.clear();
    }
    std::vector<int> unknown;
    for (int c : allowed) {
        const int n = sysfs_node ? sysfs_node(c) : -1;
        if (n >= 0) L.cpu_node[c] = n;
        else unknown.push_back(c);
    }
    bool probed = false;
    // sysfs placed every allowed CPU: that is the whole answer (the machine as before)
    if (!unknown.empty() && probe) {
        for (int c : unknown) {
            const int n = probe(c);
            if (n >= 0) {
                L.cpu_node[c] = n;
                probed = true;
            }
        }
    }
    const bool from_sysfs = L.cpu_node.size() > (size_t) 0 && unknown.size() < allowed.size();
    L.source = from_sysfs && probed ? "sysfs + getcpu" : from_sysfs ? "sysfs" : probed ? "getcpu" : "";
    std::set<int> nodes;
    for (const auto& kv : L.cpu_node) nodes.insert(kv.second);
    L.machine_nodes = (std::max)((int) nodes.size(), sysfs_nodes);
    return L;
}

/// The worker cores reordered round-robin over their nodes (in first-seen order, stable inside each node), so any count
/// under the core count spans the nodes evenly.  A core with no known node keeps a group of its own.  One node: unchanged.
inline std::vector<int> interleave_by_node(const std::vector<int>& cores, const std::function<int(int)>& node_of) {
    std::vector<int> ids;
    std::vector<std::vector<int>> of;
    for (int c : cores) {
        const int key = node_of(c) >= 0 ? node_of(c) : -1000 - c;
        const size_t k = (size_t) (std::find(ids.begin(), ids.end(), key) - ids.begin());
        if (k == ids.size()) {
            ids.push_back(key);
            of.emplace_back();
        }
        of[k].push_back(c);
    }
    if (ids.size() < 2) return cores;
    std::vector<int> out;
    for (size_t r = 0; out.size() < cores.size(); ++r)
        for (const auto& v : of)
            if (r < v.size()) out.push_back(v[r]);
    return out;
}

/// Whether text is "0", "off", "false" or "no" (any case): the spelling of every off switch of the pool.
inline bool numa_switch_off(const char* v) {
    if (v == nullptr) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s == "0" || s == "off" || s == "false" || s == "no";
}

/// The NUMA nodes with CPUs under a sysfs node folder (`node0`, `node1`, ... each holding a `cpulist`). A node without CPUs
/// (CXL or other memory-only nodes) is not counted: the pool splits threads, and a node with none has none to split.
/// A folder that is missing or has no node entries counts 0 (the caller treats 0 and 1 alike: no split).
inline int numa_cpu_node_count_at(const std::string& node_dir) {
    std::error_code ec;
    int nodes = 0;
    for (std::filesystem::directory_iterator it(node_dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().string();
        if (name.size() < 5 || name.compare(0, 4, "node") != 0) continue;
        if (!std::all_of(name.begin() + 4, name.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) continue;
        std::ifstream f(it->path() / "cpulist");
        std::string list;
        std::getline(f, list);
        if (list.find_first_of("0123456789") != std::string::npos) ++nodes;
    }
    return nodes;
}

enum class NumaPoolChoice {
    Unchanged,   ///< nothing new: STRATA_NUMA_SPLIT (set or not) decides as it did before 0.2.2
    Off,         ///< STRATA_NUMA_POOL=0 on a machine that matters: the split is off, whatever STRATA_NUMA_SPLIT says
    Auto,        ///< two or more NUMA nodes and nothing set: try the split by itself; if it cannot run the pool stays as it was
};
struct NumaPoolPick {
    NumaPoolChoice choice = NumaPoolChoice::Unchanged;
    std::string log;   ///< the line to print ("" = print nothing)
};

/// `pool_env` is STRATA_NUMA_POOL, `split_env` STRATA_NUMA_SPLIT (nullptr or "" = unset), `nodes` the NUMA nodes with CPUs.
///   one node, nothing set                       -> Unchanged, silent (today's code path to the instruction)
///   STRATA_NUMA_POOL=0, two or more nodes       -> Off, with a line that says so
///   STRATA_NUMA_POOL=0, STRATA_NUMA_SPLIT set   -> Off (the off switch wins), with a line
///   STRATA_NUMA_SPLIT set (1, test, 0, ...)     -> Unchanged: the contributor's variable keeps its old meaning
///   two or more nodes, nothing set              -> Auto
inline NumaPoolPick pick_numa_pool(const char* pool_env, const char* split_env, int nodes) {
    const bool split_set = split_env != nullptr && *split_env != '\0';
    NumaPoolPick pick;
    if (numa_switch_off(pool_env) && (nodes >= 2 || split_set)) {
        pick.choice = NumaPoolChoice::Off;
        pick.log = "strata cpu pool: NUMA pool off (STRATA_NUMA_POOL=0): " +
                   (nodes >= 2 ? std::to_string(nodes) + " NUMA nodes found" : std::string("one NUMA node")) +
                   ", the pool runs as before";
        return pick;
    }
    if (split_set || nodes < 2) return pick;
    pick.choice = NumaPoolChoice::Auto;
    return pick;
}

struct NumaSplitDecision {
    bool on = false;
    std::string why;               ///< the reason when off
    int nodes[2] = {-1, -1};       ///< the two node ids, ascending
    std::vector<int> slice;        ///< each thread's slice (0 or 1)
    int per[2] = {0, 0};           ///< threads per slice
};

/// The real-node split of a pool whose threads sit on `thread_core` (workers first, then the host when it drains; -1 = no core).
/// `n_workers` tells a worker from the host in the messages; `pinned` is whether the pool pins its threads; `rotate` is
/// STRATA_EXCHANGE_ROTATE=1 (experts move between slots, the split needs fixed ones); `node_of(core)` is the core's NUMA node or -1.
inline NumaSplitDecision decide_numa_split(const std::vector<int>& thread_core, int n_workers, bool pinned, bool rotate,
                                           const std::function<int(int)>& node_of) {
    NumaSplitDecision d;
    const int threads = (int) thread_core.size();
    std::vector<int> node((size_t) threads, -1);
    if (threads < 2) {
        d.why = "the pool has " + std::to_string(threads) + " thread, the split needs 2 or more";
        return d;
    }
    if (rotate) {
        d.why = "STRATA_EXCHANGE_ROTATE=1 moves experts between slots, the split needs fixed slots";
        return d;
    }
    if (!pinned) {
        d.why = "the pool's threads are not pinned";
        return d;
    }
    for (int j = 0; j < threads; ++j) {
        const int c = thread_core[(size_t) j];
        if (c < 0) {
            d.why = j < n_workers ? "worker " + std::to_string(j) + " has no core to be pinned to" : "the host thread has no core";
            return d;
        }
        if ((node[(size_t) j] = node_of(c)) < 0) {
            d.why = "CPU " + std::to_string(c) + " has no known NUMA node";
            return d;
        }
    }
    std::vector<int> nodes = node;
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
    if (nodes.size() != 2) {
        d.why = "the pool's threads span " + std::to_string(nodes.size()) + " NUMA node(s), the NUMA split supports exactly 2";
        return d;
    }
    d.nodes[0] = nodes[0];
    d.nodes[1] = nodes[1];
    d.slice.assign((size_t) threads, 0);
    for (int j = 0; j < threads; ++j) {
        d.slice[(size_t) j] = node[(size_t) j] == nodes[1] ? 1 : 0;
        ++d.per[d.slice[(size_t) j]];
    }
    if (!numa_slices_balanced(d.per[0], d.per[1])) {
        d.why = "slices too unbalanced (node " + std::to_string(nodes[0]) + ": " + std::to_string(d.per[0]) + " thread" +
                (d.per[0] == 1 ? "" : "s") + ", node " + std::to_string(nodes[1]) + ": " + std::to_string(d.per[1]) + " thread" +
                (d.per[1] == 1 ? "" : "s") + ")";
        d.nodes[0] = d.nodes[1] = -1;
        d.slice.clear();
        d.per[0] = d.per[1] = 0;
        return d;
    }
    d.on = true;
    return d;
}

}  // namespace strata::kernels::cpu
