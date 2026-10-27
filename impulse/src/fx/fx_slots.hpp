// impulse-fx: several requests at once in the engine's batch slots ("parallel": N -> the engine's --batch N).
//
// A port of Strata's serve/server.py StrataEngine.generate_batched / pick_slot (flashnext/docs/BATCHING.md):
//   - a request alone (no slot busy, nobody waiting) runs on the solo path: `GEN` / `GENI`, with the MTP drafts, the
//     fastest single stream;
//   - when another request arrives the solo request is STOPped and continues in a batch slot: `BGEN <slot> <left> ...`
//     with its prompt plus what it generated (the engine's prompt cache holds exactly that: nothing is read again), then
//     the slot's own `BT <slot> <id>` lines until `BDONE <slot> ...`;
//   - a request left alone in a slot goes back to the solo path with its drafts (INFO slot_cache=1; at most twice);
//   - the control lines (a prompt read, an admission) are held by one request at a time, given in arrival order (FIFO
//     tickets: fair queueing when every slot is busy); a request waiting for a free slot gives them back meanwhile;
//   - a slot is picked by the conversation it holds (the longest held start of the prompt), else an empty one, else
//     the one used longest ago; with pipelined groups (--batch-groups) holes below a busy slot first, then spread;
//   - `ERR BGEN-LONG` (prompt + max_tokens do not fit a slot, --batch-max-context): the request runs alone on the solo
//     path once the slots are idle; `ERR BGEN:` (the engine's view of a slot differs for a moment): the slot is parked
//     for 2 s and another tried, at most 6 times;
//   - a consumer that stops early (stop token, stop string, a client that left) STOPs the solo pass / BSTOPs its slot
//     and reads to the DONE / BDONE, so the engine and this server stay in step.
// Not ported (yet): BYIELD (a long prompt read giving way to a shorter waiting one), server.py's orphan-slot reaper
// (every slot here is released by the request that claimed it, synchronously).
//
// The engine side is an interface (Wire): impulse-fx's Engine (src/fx/fx_server.cpp) and the unit test's fake engine
// (tests/fx_slots_test.cpp) implement it.  No llama.cpp, no JSON: the DONE / BDONE lines go back as they are.
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fxs {

struct EngineDied : std::runtime_error { using std::runtime_error::runtime_error; };
struct EngineSilent : EngineDied { using EngineDied::EngineDied; };
struct EngineNonFinite : std::runtime_error { using std::runtime_error::runtime_error; };
struct EngineErr : std::runtime_error { using std::runtime_error::runtime_error; };   // the engine's ERR (server.py: ValueError)

// what the scheduler needs of the engine's two pipes
struct Wire {
    virtual ~Wire() = default;
    virtual bool send(const std::string & line) = 0;
    virtual void ctl_open() = 0;                                     // this request reads the control lines from now
    virtual void ctl_close() = 0;                                    // ... no longer (unread lines go to the log)
    virtual bool ctl_next(std::string & out, double seconds) = 0;    // false: nothing in time (or the engine ended)
    virtual bool slot_next(int slot, std::string & out, double seconds) = 0;   // BT / BDONE of that slot
    virtual void slot_clear(int slot) = 0;
    virtual bool ended() = 0;
    virtual bool can_stop() = 0;                                     // READY <ctx> stop
};

struct Config {
    int slots = 0;                 // INFO batch_slots
    bool slot_cache = false;       // INFO slot_cache: a slot's sessions can go back to the solo path
    long long batch_ctx = 0;       // INFO batch_ctx: the cells a slot holds (0: the whole context)
    int groups = 1;                // INFO batch_groups (pipelined slot groups on a layer split)
    double silence_s = 300.0;      // ENGINE_SILENCE_S (0: no limit)
    std::set<int32_t> eos;         // the ids that end a reply (im_end, endoftext)
};

struct Request {
    long long max_new = 0;
    std::string keys;              // " key=value ..." (fxf::sampling_keys), may be empty
    std::string embeddings;        // GENI's file ("" = text only)
    std::vector<int32_t> ids;
};

struct Hooks {
    std::function<bool(int32_t)> token;      // false: the consumer stops here (the pass is ended and drained)
    std::function<void()> ping;              // a heartbeat: 10 s without a token (a queued or reading request)
    std::function<bool()> cancelled;         // the client left
};

// one engine leg of a request: the DONE line of its pass or admission and, for a slot leg, the slot's BDONE line
struct Leg { std::string done, bdone; };
struct Result { std::vector<Leg> legs; long long generated = 0; };

struct SlotView { int id; bool busy; const char * state; long long prompt_tokens, generated, held_tokens; };

static const double PP_CHUNK_MAX_S = 32768, PP_FLOOR_TOK_S_S = 50.0, PP_SLACK_S = 3.0;

class Scheduler {
  public:
    Scheduler(Wire & w, Config c, std::function<void(const std::string &)> log = nullptr)
        : W(w), C(std::move(c)), log_(std::move(log)) {
        const int n = std::max(0, C.slots);
        busy_.assign((size_t) n, false); held_.assign((size_t) n, {}); used_.assign((size_t) n, 0);
        parked_.assign((size_t) n, Clock::time_point{});
        live_.assign((size_t) n, Live{});
        const int g = C.groups > 0 && n > 0 && n % C.groups == 0 ? C.groups : 1;
        groups_ = g;
        gs_ = n > 0 ? n / g : 0;
        for (int t = 0; t < gs_; ++t) for (int k = 0; k < g; ++k) order_.push_back(k * gs_ + t);
    }
    int slots() const { return (int) busy_.size(); }

    // the slots as /slots and /metrics show them
    std::vector<SlotView> view() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<SlotView> v;
        for (int b = 0; b < slots(); ++b)
            v.push_back({ b, (bool) busy_[(size_t) b], live_[(size_t) b].state, live_[(size_t) b].prompt, live_[(size_t) b].generated,
                          (long long) held_[(size_t) b].size() });
        return v;
    }
    int busy_count() { std::lock_guard<std::mutex> lk(mu_); return (int) std::count(busy_.begin(), busy_.end(), true); }
    int waiting() { std::lock_guard<std::mutex> lk(mu_); return waiting_; }

    // One request's generation: tokens through hooks.token, the engine legs it took back.  Throws EngineDied /
    // EngineSilent / EngineNonFinite / EngineErr as impulse-fx's single-turn pass does.
    Result generate(const Request & q, const Hooks & H) {
        Result R;
        Run run{ *this, q, H, R };
        run.go();
        return R;
    }

  private:
    using Clock = std::chrono::steady_clock;
    struct Live { const char * state = "idle"; long long prompt = 0, generated = 0; };

    Wire & W;
    Config C;
    std::function<void(const std::string &)> log_;
    std::mutex mu_;
    std::condition_variable cv_;
    // the control lines: FIFO tickets
    bool ctl_held_ = false;
    uint64_t next_ticket_ = 0, serving_ = 0;
    std::set<uint64_t> abandoned_;
    int waiting_ = 0;              // requests holding a ticket not served yet
    int long_held_ = 0;            // long requests waiting for, or on, the solo path
    std::deque<uint64_t> slot_wait_;   // requests waiting for a free slot, in arrival order
    uint64_t next_slot_ticket_ = 0;
    // the slots
    std::vector<bool> busy_;
    std::vector<std::vector<int32_t>> held_;   // what each slot's sessions hold (prompt + every token a window fed)
    std::vector<uint64_t> used_;
    std::vector<Clock::time_point> parked_;
    std::vector<Live> live_;
    uint64_t clock_ = 0;
    int groups_ = 1, gs_ = 0;
    std::vector<int> order_;       // slots in the order that spreads requests over the pipeline's groups

    static constexpr int SOLO_AGAIN_MAX = 2, SOLO_AGAIN_MIN_LEFT = 32, BGEN_REFUSALS_MAX = 6;
    static constexpr double SLOT_PARK_S = 2.0;
    static constexpr long long SLOT_PREFIX_MIN = 512;

    void say(const std::string & s) { if (log_) log_(s); }
    static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

    void skip_abandoned() { while (abandoned_.count(serving_)) abandoned_.erase(serving_++); }

    // waits for the control lines in arrival order; false when the client left meanwhile
    bool take_control(const Hooks & H) {
        std::unique_lock<std::mutex> lk(mu_);
        const uint64_t ticket = next_ticket_++;
        ++waiting_;
        auto beat = Clock::now();
        for (;;) {
            if (!ctl_held_ && serving_ == ticket) break;
            cv_.wait_for(lk, std::chrono::milliseconds(250));
            if (!ctl_held_ && serving_ == ticket) break;
            lk.unlock();
            const bool gone = H.cancelled && H.cancelled();
            const bool dead = W.ended();
            bool beat_now = false;
            if (since(beat) >= 10.0) { beat = Clock::now(); beat_now = true; }
            if (beat_now && !gone && H.ping) H.ping();
            lk.lock();
            if (gone || dead) {
                if (serving_ == ticket) ++serving_; else abandoned_.insert(ticket);
                skip_abandoned();
                --waiting_;
                cv_.notify_all();
                if (dead) throw EngineDied("the engine stopped while this request waited; it was not sent");
                return false;
            }
        }
        ctl_held_ = true;
        ++serving_;
        skip_abandoned();
        --waiting_;
        lk.unlock();
        W.ctl_open();
        return true;
    }
    void release_control() {
        W.ctl_close();
        std::lock_guard<std::mutex> lk(mu_);
        ctl_held_ = false;
        cv_.notify_all();
    }

    // a free slot for `prompt` (the caller holds mu_), -1: every slot busy
    int pick_slot(const std::vector<int32_t> & prompt) {
        const auto now = Clock::now();
        std::vector<int> fr;
        for (int b : order_) if (!busy_[(size_t) b] && parked_[(size_t) b] <= now) fr.push_back(b);
        if (fr.empty()) return -1;
        auto rank = [&](int b) { return (int) (std::find(order_.begin(), order_.end(), b) - order_.begin()); };
        if (groups_ > 1) {
            // pipelined groups keep no conversation: fill a hole below a busy slot of the same group first (no extra
            // row), then the spreading order
            auto fills_hole = [&](int b) {
                const int g0 = b - b % gs_;
                for (int x = b + 1; x < g0 + gs_; ++x) if (busy_[(size_t) x]) return true;
                return false;
            };
            return *std::min_element(fr.begin(), fr.end(), [&](int a, int b) {
                return std::make_pair(!fills_hole(a), rank(a)) < std::make_pair(!fills_hole(b), rank(b));
            });
        }
        auto held_prefix = [&](int b) -> long long {
            const auto & h = held_[(size_t) b];
            return !h.empty() && h.size() < prompt.size() && std::equal(h.begin(), h.end(), prompt.begin()) ? (long long) h.size() : 0;
        };
        int best = fr[0];
        for (int b : fr) if (held_prefix(b) > held_prefix(best)) best = b;
        if (held_prefix(best) >= 1) return best;
        return *std::min_element(fr.begin(), fr.end(), [&](int a, int b) {
            return std::make_pair(!held_[(size_t) a].empty(), used_[(size_t) a]) < std::make_pair(!held_[(size_t) b].empty(), used_[(size_t) b]);
        });
    }
    void free_slot(int b, std::vector<int32_t> held) {
        std::lock_guard<std::mutex> lk(mu_);
        busy_[(size_t) b] = false;
        held_[(size_t) b] = std::move(held);
        used_[(size_t) b] = ++clock_;
        live_[(size_t) b] = Live{};
        cv_.notify_all();
    }

    // ------------------------------------------------------------------------------------------------ one request
    struct Run {
        Scheduler & S;
        const Request & q;
        const Hooks & H;
        Result & R;
        std::vector<int32_t> out;          // every token the engine produced for this request
        bool holding = false;
        int slot = -1;                     // claimed (busy) and not given back
        bool consumer_stopped = false;
        Clock::time_point beat = Clock::now();

        ~Run() {
            if (slot >= 0) S.free_slot(slot, {});
            if (holding) S.release_control();
        }
        bool gone() { return H.cancelled && H.cancelled(); }
        void maybe_ping() { if (since(beat) >= 10.0) { beat = Clock::now(); if (H.ping) H.ping(); } }
        bool deliver(int32_t t) {
            ++R.generated;
            beat = Clock::now();
            if (consumer_stopped) return false;
            if (H.token && !H.token(t)) consumer_stopped = true;
            return !consumer_stopped;
        }
        bool ends_eos() const { return !out.empty() && S.C.eos.count(out.back()); }

        // a free slot, claimed (busy).  With none free it queues behind the requests already waiting for one (fair: in
        // the order they came) and gives the control lines back meanwhile (a slot may need them to finish).  False:
        // the client left.
        bool wait_slot(const std::vector<int32_t> & prompt) {
            uint64_t mine = 0;
            bool queued = false;
            struct Unqueue { Scheduler & S; bool & q; uint64_t & m; ~Unqueue() {
                if (!q) return;
                std::lock_guard<std::mutex> lk(S.mu_);
                S.slot_wait_.erase(std::remove(S.slot_wait_.begin(), S.slot_wait_.end(), m), S.slot_wait_.end());
                S.cv_.notify_all();
            } } unq{ S, queued, mine };
            for (;;) {
                {
                    std::lock_guard<std::mutex> lk(S.mu_);
                    if ((!queued && S.slot_wait_.empty()) || (queued && S.slot_wait_.front() == mine)) {
                        slot = S.pick_slot(prompt);
                        if (slot >= 0) {
                            S.busy_[(size_t) slot] = true;
                            if (queued) { S.slot_wait_.pop_front(); queued = false; S.cv_.notify_all(); }
                            return true;
                        }
                    }
                    if (!queued) { mine = S.next_slot_ticket_++; S.slot_wait_.push_back(mine); queued = true; }
                }
                if (holding) { S.release_control(); holding = false; }
                {
                    std::unique_lock<std::mutex> lk(S.mu_);
                    S.cv_.wait_for(lk, std::chrono::milliseconds(500));
                }
                if (gone()) return false;
                maybe_ping();
                if (S.W.ended()) throw EngineDied("the engine stopped unexpectedly");
            }
        }

        static std::string join(const std::vector<int32_t> & v) {
            std::string s;
            s.reserve(v.size() * 7);
            for (size_t i = 0; i < v.size(); ++i) { if (i) s += ','; s += std::to_string(v[i]); }
            return s;
        }
        static std::vector<std::string> fields(const std::string & l) {
            std::istringstream is(l);
            std::vector<std::string> f;
            for (std::string w; is >> w;) f.push_back(w);
            return f;
        }

        // the control lines of a pass (solo) or an admission (to its BADM).  Returns the ERR text ("" when none).
        struct Ctl { std::string done, err; int badm = -1; };
        Ctl read_ctl(bool solo, size_t n_ids, const std::function<bool()> & stop_when) {
            Ctl c;
            bool stop_sent = false;
            const double silence = S.C.silence_s;
            double allow = silence > 0 ? silence + std::min<double>((double) n_ids, PP_CHUNK_MAX_S) / PP_FLOOR_TOK_S_S : 0.0;
            auto heard = Clock::now();
            long long read_to = 0;
            auto stop = [&]() { if (solo && !stop_sent && S.W.can_stop()) { S.W.send("STOP"); stop_sent = true; } };
            for (;;) {
                if (allow > 0 && since(heard) >= allow)
                    throw EngineSilent("the engine said nothing for " + std::to_string((int) since(heard)) + " s during the request; the server ended the engine");
                std::string l;
                if (!S.W.ctl_next(l, 0.5)) {
                    if (S.W.ended()) throw EngineDied("the engine stopped unexpectedly");
                    if (gone()) stop();
                    maybe_ping();
                    continue;
                }
                heard = Clock::now();
                if (l.rfind("T ", 0) == 0) {
                    allow = silence;
                    const int32_t t = (int32_t) std::atol(l.c_str() + 2);
                    out.push_back(t);
                    deliver(t);
                    if (gone() || consumer_stopped || (stop_when && stop_when())) stop();
                } else if (l.rfind("PP ", 0) == 0) {
                    std::istringstream is(l.substr(3));
                    long long a = -1, b = -1; double ms = 0, rate = 0;
                    is >> a >> b;
                    if (is >> ms) is >> rate;
                    if (a >= 0 && b >= 0) {
                        const long long chunk = a - read_to;
                        read_to = a;
                        if (silence > 0 && rate > 0 && chunk > 0) allow = std::max(silence, PP_SLACK_S * (double) chunk / rate);
                    }
                    if (gone()) stop();
                    beat = Clock::now();
                    if (H.ping) H.ping();
                } else if (l.rfind("RESUME ", 0) == 0) {
                    read_to = std::atoll(l.c_str() + 7);
                } else if (l.rfind("DONE", 0) == 0) {
                    c.done = l;
                    if (solo) return c;
                } else if (l.rfind("BADM ", 0) == 0) {
                    const auto f = fields(l);
                    c.badm = f.size() >= 3 && f[2] == "1" ? 1 : 0;
                    return c;
                } else if (l.rfind("ERR", 0) == 0) {
                    c.err = l.size() > 4 ? l.substr(4) : std::string();
                    return c;   // the engine refused it and is idle: no DONE follows
                } else if (l.rfind("INFO ", 0) != 0 && l.rfind("YIELDED ", 0) != 0) {
                    S.say("impulse fx engine: " + l);
                }
            }
        }
        static std::string finish_of(const std::string & done) {
            const auto f = fields(done);
            return f.size() > 5 ? f[5] : std::string();
        }

        // a slot's BT lines to its BDONE; returns the BDONE line.  going_solo: BSTOPped to go back to the solo path
        std::string read_slot(int b, bool may_go_solo, int solo_again, bool & going_solo) {
            bool bstop = false;
            const double silence = S.C.silence_s;
            auto heard = Clock::now();
            auto bstop_now = [&]() { if (!bstop) { S.W.send("BSTOP " + std::to_string(b)); bstop = true; } };
            if (consumer_stopped || gone()) bstop_now();   // it stopped at the admission's own token
            for (;;) {
                if (silence > 0 && since(heard) >= silence)
                    throw EngineSilent("the engine said nothing for " + std::to_string((int) since(heard)) + " s in batch slot " + std::to_string(b) + "; the server ended the engine");
                std::string l;
                if (!S.W.slot_next(b, l, 0.5)) {
                    if (S.W.ended()) throw EngineDied("the engine stopped unexpectedly");
                    if (gone()) bstop_now();
                    maybe_ping();
                    continue;
                }
                heard = Clock::now();
                if (l.rfind("BT ", 0) == 0) {
                    const auto f = fields(l);
                    if (f.size() < 3) continue;
                    const int32_t t = (int32_t) std::atol(f[2].c_str());
                    out.push_back(t);
                    { std::lock_guard<std::mutex> lk(S.mu_); S.live_[(size_t) b].generated = (long long) out.size(); }
                    if (bstop) continue;
                    if (gone()) { bstop_now(); continue; }
                    if (!deliver(t)) { bstop_now(); continue; }
                    if (may_go_solo && !going_solo && S.may_go_solo(q.max_new - (long long) out.size(), solo_again, q.embeddings)) {
                        bstop_now();
                        going_solo = true;
                    }
                } else if (l.rfind("BDONE ", 0) == 0) {
                    return l;
                }
            }
        }

        void go() {
            if (S.W.ended()) throw EngineDied("the engine is unavailable; this request was not sent");
            std::vector<int32_t> prompt = q.ids;
            long long left = q.max_new;
            if (!S.take_control(H)) return;
            holding = true;
            bool long_req = S.C.batch_ctx > 0 && (long long) prompt.size() + left + 8 > S.C.batch_ctx;
            bool long_counted = false;
            struct LongGuard { Scheduler & S; bool & on; ~LongGuard() { if (on) { std::lock_guard<std::mutex> lk(S.mu_); S.long_held_ = std::max(0, S.long_held_ - 1); S.cv_.notify_all(); } } } lg{ S, long_counted };
            int solo_again = 0, refused = 0, badm = -1;
            for (;;) {
                if (!holding) {
                    if (!S.take_control(H)) return;
                    holding = true;
                }
                if (long_req && !long_counted) {
                    long_counted = true;
                    std::lock_guard<std::mutex> lk(S.mu_);
                    ++S.long_held_;
                }
                while (long_req) {   // it cannot move to a slot: it waits for the slots to finish, holding the lines
                    {
                        std::unique_lock<std::mutex> lk(S.mu_);
                        if (std::find(S.busy_.begin(), S.busy_.end(), true) == S.busy_.end()) break;
                        S.cv_.wait_for(lk, std::chrono::milliseconds(500));
                    }
                    if (gone()) return;
                    maybe_ping();
                }
                bool alone;
                {
                    std::lock_guard<std::mutex> lk(S.mu_);
                    alone = std::find(S.busy_.begin(), S.busy_.end(), true) == S.busy_.end() && (S.waiting_ == 0 || long_req);
                }
                if (alone && (left > 1 || long_req)) {
                    // ---- the solo path (drafts); STOPped when another request arrives, then on in a slot
                    const std::string head = q.embeddings.empty() ? "GEN " + std::to_string(left) + q.keys
                                                                  : "GENI " + std::to_string(left) + q.keys + " " + q.embeddings;
                    if (!S.W.send(head + " " + join(prompt))) throw EngineDied("the engine stopped unexpectedly");
                    Ctl c = read_ctl(true, prompt.size(), [&]() {
                        if (long_req) return false;
                        std::lock_guard<std::mutex> lk(S.mu_);
                        return S.waiting_ > 0;
                    });
                    if (!c.err.empty()) throw EngineErr(c.err);
                    R.legs.push_back(Leg{ c.done, "" });
                    const std::string fin = finish_of(c.done);
                    if (fin == "nonfinite") throw EngineNonFinite("the engine's logits became non-finite (#879)");
                    left = q.max_new - (long long) out.size();
                    if (consumer_stopped || gone() || left <= 0 || fin == "stop" || fin == "length" || ends_eos()) return;
                    prompt = q.ids;   // promoted: it continues in a batch slot with its prompt and output
                    prompt.insert(prompt.end(), out.begin(), out.end());
                }
                // ---- a slot
                for (;;) {
                    if (slot < 0 && !wait_slot(prompt)) return;
                    if (!holding) {
                        // the slot is held while this request queues for the lines again (in order)
                        if (!S.take_control(H)) return;
                        holding = true;
                    }
                    S.W.slot_clear(slot);
                    {
                        std::lock_guard<std::mutex> lk(S.mu_);
                        S.live_[(size_t) slot] = Live{ "reading", (long long) prompt.size(), 0 };
                        S.held_[(size_t) slot].clear();   // the admission overwrites what the slot held
                    }
                    const std::string head = q.embeddings.empty() ? "BGEN " + std::to_string(slot) + " " + std::to_string(left) + q.keys
                                                                  : "BGENI " + std::to_string(slot) + " " + std::to_string(left) + q.keys + " " + q.embeddings;
                    if (!S.W.send(head + " " + join(prompt))) throw EngineDied("the engine stopped unexpectedly");
                    Ctl c = read_ctl(false, prompt.size(), nullptr);
                    if (!c.err.empty()) {
                        if (c.err.rfind("BGEN-LONG:", 0) == 0 && !gone()) {
                            // longer than a slot (--batch-max-context): alone on the solo path, the lines kept
                            S.say("[impulse fx] parallel: " + c.err + "; it runs alone on the solo path");
                            S.free_slot(slot, {});
                            slot = -1;
                            long_req = true;
                            break;
                        }
                        if (c.err.rfind("BGEN:", 0) == 0 && left >= 1 && !gone()) {
                            ++refused;
                            S.say("[impulse fx] parallel: the engine refused slot " + std::to_string(slot) + " (" + c.err + "); trying another slot (" +
                                  std::to_string(refused) + "/" + std::to_string(BGEN_REFUSALS_MAX) + ")");
                            {
                                std::lock_guard<std::mutex> lk(S.mu_);
                                S.busy_[(size_t) slot] = false;
                                S.parked_[(size_t) slot] = Clock::now() + std::chrono::milliseconds((int) (SLOT_PARK_S * 1000));
                                S.live_[(size_t) slot] = Live{};
                            }
                            slot = -1;
                            if (refused > BGEN_REFUSALS_MAX) throw EngineErr("the engine's batch slots are not free right now; try again in a moment");
                            continue;
                        }
                        throw EngineErr(c.err);
                    }
                    R.legs.push_back(Leg{ c.done, "" });
                    badm = c.badm;
                    break;
                }
                if (long_req && slot < 0) continue;   // BGEN-LONG: back to the top (solo, alone)
                S.release_control();
                holding = false;
                const std::string fin = finish_of(R.legs.back().done);
                if (badm != 1) {   // BADM 0: the admission ended the request (an EOS, max_new 1, a STOP)
                    S.free_slot(slot, {});
                    slot = -1;
                    if (fin == "nonfinite") throw EngineNonFinite("the engine's logits became non-finite (#879)");
                    return;
                }
                const size_t gen0 = out.empty() ? 0 : out.size() - 1;   // the admission's own token: the slot feeds it first
                { std::lock_guard<std::mutex> lk(S.mu_); S.live_[(size_t) slot].state = "decoding"; S.live_[(size_t) slot].generated = (long long) out.size(); }
                bool going_solo = false;
                const std::string bd = read_slot(slot, !consumer_stopped && !gone(), solo_again, going_solo);
                R.legs.back().bdone = bd;
                const auto f = fields(bd);
                const std::string bfin = f.size() > 3 ? f[3] : std::string();
                std::vector<int32_t> held;
                if (bfin != "nonfinite" && out.size() > gen0) {
                    held = prompt;
                    held.insert(held.end(), out.begin() + (long) gen0, out.end() - 1);   // all but the last token were fed
                }
                S.free_slot(slot, std::move(held));
                slot = -1;
                if (bfin == "nonfinite") throw EngineNonFinite("the engine's logits became non-finite (#879)");
                if (going_solo && bfin == "cancel" && !gone() && !consumer_stopped && (long long) out.size() < q.max_new && !ends_eos()) {
                    prompt = q.ids;   // the solo path continues it from the slot's sessions (the engine copies them back)
                    prompt.insert(prompt.end(), out.begin(), out.end());
                    left = q.max_new - (long long) out.size();
                    ++solo_again;
                    continue;
                }
                return;
            }
        }
    };

    bool may_go_solo(long long left, int times, const std::string & embeddings) {
        if (!embeddings.empty() || times >= SOLO_AGAIN_MAX || left < SOLO_AGAIN_MIN_LEFT || !C.slot_cache) return false;
        if (const char * e = std::getenv("STRATA_PARALLEL_SOLO")) if (std::string(e) == "0") return false;
        std::lock_guard<std::mutex> lk(mu_);
        return std::count(busy_.begin(), busy_.end(), true) == 1 && waiting_ == 0 && long_held_ == 0;
    }
};

// The startup line on what the batch slots cost: one slot's sessions (INFO batch_slot_mib, every stage of a layer split
// counted) in expert-cache slots (INFO expert_cache_mib / expert_slots: the cache as it is with the slots carved).
// Every slot's VRAM is VRAM the expert cache does not get (flashnext/docs/BATCHING.md "What a slot costs").
inline std::string sizing_note(int slots, double slot_mib, double expert_slots, double cache_mib, long long batch_ctx) {
    char buf[512];
    const std::string cells = batch_ctx > 0 ? std::to_string(batch_ctx) + " cells each" : std::string("the whole context each");
    if (slot_mib <= 0 || expert_slots <= 0 || cache_mib <= 0) {
        std::snprintf(buf, sizeof buf, "impulse fx: %d batch slots (%s); the engine does not say what a slot costs (an engine "
                                       "before batch_slot_mib)\n", slots, cells.c_str());
        return buf;
    }
    const double per_expert = cache_mib / expert_slots;
    const double each = slot_mib / per_expert;
    std::snprintf(buf, sizeof buf, "impulse fx: %d batch slots (%s): one slot's sessions take %.0f MiB = about %.0f expert-cache "
                                   "slots (an expert slot %.2f MiB); all %d: %.0f expert slots, %.1f %% of the cache beside them "
                                   "(%.0f slots, %.0f MiB)\n",
                  slots, cells.c_str(), slot_mib, each, per_expert, slots, each * slots,
                  100.0 * each * slots / (expert_slots + each * slots), expert_slots, cache_mib);
    return buf;
}

}  // namespace fxs
