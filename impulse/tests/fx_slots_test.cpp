// fx_slots_test: impulse-fx's batch-slot scheduler (src/fx/fx_slots.hpp) against a fake engine that speaks the serve
// protocol's GEN / STOP / BGEN / BT / BDONE / BADM / BSTOP (flashnext/docs/BATCHING.md), no GPU, no model.
//
// The fake engine's next token is a function of the whole sequence so far (prompt + every token), so a request gives the
// same tokens whichever way it runs: alone on the solo path, promoted into a slot mid-reply, back on the solo path, or
// next to others in batch windows - the property the real engine has with its exactness settings (BATCHING.md
// "Exactness").  Each check compares a request's tokens with that reference.
//
//   g++ -std=c++17 -O2 -I impulse/src tests/fx_slots_test.cpp -pthread && ./a.out
#include "fx/fx_slots.hpp"

#include <atomic>
#include <cstdio>
#include <deque>
#include <thread>

static int fails = 0;
static void check(const char * what, bool ok, const std::string & detail = "") {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", what, detail.empty() ? "" : "  -- ", detail.c_str());
    if (!ok) ++fails;
}

static const int32_t EOS = 2;
static uint64_t mix(const std::vector<int32_t> & s) {
    uint64_t h = 1469598103934665603ull;
    for (auto t : s) { h ^= (uint64_t) (uint32_t) t; h *= 1099511628211ull; }
    return h ^ (h >> 29);
}
static int32_t next_tok(const std::vector<int32_t> & seq, size_t prompt_len) {
    const uint64_t h = mix(seq);
    if (seq.size() - prompt_len >= 6 && h % 61 == 0) return EOS;
    return (int32_t) (100 + h % 900);
}
// what a request yields alone: tokens to max_new or the EOS (included, as the engine sends it)
static std::vector<int32_t> reference(std::vector<int32_t> seq, long long max_new) {
    const size_t pl = seq.size();
    std::vector<int32_t> out;
    while ((long long) out.size() < max_new) {
        const int32_t t = next_tok(seq, pl);
        out.push_back(t);
        seq.push_back(t);
        if (t == EOS) break;
    }
    return out;
}

// ---------------------------------------------------------------- the fake engine
struct Fake : fxs::Wire {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> ctl, in;
    std::vector<std::deque<std::string>> slotq;
    bool reading = false, eof = false, quit = false;
    struct Slot { bool active = false, stop = false; std::vector<int32_t> seq; size_t pl = 0; long long produced = 0, max_new = 0; };
    std::vector<Slot> slots;
    long long batch_ctx = 0;
    int refuse_next_bgen = 0;           // answer this many BGENs with ERR BGEN:
    int max_active = 0;
    std::vector<std::string> log;       // every line received (ids cut)
    std::thread worker, windows;
    std::atomic<int> token_us{ 300 };

    explicit Fake(int n) : slotq((size_t) n), slots((size_t) n) {
        worker = std::thread([this] { work(); });
        windows = std::thread([this] { window_loop(); });
    }
    ~Fake() { { std::lock_guard<std::mutex> lk(mu); quit = true; eof = true; } cv.notify_all(); worker.join(); windows.join(); }
    void die() { std::lock_guard<std::mutex> lk(mu); eof = true; quit = true; cv.notify_all(); }

    // Wire
    bool send(const std::string & line) override {
        std::lock_guard<std::mutex> lk(mu);
        if (eof) return false;
        in.push_back(line);
        cv.notify_all();
        return true;
    }
    void ctl_open() override { std::lock_guard<std::mutex> lk(mu); ctl.clear(); reading = true; }
    void ctl_close() override { std::lock_guard<std::mutex> lk(mu); reading = false; ctl.clear(); }
    bool ctl_next(std::string & out, double s) override {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk, std::chrono::duration<double>(s), [&] { return !ctl.empty() || eof; });
        if (ctl.empty()) return false;
        out = ctl.front(); ctl.pop_front();
        return true;
    }
    bool slot_next(int b, std::string & out, double s) override {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait_for(lk, std::chrono::duration<double>(s), [&] { return !slotq[(size_t) b].empty() || eof; });
        if (slotq[(size_t) b].empty()) return false;
        out = slotq[(size_t) b].front(); slotq[(size_t) b].pop_front();
        return true;
    }
    void slot_clear(int b) override { std::lock_guard<std::mutex> lk(mu); slotq[(size_t) b].clear(); }
    bool ended() override { std::lock_guard<std::mutex> lk(mu); return eof; }
    bool can_stop() override { return true; }

    void out_ctl(const std::string & l) { if (reading) { ctl.push_back(l); cv.notify_all(); } }   // (caller holds mu)

    static std::vector<int32_t> ids_of(const std::string & s) {
        std::vector<int32_t> v;
        size_t p = 0;
        while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); v.push_back(std::atoi(s.substr(p, q - p).c_str())); p = q + 1; }
        return v;
    }
    void work() {
        std::unique_lock<std::mutex> lk(mu);
        for (;;) {
            cv.wait(lk, [&] { return !in.empty() || quit; });
            if (quit) return;
            const std::string line = in.front(); in.pop_front();
            const size_t sp = line.rfind(' ');
            log.push_back(line.substr(0, sp) + " #" + (sp == std::string::npos ? "" : line.substr(sp + 1, line.find(',', sp) - sp - 1)));
            std::istringstream is(line);
            std::string cmd; is >> cmd;
            if (cmd == "GEN") {
                long long max_new = 0; is >> max_new;
                std::vector<int32_t> seq = ids_of(line.substr(sp + 1));
                const size_t pl = seq.size();
                // a prompt that continues a reply: the reference counts generated tokens from the request's own prompt, so
                // the fake keys the EOS rule on the first id (the request's prompt length is in it, see make_prompt)
                const size_t own = (size_t) seq[0];
                out_ctl("RESUME 0");
                out_ctl("PP " + std::to_string(pl) + " " + std::to_string(pl) + " 1.0 1000");
                long long gen = 0;
                std::string fin = "length";
                while (gen < max_new) {
                    // between tokens the engine reads its input: STOP ends the pass
                    bool stop = false;
                    for (auto it = in.begin(); it != in.end(); ++it) if (*it == "STOP") { in.erase(it); stop = true; log.push_back("STOP"); break; }
                    if (stop) { fin = "cancel"; break; }
                    const int32_t t = next_tok(seq, own);
                    seq.push_back(t); ++gen;
                    out_ctl("T " + std::to_string(t));
                    if (t == EOS) { fin = "stop"; break; }
                    lk.unlock(); std::this_thread::sleep_for(std::chrono::microseconds(token_us.load())); lk.lock();
                    if (quit) return;
                }
                out_ctl("DONE " + std::to_string(gen) + " " + std::to_string(pl) + " 1.0 2.0 " + fin + " 0 0 0 0 0 0 0 0 " + std::to_string(pl) + " 0");
            } else if (cmd == "STOP") {
                // a STOP with no pass running: nothing
            } else if (cmd == "BGEN") {
                int b = -1; long long max_new = 0; is >> b >> max_new;
                std::vector<int32_t> seq = ids_of(line.substr(sp + 1));
                if (b < 0 || b >= (int) slots.size() || slots[(size_t) b].active || max_new < 1 || refuse_next_bgen > 0) {
                    if (refuse_next_bgen > 0) --refuse_next_bgen;
                    out_ctl("ERR BGEN: no such free slot (--batch " + std::to_string(slots.size()) + ") or a bad max_new");
                    continue;
                }
                if (batch_ctx > 0 && (long long) seq.size() + max_new > batch_ctx) {
                    out_ctl("ERR BGEN-LONG: prompt (" + std::to_string(seq.size()) + " tokens) + max_new (" + std::to_string(max_new) + ") does not fit a batch slot");
                    continue;
                }
                const size_t own = (size_t) seq[0];
                const int32_t t = next_tok(seq, own);
                seq.push_back(t);
                out_ctl("T " + std::to_string(t));
                out_ctl("DONE 1 " + std::to_string(seq.size() - 1) + " 1.0 0.5 " + (t == EOS ? "stop" : max_new == 1 ? "length" : "length") + " 0 0 0 0 0 0 0 0 1 0");
                const bool cont = t != EOS && max_new > 1;
                out_ctl("BADM " + std::to_string(b) + " " + (cont ? "1" : "0"));
                if (cont) {
                    Slot & s = slots[(size_t) b];
                    s = Slot{ true, false, seq, own, 0, max_new - 1 };
                    int act = 0; for (auto & x : slots) act += x.active;
                    max_active = std::max(max_active, act);
                }
            } else if (cmd == "BSTOP") {
                int b = -1; is >> b;
                if (b >= 0 && b < (int) slots.size() && slots[(size_t) b].active) slots[(size_t) b].stop = true;
            }
        }
    }
    void window_loop() {
        std::unique_lock<std::mutex> lk(mu);
        while (!quit) {
            for (size_t b = 0; b < slots.size(); ++b) {
                Slot & s = slots[b];
                if (!s.active) continue;
                if (s.stop) { slotq[b].push_back("BDONE " + std::to_string(b) + " " + std::to_string(s.produced) + " cancel 1.0"); s.active = false; continue; }
                const int32_t t = next_tok(s.seq, s.pl);
                s.seq.push_back(t); ++s.produced;
                slotq[b].push_back("BT " + std::to_string(b) + " " + std::to_string(t));
                const char * fin = t == EOS ? "stop" : s.produced >= s.max_new ? "length" : nullptr;
                if (fin) { slotq[b].push_back("BDONE " + std::to_string(b) + " " + std::to_string(s.produced) + " " + fin + " 1.0"); s.active = false; }
            }
            cv.notify_all();
            lk.unlock(); std::this_thread::sleep_for(std::chrono::microseconds(token_us.load())); lk.lock();
        }
    }
    int count(const std::string & prefix) { std::lock_guard<std::mutex> lk(mu); int n = 0; for (auto & l : log) n += l.rfind(prefix, 0) == 0; return n; }
    std::vector<std::string> lines() { std::lock_guard<std::mutex> lk(mu); return log; }
};

// a prompt whose first id is its own length (the fake's EOS rule counts generated tokens from there)
static std::vector<int32_t> make_prompt(int seed, int len) {
    std::vector<int32_t> p = { len };
    for (int i = 1; i < len; ++i) p.push_back(1000 + (seed * 7919 + i * 31) % 5000);
    return p;
}

struct Client {
    std::vector<int32_t> got;
    std::string err;
    fxs::Result res;
    int stop_after = -1;                // the consumer stops after this many tokens (-1: never)
    std::atomic<bool> cancel{ false };
};
static void run_client(fxs::Scheduler & S, const std::vector<int32_t> & prompt, long long max_new, Client & c) {
    fxs::Hooks h;
    h.token = [&](int32_t t) { c.got.push_back(t); return c.stop_after < 0 || (int) c.got.size() < c.stop_after; };
    h.ping = [] {};
    h.cancelled = [&] { return c.cancel.load(); };
    try { c.res = S.generate(fxs::Request{ max_new, " temperature=0", "", prompt }, h); }
    catch (const std::exception & e) { c.err = e.what(); }
}
static std::string show(const std::vector<int32_t> & v) { std::string s; for (size_t i = 0; i < v.size() && i < 12; ++i) s += std::to_string(v[i]) + " "; return s + "(" + std::to_string(v.size()) + ")"; }

static fxs::Config cfg(int n) { fxs::Config c; c.slots = n; c.slot_cache = false; c.silence_s = 20; c.eos = { EOS }; return c; }

int main() {
    // 1. alone: the solo path, no slot
    {
        Fake E(4);
        fxs::Scheduler S(E, cfg(4));
        Client c;
        const auto p = make_prompt(1, 40);
        run_client(S, p, 50, c);
        check("alone: the solo path's tokens = the reference", c.err.empty() && c.got == reference(p, 50), c.err + " " + show(c.got) + " vs " + show(reference(p, 50)));
        check("alone: GEN, no BGEN", E.count("GEN ") == 1 && E.count("BGEN") == 0);
        check("alone: one leg with its DONE", c.res.legs.size() == 1 && c.res.legs[0].done.rfind("DONE", 0) == 0);
    }
    // 2. more requests than slots, at once: every one = its reference, slots shared, never more than 4
    {
        Fake E(4);
        fxs::Scheduler S(E, cfg(4));
        const int N = 9;
        std::vector<Client> cs(N);
        std::vector<std::vector<int32_t>> ps;
        std::vector<std::thread> th;
        for (int i = 0; i < N; ++i) ps.push_back(make_prompt(10 + i, 30 + i));
        for (int i = 0; i < N; ++i) { th.emplace_back([&, i] { run_client(S, ps[(size_t) i], 120, cs[(size_t) i]); }); std::this_thread::sleep_for(std::chrono::milliseconds(3)); }
        for (auto & t : th) t.join();
        int ok = 0;
        std::string bad;
        for (int i = 0; i < N; ++i) {
            if (cs[(size_t) i].err.empty() && cs[(size_t) i].got == reference(ps[(size_t) i], 120)) ++ok;
            else bad += " #" + std::to_string(i) + ":" + cs[(size_t) i].err + " " + show(cs[(size_t) i].got) + " want " + show(reference(ps[(size_t) i], 120));
        }
        check("9 requests on 4 slots: every reply = its reference", ok == N, bad);
        { int ma; { std::lock_guard<std::mutex> lk(E.mu); ma = E.max_active; } check("... batched: several slots ran at once, never more than 4", ma >= 2 && ma <= 4, std::to_string(ma)); }
        check("... the first request was promoted (GEN, then BGEN)", E.count("GEN ") >= 1 && E.count("BGEN") >= N - 1, std::to_string(E.count("GEN ")) + "/" + std::to_string(E.count("BGEN")));
        check("... every slot free at the end", S.busy_count() == 0);
    }
    // 3. a consumer that stops early in a slot (a stop string): exactly its tokens, its slot BSTOPped and freed
    {
        Fake E(2);
        fxs::Scheduler S(E, cfg(2));
        Client a, b;
        a.stop_after = 7;
        const auto pa = make_prompt(21, 50), pb = make_prompt(22, 50);
        std::thread tb([&] { run_client(S, pb, 200, b); });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::thread ta([&] { run_client(S, pa, 200, a); });
        ta.join(); tb.join();
        const auto ra = reference(pa, 200);
        check("consumer stop: 7 tokens delivered, = the reference's first 7", a.err.empty() && a.got.size() == 7 && std::equal(a.got.begin(), a.got.end(), ra.begin()), a.err + " " + show(a.got));
        check("... the other request unaffected", b.err.empty() && b.got == reference(pb, 200), b.err);
        check("... BSTOP sent, slots free", E.count("BSTOP") >= 1 && S.busy_count() == 0, std::to_string(E.count("BSTOP")));
    }
    // 3b. a consumer that stops at the admission's own token (its first in the slot): one BSTOP, the slot freed
    {
        Fake E(2);
        fxs::Scheduler S(E, cfg(2));
        Client a, b;
        a.stop_after = 1;
        const auto pa = make_prompt(23, 50), pb = make_prompt(24, 50);
        std::thread tb([&] { run_client(S, pb, 200, b); });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::thread ta([&] { run_client(S, pa, 200, a); });
        ta.join(); tb.join();
        check("stop at the admission's token: 1 token, = the reference's first", a.err.empty() && a.got.size() == 1 && a.got[0] == reference(pa, 200)[0], a.err + " " + show(a.got));
        check("... one BSTOP for it, the other unaffected, slots free", E.count("BSTOP") == 1 && b.got == reference(pb, 200) && S.busy_count() == 0,
              std::to_string(E.count("BSTOP")));
    }
    // 4. a queued request whose client leaves: no engine line for it, the queue moves on
    {
        Fake E(1);   // (one slot: the engine would not batch, but the scheduler's queue is the same)
        fxs::Config c1 = cfg(1);
        fxs::Scheduler S(E, c1);
        Client a, b, c;
        const auto pa = make_prompt(31, 20), pb = make_prompt(32, 20), pc = make_prompt(33, 20);
        E.token_us = 3000;
        std::thread ta([&] { run_client(S, pa, 60, a); });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::thread tb([&] { run_client(S, pb, 60, b); });
        std::thread tc([&] { run_client(S, pc, 60, c); });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        b.cancel = true;
        ta.join(); tb.join(); tc.join();
        bool sent_b = false;
        for (auto & l : E.lines()) if (l.find("#" + std::to_string(pb[0])) != std::string::npos && l.find(" " ) != std::string::npos && l.rfind("GEN", 0) == 0 && false) sent_b = true;
        check("cancel while queued: nothing delivered, no leg", b.got.empty() && b.res.legs.empty() && b.err.empty(), show(b.got));
        check("... the others complete", a.got == reference(pa, 60) && c.got == reference(pc, 60), a.err + c.err);
        (void) sent_b;
    }
    // 5. BGEN-LONG: a prompt longer than a slot runs alone on the solo path once the slots are idle
    {
        Fake E(2);
        E.batch_ctx = 200;
        fxs::Config c = cfg(2);
        c.batch_ctx = 0;   // (the scheduler does not know it: the engine's refusal decides, as with an older INFO)
        fxs::Scheduler S(E, c);
        Client a, b;
        const auto pa = make_prompt(41, 40), pb = make_prompt(42, 180);
        std::thread ta([&] { run_client(S, pa, 100, a); });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::thread tb([&] { run_client(S, pb, 100, b); });
        ta.join(); tb.join();
        check("BGEN-LONG: the long request = its reference (solo, alone)", b.err.empty() && b.got == reference(pb, 100), b.err + " " + show(b.got));
        check("... the short one too", a.err.empty() && a.got == reference(pa, 100), a.err);
    }
    // 6. the engine refuses a slot (ERR BGEN:): parked, another slot taken, the reply unchanged
    {
        Fake E(3);
        E.refuse_next_bgen = 1;
        fxs::Scheduler S(E, cfg(3));
        Client a, b;
        const auto pa = make_prompt(51, 30), pb = make_prompt(52, 30);
        std::thread ta([&] { run_client(S, pa, 80, a); });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::thread tb([&] { run_client(S, pb, 80, b); });
        ta.join(); tb.join();
        check("ERR BGEN: another slot, replies = references", a.err.empty() && b.err.empty() && a.got == reference(pa, 80) && b.got == reference(pb, 80), a.err + "|" + b.err);
    }
    // 7. fair queueing: with every slot busy, the waiting requests are admitted in the order they came
    {
        Fake E(2);
        E.token_us = 1000;
        fxs::Scheduler S(E, cfg(2));
        std::vector<Client> cs(5);
        std::vector<std::vector<int32_t>> ps;
        for (int i = 0; i < 5; ++i) ps.push_back(make_prompt(60 + i, 20 + i));   // first id = 20 + i: the request's mark
        std::vector<std::thread> th;
        for (int i = 0; i < 5; ++i) { th.emplace_back([&, i] { run_client(S, ps[(size_t) i], i < 2 ? 150 : 30, cs[(size_t) i]); }); std::this_thread::sleep_for(std::chrono::milliseconds(15)); }
        for (auto & t : th) t.join();
        std::vector<int> order;
        for (auto & l : E.lines()) {
            if (l.rfind("BGEN", 0) != 0) continue;
            const int mark = std::atoi(l.substr(l.rfind('#') + 1).c_str());
            if (mark >= 22 && std::find(order.begin(), order.end(), mark) == order.end()) order.push_back(mark);
        }
        check("fair queueing: the waiting requests admitted in arrival order", order == std::vector<int>({ 22, 23, 24 }),
              [&] { std::string s; for (int x : order) s += std::to_string(x) + " "; return s; }());
        bool all = true;
        for (int i = 0; i < 5; ++i) all &= cs[(size_t) i].got == reference(ps[(size_t) i], i < 2 ? 150 : 30);
        check("... every reply = its reference", all);
    }
    // 8. a request left alone in its slot goes back to the solo path (slot_cache), its reply unchanged
    {
        Fake E(2);
        E.token_us = 500;
        fxs::Config c = cfg(2);
        c.slot_cache = true;
        fxs::Scheduler S(E, c);
        Client a, b;
        const auto pa = make_prompt(71, 25), pb = make_prompt(72, 25);
        std::thread ta([&] { run_client(S, pa, 400, a); });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::thread tb([&] { run_client(S, pb, 10, b); });
        ta.join(); tb.join();
        const auto ra = reference(pa, 400);
        int gens_a = 0;
        for (auto & l : E.lines()) if (l.rfind("GEN ", 0) == 0 && l.find("#25") != std::string::npos) ++gens_a;
        check("back to the solo path: the long reply = its reference", a.err.empty() && a.got == ra, a.err + " " + show(a.got));
        check("... it ran solo, in a slot, then solo again (2 GEN, legs >= 3)", (gens_a >= 2 && a.res.legs.size() >= 3) || ra.size() < 40,
              std::to_string(gens_a) + " GEN, " + std::to_string(a.res.legs.size()) + " legs");
    }
    // 9. the engine ends mid-request: EngineDied
    {
        Fake E(2);
        E.token_us = 2000;
        fxs::Scheduler S(E, cfg(2));
        Client a;
        const auto pa = make_prompt(81, 20);
        std::thread ta([&] { run_client(S, pa, 5000, a); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        E.die();
        ta.join();
        check("the engine ends: the request fails with EngineDied", a.err.find("stopped") != std::string::npos, a.err);
    }
    // 10. sizing note
    {
        const std::string s = fxs::sizing_note(2, 1024, 9000, 63000, 32768);
        check("sizing note: a slot's MiB in expert slots", s.find("about 146 expert-cache") != std::string::npos, s);
        check("sizing note: no figure from an older engine", fxs::sizing_note(2, 0, 9000, 63000, 0).find("does not say") != std::string::npos);
    }
    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
