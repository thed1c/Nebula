#!/usr/bin/env python3
"""Build a Qwen3.8 Flash-Next (qwen4exp) MTP drafter GGUF from Strata's drafter files.

Pure Python, stdlib only (no numpy, no gguf-py). Every tensor is a byte copy except the
router and shared-expert gate, which are widened BF16 -> F32 (a 16-bit shift).

Sources (Strata, `strata-data/mtp/`):
  --mtp-gguf      mtp-q2_0.gguf: Q2_0 routed experts (fused gate_up: per expert rows 0..639 gate,
                  640..1279 up, 720-byte rows; down_proj 180-byte rows).
  --dense-dir     rt/ with dense.bin + dense.txt (`name kind rows cols offset bytes`): Q8_0 projections,
                  BF16 hyper-connection/router weights, F32 norms with the Gemma +1 already applied.
  --target-shard1 the target's shard-1 GGUF: the KV block is copied from its header.

Output layout (llama.cpp qwen4exp/mtp branch, trailing block blk.48):
  arch qwen4exp, block_count 49, nextn_predict_layers 1, compress_ratios extended with a 0 (dense),
  split.* dropped, no token_embd / output (the engine borrows the target's).
  Experts Q2_0, projections Q8_0, HC weights BF16, norms F32, ffn_gate_inp(+_shexp) F32.

Usage:
  q4x_mtp_gguf.py --mtp-gguf .../mtp-q2_0.gguf --dense-dir .../rt \
      --target-shard1 .../Qwen3.8-Flash-Next-...-00001-of-0000N.gguf --out drafter.gguf
"""
import argparse
import hashlib
import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
ALIGN_DEFAULT = 32

# GGUF value types
VT_U8, VT_I8, VT_U16, VT_I16, VT_U32, VT_I32, VT_F32, VT_BOOL, VT_STR, VT_ARR, VT_U64, VT_I64, VT_F64 = range(13)
_SCALAR = {VT_U8: "<B", VT_I8: "<b", VT_U16: "<H", VT_I16: "<h", VT_U32: "<I", VT_I32: "<i",
           VT_F32: "<f", VT_BOOL: "<?", VT_U64: "<Q", VT_I64: "<q", VT_F64: "<d"}

# ggml types used here: (name, block elements, block bytes)
GT_F32, GT_Q8_0, GT_BF16, GT_Q2_0 = 0, 8, 30, 42
GGML = {GT_F32: ("F32", 1, 4), GT_Q8_0: ("Q8_0", 32, 34), GT_BF16: ("BF16", 1, 2), GT_Q2_0: ("Q2_0", 64, 18)}
DENSE_KIND = {"f32": GT_F32, "q8_0": GT_Q8_0, "bf16": GT_BF16}

N_EMBD, HC, HC_LR, N_EXPERT, N_FF_EXP = 2560, 4, 320, 512, 640
HC_DIM = HC * N_EMBD
MTP_BLK = 48


def nbytes(gtype, dims):
    _, blk, bsz = GGML[gtype]
    n = 1
    for d in dims:
        n *= d
    if dims[0] % blk:
        raise ValueError(f"row of {dims[0]} is not a multiple of the {GGML[gtype][0]} block ({blk})")
    return n // blk * bsz


def align(x, a):
    return (x + a - 1) // a * a


# ---------------------------------------------------------------- GGUF header reader

class _R:
    def __init__(self, f):
        self.f = f

    def u(self, fmt):
        s = struct.calcsize(fmt)
        b = self.f.read(s)
        if len(b) != s:
            raise EOFError("truncated GGUF header")
        return struct.unpack(fmt, b)[0]

    def s(self):
        return self.f.read(self.u("<Q")).decode("utf-8")

    def val(self, t):
        if t == VT_STR:
            return self.s()
        if t == VT_ARR:
            et = self.u("<I")
            n = self.u("<Q")
            if et in _SCALAR:  # fast path for numeric arrays
                fmt = _SCALAR[et][1]
                sz = struct.calcsize("<" + fmt)
                return (et, list(struct.unpack(f"<{n}{fmt}", self.f.read(sz * n))))
            return (et, [self.val(et) for _ in range(n)])
        return self.u(_SCALAR[t])


def read_gguf_header(path):
    """Return dict(version, kv=[(key, vtype, value, raw_bytes)], tensors=[(name, dims, gtype, rel_off)],
    alignment, data_start)."""
    with open(path, "rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        r = _R(f)
        ver = r.u("<I")
        nt = r.u("<Q")
        nkv = r.u("<Q")
        kv = []
        for _ in range(nkv):
            start = f.tell()
            k = r.s()
            t = r.u("<I")
            v = r.val(t)
            end = f.tell()
            f.seek(start)
            raw = f.read(end - start)
            kv.append((k, t, v, raw))
        tensors = []
        for _ in range(nt):
            name = r.s()
            nd = r.u("<I")
            dims = [r.u("<Q") for _ in range(nd)]
            gt = r.u("<I")
            off = r.u("<Q")
            tensors.append((name, dims, gt, off))
        hdr_end = f.tell()
    al = ALIGN_DEFAULT
    for k, t, v, _ in kv:
        if k == "general.alignment":
            al = int(v)
    return {"version": ver, "kv": kv, "tensors": tensors, "alignment": al, "data_start": align(hdr_end, al)}


# ---------------------------------------------------------------- GGUF encoding

def enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def enc_kv(key, vtype, value):
    out = enc_str(key) + struct.pack("<I", vtype)
    if vtype == VT_STR:
        return out + enc_str(value)
    if vtype == VT_ARR:
        et, items = value
        out += struct.pack("<IQ", et, len(items))
        if et == VT_STR:
            return out + b"".join(enc_str(x) for x in items)
        return out + struct.pack(f"<{len(items)}{_SCALAR[et][1]}", *items)
    return out + struct.pack(_SCALAR[vtype], value)


# ---------------------------------------------------------------- sources

def read_dense_index(dense_dir):
    idx = {}
    with open(os.path.join(dense_dir, "dense.txt")) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            name, kind, rows, cols, off, nb = p[0], p[1], int(p[2]), int(p[3]), int(p[4]), int(p[5])
            gt = DENSE_KIND[kind]
            if nbytes(gt, [cols, rows]) != nb:
                raise ValueError(f"dense.txt: {name} {kind} {rows}x{cols} should be {nbytes(gt, [cols, rows])} bytes, says {nb}")
            idx[name] = (gt, rows, cols, off, nb)
    return idx


class Src:
    """A tensor source: produces its bytes in chunks via a list of (path, offset, length) pieces,
    optionally widened BF16 -> F32."""

    def __init__(self, pieces, widen=False):
        self.pieces = pieces
        self.widen = widen

    def size(self):
        n = sum(p[2] for p in self.pieces)
        return n * 2 if self.widen else n

    def chunks(self, files, chunk=16 << 20):
        for path, off, ln in self.pieces:
            f = files[path]
            f.seek(off)
            left = ln
            while left:
                b = f.read(min(chunk, left))
                if not b:
                    raise EOFError(f"{path}: short read at {off}")
                left -= len(b)
                yield bf16_to_f32(b) if self.widen else b


def bf16_to_f32(b):
    if len(b) % 2:
        raise ValueError("odd BF16 byte count")
    n = len(b) // 2
    out = bytearray(n * 4)
    # each little-endian bf16 (lo, hi) becomes f32 bytes (0, 0, lo, hi)
    out[2::4] = b[0::2]
    out[3::4] = b[1::2]
    return bytes(out)


def build_plan(a):
    dense_bin = os.path.join(a.dense_dir, "dense.bin")
    di = read_dense_index(a.dense_dir)
    mg = read_gguf_header(a.mtp_gguf)
    mt = {n: (dims, gt, mg["data_start"] + off) for n, dims, gt, off in mg["tensors"]}

    p = f"blk.{MTP_BLK}."
    plan = []  # (name, dims, gtype, Src)

    def dense(out, src, dims, gtype=None, widen=False):
        gt, rows, cols, off, nb = di[src]
        want = [cols] if rows == 1 and len(dims) == 1 else [cols, rows]
        if dims != want and not (len(dims) == 2 and dims[1] == 1 and rows == 1 and dims[0] == cols):
            raise ValueError(f"{src}: dense.txt is {rows}x{cols}, mapping wants {dims}")
        if widen:
            if gt != GT_BF16:
                raise ValueError(f"{src}: widen needs BF16, is {GGML[gt][0]}")
            plan.append((p + out, dims, GT_F32, Src([(dense_bin, off, nb)], widen=True)))
        else:
            if gtype is not None and gt != gtype:
                raise ValueError(f"{src}: expected {GGML[gtype][0]}, dense.txt says {GGML[gt][0]}")
            plan.append((p + out, dims, gt, Src([(dense_bin, off, nb)])))

    def q8_rows(src, r0, r1):
        gt, rows, cols, off, nb = di[src]
        if gt != GT_Q8_0:
            raise ValueError(f"{src}: expected Q8_0")
        rb = nbytes(GT_Q8_0, [cols])
        return (dense_bin, off + r0 * rb, (r1 - r0) * rb), cols

    # hyper-connection head mixer (final, no inject)
    dense("nextn.hc_head_norm.weight", "hyper_connection_mixer.hc_norm.weight", [HC_DIM], GT_F32)
    dense("nextn.hc_head_down.weight", "hyper_connection_mixer.input_mix_weight_down.weight", [HC_DIM, HC_LR], GT_BF16)
    dense("nextn.hc_head_up.weight", "hyper_connection_mixer.input_mix_weight_up.weight", [HC_LR, HC_DIM], GT_BF16)
    # attention HC
    dense("hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight", [HC_DIM, HC], GT_BF16)
    dense("hc_attn_norm.weight", "attn_hyper_connection.hc_norm.weight", [HC_DIM], GT_F32)
    dense("hc_attn_down.weight", "attn_hyper_connection.input_mix_weight_down.weight", [HC_DIM, HC_LR], GT_BF16)
    dense("hc_attn_up.weight", "attn_hyper_connection.input_mix_weight_up.weight", [HC_LR, HC_DIM], GT_BF16)

    # routed experts, Q2_0 from mtp-q2_0.gguf
    gu_dims, gu_gt, gu_off = mt["mtp.layers.0.mlp.experts.gate_up_proj"]
    dn_dims, dn_gt, dn_off = mt["mtp.layers.0.mlp.experts.down_proj"]
    if gu_dims != [N_EMBD, 2 * N_FF_EXP, N_EXPERT] or gu_gt != GT_Q2_0:
        raise ValueError(f"gate_up_proj is {gu_dims} type {gu_gt}, expected [2560,1280,512] Q2_0")
    if dn_dims != [N_FF_EXP, N_EMBD, N_EXPERT] or dn_gt != GT_Q2_0:
        raise ValueError(f"down_proj is {dn_dims} type {dn_gt}, expected [640,2560,512] Q2_0")
    row = nbytes(GT_Q2_0, [N_EMBD])            # 720
    half = N_FF_EXP * row                      # 640 rows of one expert
    per_e = 2 * half
    gate = [(a.mtp_gguf, gu_off + e * per_e, half) for e in range(N_EXPERT)]
    up = [(a.mtp_gguf, gu_off + e * per_e + half, half) for e in range(N_EXPERT)]
    plan.append((p + "ffn_gate_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], GT_Q2_0, Src(gate)))
    plan.append((p + "ffn_up_exps.weight", [N_EMBD, N_FF_EXP, N_EXPERT], GT_Q2_0, Src(up)))
    plan.append((p + "ffn_down_exps.weight", [N_FF_EXP, N_EMBD, N_EXPERT], GT_Q2_0,
                 Src([(a.mtp_gguf, dn_off, nbytes(GT_Q2_0, dn_dims))])))

    # router + shared expert
    dense("ffn_gate_inp.weight", "mlp.gate.weight", [N_EMBD, N_EXPERT], widen=True)
    dense("ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight", [N_EMBD, 1], widen=True)
    dense("ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight", [N_EMBD, N_FF_EXP], GT_Q8_0)
    dense("ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight", [N_EMBD, N_FF_EXP], GT_Q8_0)
    dense("ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight", [N_FF_EXP, N_EMBD], GT_Q8_0)
    # MoE HC
    dense("hc_ffn_inject.weight", "mlp_hyper_connection.block_inject_weight.weight", [HC_DIM, HC], GT_BF16)
    dense("hc_ffn_norm.weight", "mlp_hyper_connection.hc_norm.weight", [HC_DIM], GT_F32)
    dense("hc_ffn_down.weight", "mlp_hyper_connection.input_mix_weight_down.weight", [HC_DIM, HC_LR], GT_BF16)
    dense("hc_ffn_up.weight", "mlp_hyper_connection.input_mix_weight_up.weight", [HC_LR, HC_DIM], GT_BF16)

    # indexer: index_qk_proj rows 0..511 -> q_proj (4 heads x 128), 512..639 -> k_proj
    piece, cols = q8_rows("self_attn.indexer.index_qk_proj.weight", 0, 512)
    plan.append((p + "indexer.q_proj.weight", [cols, 512], GT_Q8_0, Src([piece])))
    piece, cols = q8_rows("self_attn.indexer.index_qk_proj.weight", 512, 640)
    plan.append((p + "indexer.k_proj.weight", [cols, 128], GT_Q8_0, Src([piece])))
    dense("indexer.k_norm.weight", "self_attn.indexer.k_layernorm.weight", [128], GT_F32)
    dense("indexer.q_norm.weight", "self_attn.indexer.q_layernorm.weight", [128], GT_F32)

    # attention
    dense("attn_k_norm.weight", "self_attn.k_norm.weight", [256], GT_F32)
    dense("attn_output.weight", "self_attn.o_proj.weight", [6144, N_EMBD], GT_Q8_0)
    dense("attn_k.weight", "self_attn.k_proj.weight", [N_EMBD, 512], GT_Q8_0)
    dense("attn_q.weight", "self_attn.q_proj.weight", [N_EMBD, 12288], GT_Q8_0)
    dense("attn_v.weight", "self_attn.v_proj.weight", [N_EMBD, 512], GT_Q8_0)
    dense("attn_q_norm.weight", "self_attn.q_norm.weight", [256], GT_F32)

    # nextn input norms + combiner
    dense("nextn.enorm.weight", "pre_fc_norm_embedding.weight", [N_EMBD], GT_F32)
    dense("nextn.hnorm.weight", "pre_fc_norm_hidden.weight", [HC_DIM], GT_F32)
    # eh_proj = [W_e | W_h] row by row (embedding first, as the branch converter and graph concat do)
    _, er, ec, eoff, _ = di["fc_embedding.weight"]
    _, hr, hcols, hoff, _ = di["fc_hidden.weight"]
    if (er, ec, hr, hcols) != (N_EMBD, N_EMBD, N_EMBD, N_EMBD):
        raise ValueError("fc_embedding / fc_hidden must both be 2560x2560")
    for s in ("fc_embedding.weight", "fc_hidden.weight"):
        if di[s][0] != GT_Q8_0:
            raise ValueError(f"{s}: expected Q8_0")
    rb = nbytes(GT_Q8_0, [N_EMBD])            # 2720
    eh = []
    for r in range(N_EMBD):
        eh.append((dense_bin, eoff + r * rb, rb))
        eh.append((dense_bin, hoff + r * rb, rb))
    plan.append((p + "nextn.eh_proj.weight", [2 * N_EMBD, N_EMBD], GT_Q8_0, Src(eh)))

    for name, dims, gt, src in plan:
        if src.size() != nbytes(gt, dims):
            raise ValueError(f"{name}: source gives {src.size()} bytes, {GGML[gt][0]} {dims} needs {nbytes(gt, dims)}")
    return plan, (dense_bin, a.mtp_gguf)


def build_kv(target_path, n_ratio_extra=1):
    th = read_gguf_header(target_path)
    arch = None
    for k, t, v, _ in th["kv"]:
        if k == "general.architecture":
            arch = v
    if arch != "qwen4exp":
        raise ValueError(f"target arch is {arch!r}, expected qwen4exp")
    n_layer = None
    for k, t, v, _ in th["kv"]:
        if k == "qwen4exp.block_count":
            n_layer = int(v)
    if n_layer != MTP_BLK:
        raise ValueError(f"target block_count {n_layer}, expected {MTP_BLK}")
    out = []
    notes = []
    have_nextn = False
    for k, t, v, raw in th["kv"]:
        if k.startswith("split."):
            notes.append(f"dropped {k}")
            continue
        if k == "qwen4exp.block_count":
            out.append(enc_kv(k, VT_U32, n_layer + 1))
            notes.append(f"{k}: {n_layer} -> {n_layer + 1}")
            continue
        if k == "qwen4exp.nextn_predict_layers":
            have_nextn = True
            out.append(enc_kv(k, VT_U32, 1))
            continue
        if t == VT_ARR and len(v[1]) == n_layer and k.startswith("qwen4exp."):
            et, items = v
            if et not in (VT_I32, VT_U32, VT_I8, VT_U8, VT_I16, VT_U16):
                raise ValueError(f"per-layer array {k} of type {et}: no rule to extend it")
            out.append(enc_kv(k, VT_ARR, (et, list(items) + [0] * n_ratio_extra)))
            notes.append(f"{k}: extended {n_layer} -> {n_layer + n_ratio_extra} with 0 (dense)")
            continue
        out.append(raw)
        if k == "qwen4exp.expert_shared_feed_forward_length" and not have_nextn:
            out.append(enc_kv("qwen4exp.nextn_predict_layers", VT_U32, 1))
            have_nextn = True
            notes.append("added qwen4exp.nextn_predict_layers = 1")
    if not have_nextn:
        out.append(enc_kv("qwen4exp.nextn_predict_layers", VT_U32, 1))
        notes.append("added qwen4exp.nextn_predict_layers = 1")
    return out, notes, th["alignment"]


def write(a):
    plan, paths = build_plan(a)
    kvs, notes, al = build_kv(a.target_shard1)
    if al != ALIGN_DEFAULT:
        kvs.append(enc_kv("general.alignment", VT_U32, al))

    infos = []
    off = 0
    for name, dims, gt, src in plan:
        infos.append(enc_str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
                     + struct.pack("<IQ", gt, off))
        off = align(off + nbytes(gt, dims), al)
    header = GGUF_MAGIC + struct.pack("<IQQ", 3, len(plan), len(kvs)) + b"".join(kvs) + b"".join(infos)

    tmp = a.out + ".partial"
    files = {pth: open(pth, "rb") for pth in paths}
    try:
        with open(tmp, "wb") as fo:
            fo.write(header)
            fo.write(b"\0" * (align(len(header), al) - len(header)))
            base = fo.tell()
            for name, dims, gt, src in plan:
                cur = fo.tell() - base
                want = align(cur, al)
                fo.write(b"\0" * (want - cur))
                n = 0
                for b in src.chunks(files):
                    fo.write(b)
                    n += len(b)
                if n != nbytes(gt, dims):
                    raise RuntimeError(f"{name}: wrote {n} bytes, expected {nbytes(gt, dims)}")
            fo.flush()
            os.fsync(fo.fileno())
    finally:
        for f in files.values():
            f.close()
    os.replace(tmp, a.out)

    for n_ in notes:
        print("kv:", n_)
    print(f"tensors {len(plan)}  kv {len(kvs)}  bytes {os.path.getsize(a.out)}")
    for name, dims, gt, src in plan:
        print(f"  {name:40s} {str(dims):22s} {GGML[gt][0]:5s} {nbytes(gt, dims):>11d}")


def check(path):
    """Re-read the written file: header parses, offsets are aligned and contiguous, file size matches."""
    h = read_gguf_header(path)
    size = os.path.getsize(path)
    end = 0
    for name, dims, gt, off in h["tensors"]:
        if off % h["alignment"] or off < end:
            raise ValueError(f"{name}: bad offset {off}")
        end = off + nbytes(gt, dims)
    if h["data_start"] + end != size:
        raise ValueError(f"data ends at {h['data_start'] + end}, file is {size}")
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            sha.update(b)
    print(f"check ok: {len(h['tensors'])} tensors, {len(h['kv'])} kv, {size} bytes, sha256 {sha.hexdigest()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtp-gguf", required=True)
    ap.add_argument("--dense-dir", required=True)
    ap.add_argument("--target-shard1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists; refusing to overwrite")
    write(a)
    check(a.out)


if __name__ == "__main__":
    main()
