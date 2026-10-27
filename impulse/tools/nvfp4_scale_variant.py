#!/usr/bin/env python3
# nvfp4_scale_variant.py: a known-answer test file for NVFP4's second-level scales.
#
# Reads a GGUF whose matmul weights are NVFP4 (llama-quantize --tensor-type ...=nvfp4) and writes a variant with real
# "<weight>.scale" tensors that computes the same function: for every NVFP4 weight llama.cpp b6b003d loads a scale for
# (its generic pass in src/llama-model.cpp: per tensor [1], per expert [n_expert] for ffn_{gate,up,down}_exps, output
# only when NVFP4), the scale is a power of two 2^k and every UE4M3 block scale of that tensor (expert) is multiplied by
# 2^-k exactly (its exponent field moved by k, or a subnormal re-encoded, only where the result is exact). A tensor or
# expert where some block scale cannot be moved exactly keeps its bytes and gets the scale 1. So
#     variant weight x its scale == original weight, value for value,
# and an engine that applies the scales as llama.cpp does gives the original's logits (bit for bit wherever the kernels
# scale by powers of two exactly), while one that ignores them is off by factors of 2 and 4. k varies with the tensor and
# with the expert (2, 0.5, 4 ...), so a scale gathered by the wrong expert id (a cache slot, say) shows too.
#
# Usage: nvfp4_scale_variant.py IN.gguf OUT.gguf [--check N] [--only-exps] [--skip REGEX]
#   --check N   dequantize N tensors (and a few experts of each expert tensor) of both files on the CPU and require
#               variant x scale == original exactly (default 8; 0 = off)
# Needs numpy and llama.cpp's gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
import argparse
import re
import sys

import numpy as np

import gguf
from gguf.quants import dequantize

NVFP4 = gguf.GGMLQuantizationType.NVFP4
BLOCK = 36              # block_nvfp4: d[4] (UE4M3) + qs[32], 64 values
PER_TENSOR = re.compile(r"^blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|attn_qkv|attn_gate|ffn_gate|ffn_up|ffn_down|"
                        r"ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp|ssm_in|ssm_out|ssm_alpha|ssm_beta|"
                        r"nextn\.eh_proj|nextn\.shared_head_head)\.weight$")
PER_EXPERT = re.compile(r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$")


def ue4m3(b: int) -> float:   # ggml_ue4m3_to_fp32 (ggml-impl.h), exactly
    if b == 0 or b == 0x7F:
        return 0.0
    e, m = (b >> 3) & 0xF, b & 7
    raw = m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return raw * 0.5


def shift_table(k: int) -> np.ndarray:
    """byte -> the byte whose value is value(byte) * 2^-k exactly, -1 where there is none (bit 7 set: never moved)."""
    val_to_byte = {}
    for b in range(0x7F):            # 0x00..0x7E: the encodings ggml writes (0x7F also decodes to 0)
        val_to_byte.setdefault(ue4m3(b), b)
    t = np.full(256, -1, np.int16)
    for b in range(0x80):
        v = ue4m3(b)
        if v == 0.0:
            t[b] = b                 # a zero block scale stays as it is
            continue
        w = v * 2.0 ** -k
        if w in val_to_byte and val_to_byte[w] != 0:
            t[b] = val_to_byte[w]
    return t


TABLES = {k: shift_table(k) for k in (-2, -1, 1, 2)}
PLAN = (-1, 1, -2)                   # scale 0.5, 2, 4 in turn (0.5 and 4 are exact almost everywhere; 2 where it can)


def moved(d: np.ndarray, k: int):
    """d: uint8 block scales. Returns the moved bytes, or None when one of them cannot move exactly."""
    if k == 0:
        return d
    nd = TABLES[k][d]
    return None if (nd < 0).any() else nd.astype(np.uint8)


def variant_of(blocks: np.ndarray, want: int):
    """blocks: uint8 [n, 36] of one tensor or expert. Tries k = want, then -1; returns (k, new blocks)."""
    d = blocks[:, :4]
    for k in (want, -1, 0):
        nd = moved(d, k)
        if nd is not None:
            if k == 0:
                return 0, blocks
            out = blocks.copy()
            out[:, :4] = nd
            return k, out
    raise AssertionError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--check", type=int, default=8)
    ap.add_argument("--only-exps", action="store_true", help="scale only the per-expert tensors")
    ap.add_argument("--skip", default=None, help="a regex of weight names to leave without a scale")
    a = ap.parse_args()

    r = gguf.GGUFReader(a.input)
    arch = r.fields["general.architecture"].contents()
    names = {t.name for t in r.tensors}
    tied_output = "output.weight" not in names
    skip = re.compile(a.skip) if a.skip else None

    w = gguf.GGUFWriter(a.output, arch=arch, endianess=r.endianess)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=sub)

    # the plan: which weights get a scale, and each one's (or each expert's) power of two
    plan = {}          # weight name -> (scale name, list of k per expert or [k])
    out_data = {}      # tensor name -> new data (moved blocks), for the weights that changed
    new_scales = {}    # scale name -> float32 array
    stats = {"tensors": 0, "experts": 0, "experts_scaled": 0, "tensors_scaled": 0}
    for ti, t in enumerate(r.tensors):
        if t.tensor_type != NVFP4:
            continue
        per_expert = bool(PER_EXPERT.match(t.name))
        per_tensor = bool(PER_TENSOR.match(t.name)) or (t.name == "output.weight" and not tied_output)
        if not (per_expert or per_tensor) or (a.only_exps and not per_expert) or (skip and skip.search(t.name)):
            continue
        sname = t.name[: -len(".weight")] + ".scale"
        old = None
        if sname in names:   # a scale already there (Gemma 4's ffn_down_exps.scale): it is multiplied
            old = next(x for x in r.tensors if x.name == sname)
            old = np.array(old.data, dtype=np.float32).reshape(-1)
        data = np.asarray(t.data)
        if per_expert:
            n_e = data.shape[0]
            ks, parts = [], []
            for e in range(n_e):
                k, nb = variant_of(np.ascontiguousarray(data[e]).reshape(-1, BLOCK), PLAN[(e + ti) % 3])
                ks.append(k); parts.append(nb.reshape(data[e].shape))
                stats["experts"] += 1; stats["experts_scaled"] += k != 0
            if any(ks):
                out_data[t.name] = np.stack(parts)
        else:
            k, nb = variant_of(np.ascontiguousarray(data).reshape(-1, BLOCK), PLAN[ti % 3])
            ks = [k]
            stats["tensors"] += 1; stats["tensors_scaled"] += k != 0
            if k:
                out_data[t.name] = nb.reshape(data.shape)
        s = np.array([2.0 ** k for k in ks], dtype=np.float32)
        if old is not None:
            if old.size != s.size:
                print(f"{sname}: {old.size} values, expected {s.size}", file=sys.stderr)
                return 1
            s = (old * s).astype(np.float32)   # powers of two: exact
        new_scales[sname] = s
        plan[t.name] = (sname, ks)

    # tensor infos: every tensor, each new scale right after its weight
    order = []
    for t in r.tensors:
        if t.name in new_scales and t.name not in plan:   # an existing scale: written with its new values, in place
            order.append((t.name, new_scales[t.name], gguf.GGMLQuantizationType.F32))
            continue
        d = out_data.get(t.name, t.data)
        order.append((t.name, d, t.tensor_type))
        if t.name in plan:
            sname = plan[t.name][0]
            if sname not in names:
                order.append((sname, new_scales[sname], gguf.GGMLQuantizationType.F32))
    for name, d, qt in order:
        w.add_tensor_info(name, d.shape, d.dtype, d.nbytes, qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for name, d, qt in order:
        w.write_tensor_data(np.asarray(d), tensor_endianess=r.endianess)
    w.close()
    print(f"{a.output}: {len(plan)} weights with a scale; per-tensor {stats['tensors_scaled']}/{stats['tensors']} moved, "
          f"experts {stats['experts_scaled']}/{stats['experts']} moved (the rest keep their bytes and scale 1)")

    if a.check > 0:   # variant x scale == original, value for value (the CPU's dequantization, ggml's decoding)
        rv = gguf.GGUFReader(a.output)
        vt = {t.name: t for t in rv.tensors}
        checked = 0
        for name, (sname, ks) in plan.items():
            if checked >= a.check and not PER_EXPERT.match(name):
                continue
            to = next(x for x in r.tensors if x.name == name)
            tv = vt[name]
            sv = np.array(vt[sname].data, dtype=np.float32).reshape(-1)
            if PER_EXPERT.match(name):
                for e in sorted({0, 1, 2, len(ks) - 1}):
                    o = dequantize(np.asarray(to.data[e]), NVFP4)
                    v = dequantize(np.asarray(tv.data[e]), NVFP4) * sv[e]
                    if not np.array_equal(o, v):
                        print(f"CHECK FAILED: {name} expert {e} (k {ks[e]})", file=sys.stderr)
                        return 1
            else:
                o = dequantize(np.asarray(to.data), NVFP4)
                v = dequantize(np.asarray(tv.data), NVFP4) * sv[0]
                if not np.array_equal(o, v):
                    print(f"CHECK FAILED: {name} (k {ks[0]})", file=sys.stderr)
                    return 1
            checked += 1
        print(f"check: {checked} weights, variant x scale == original exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
