#!/usr/bin/env python3
"""A GGUF holding only another GGUF's metadata (every KV pair, no tensors): enough for a vocab-only load (the tokenizer,
the chat template, the architecture's hyperparameters) without the weights.  Standard library only (no numpy, no gguf-py).

    gguf_meta_only.py SRC OUT        SRC may be just the file's first megabytes (head -c 64M model.gguf > head.bin):
                                     only the header and the KV section are read.
    gguf_meta_only.py --dump SRC     the KV keys, types and (short) values

One Engine P1 (tokenizer parity on a PC that does not hold the model): the dense GGUF of a Flash-Next pack carries the
whole tokenizer, so a few MB of it stand in for the 6 GB file.
"""
import struct
import sys

# GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FIX = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q",
        F64: "<d"}


class Reader:
    def __init__(self, buf: bytes):
        self.b, self.o = buf, 0

    def take(self, n: int) -> bytes:
        if self.o + n > len(self.b):
            raise EOFError("the file ends inside the metadata: give more of it")
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def fix(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> bytes:
        return self.take(self.fix("<Q"))

    def value(self, t: int):
        if t in _FIX:
            return self.fix(_FIX[t])
        if t == STR:
            return self.string()
        if t == ARR:
            et, n = self.fix("<I"), self.fix("<Q")
            return (et, [self.value(et) for _ in range(n)])
        raise ValueError(f"unknown GGUF value type {t}")


def read_kv(buf: bytes):
    r = Reader(buf)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = r.fix("<I")
    n_tensors, n_kv = r.fix("<Q"), r.fix("<Q")
    kv = []
    for _ in range(n_kv):
        key = r.string()
        t = r.fix("<I")
        kv.append((key, t, r.value(t)))
    return version, n_tensors, kv


def write_value(out: list, t: int, v):
    if t in _FIX:
        out.append(struct.pack(_FIX[t], v))
    elif t == STR:
        out.append(struct.pack("<Q", len(v)) + v)
    elif t == ARR:
        et, items = v
        out.append(struct.pack("<IQ", et, len(items)))
        for x in items:
            write_value(out, et, x)
    else:
        raise ValueError(t)


def write_meta(path: str, version: int, kv) -> None:
    out = [b"GGUF", struct.pack("<IQQ", version, 0, len(kv))]
    for key, t, v in kv:
        out.append(struct.pack("<Q", len(key)) + key + struct.pack("<I", t))
        write_value(out, t, v)
    data = b"".join(out)
    align = 32
    for key, t, v in kv:
        if key == b"general.alignment":
            align = v
    data += b"\0" * ((-len(data)) % align)
    with open(path, "wb") as f:
        f.write(data)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--dump":
        version, n_tensors, kv = read_kv(open(sys.argv[2], "rb").read())
        print(f"GGUF v{version}, {n_tensors} tensors, {len(kv)} keys")
        for key, t, v in kv:
            if t == ARR:
                s = f"array of {len(v[1])} (type {v[0]})"
            elif t == STR:
                s = repr(v[:80].decode("utf-8", "replace")) + (f" ... ({len(v)} bytes)" if len(v) > 80 else "")
            else:
                s = repr(v)
            print(f"  {key.decode()}: {s}")
        return 0
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    version, n_tensors, kv = read_kv(open(sys.argv[1], "rb").read())
    write_meta(sys.argv[2], version, kv)
    print(f"{sys.argv[2]}: {len(kv)} keys of {sys.argv[1]} (its {n_tensors} tensors left out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
