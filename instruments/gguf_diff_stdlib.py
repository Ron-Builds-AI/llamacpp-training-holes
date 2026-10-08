"""Compare the tensors of two GGUF files by name, standard library only (no numpy on the Dell).

usage: python3 gguf_diff_stdlib.py BASE.gguf OTHER.gguf

For every tensor present in both files it prints whether the raw bytes are identical.
For F32 tensors whose name contains attn_q, attn_k or attn_v it also counts changed elements
and the largest absolute change. Other tensors are compared by bytes only (a pure-Python
per-element loop over the FFN and embedding tables would take too long).
Instrument check: run it with the same file twice, every line must say identical.
"""
import mmap
import struct
import sys
from array import array

# GGUF value types
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0

    def read(self, fmt):
        size = struct.calcsize(fmt)
        val = struct.unpack_from(fmt, self.m, self.pos)
        self.pos += size
        return val[0] if len(val) == 1 else val

    def string(self):
        n = self.read("<Q")
        s = bytes(self.m[self.pos:self.pos + n]).decode("utf-8", "replace")
        self.pos += n
        return s

    def value(self, vtype):
        if vtype in _SCALAR:
            return self.read(_SCALAR[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            atype = self.read("<I")
            n = self.read("<Q")
            if atype in _SCALAR:  # skip the payload quickly
                self.pos += n * struct.calcsize(_SCALAR[atype])
                return None
            for _ in range(n):
                self.value(atype)
            return None
        raise ValueError("unknown gguf value type %d" % vtype)


def load(path):
    r = Reader(path)
    magic = bytes(r.m[0:4])
    if magic != b"GGUF":
        raise ValueError("%s: not a GGUF file" % path)
    r.pos = 4
    version = r.read("<I")
    n_tensors = r.read("<Q")
    n_kv = r.read("<Q")
    alignment = 32
    for _ in range(n_kv):
        key = r.string()
        vtype = r.read("<I")
        val = r.value(vtype)
        if key == "general.alignment":
            alignment = val
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.read("<I")
        ne = [r.read("<Q") for _ in range(n_dims)]
        ttype = r.read("<I")
        offset = r.read("<Q")
        infos.append((name, ne, ttype, offset))
    data_start = (r.pos + alignment - 1) // alignment * alignment
    # byte size of each tensor = distance to the next tensor (or to the end of the file)
    order = sorted(infos, key=lambda t: t[3])
    tensors = {}
    for i, (name, ne, ttype, offset) in enumerate(order):
        start = data_start + offset
        end = data_start + order[i + 1][3] if i + 1 < len(order) else len(r.m)
        n = 1
        for d in ne:
            n *= d
        if ttype == 0:  # F32: exact size, no padding
            end = start + 4 * n
        tensors[name] = (ne, ttype, start, end, n)
    return r, version, tensors


def bytes_equal(ma, a0, a1, mb, b0, b1, chunk=1 << 24):
    if a1 - a0 != b1 - b0:
        return False
    off = 0
    size = a1 - a0
    while off < size:
        k = min(chunk, size - off)
        if ma[a0 + off:a0 + off + k] != mb[b0 + off:b0 + off + k]:
            return False
        off += k
    return True


def f32_stats(ma, a0, mb, b0, n):
    xa = array("f")
    xb = array("f")
    xa.frombytes(ma[a0:a0 + 4 * n])
    xb.frombytes(mb[b0:b0 + 4 * n])
    changed = 0
    maxabs = 0.0
    for p, q in zip(xa, xb):
        if p != q:
            changed += 1
            d = abs(p - q)
            if d > maxabs:
                maxabs = d
    return changed, maxabs


def main():
    pa, pb = sys.argv[1], sys.argv[2]
    ra, va, ta = load(pa)
    rb, vb, tb = load(pb)
    print("A = %s (gguf v%d, %d tensors)" % (pa, va, len(ta)))
    print("B = %s (gguf v%d, %d tensors)" % (pb, vb, len(tb)))
    only_a = sorted(set(ta) - set(tb))
    only_b = sorted(set(tb) - set(ta))
    print("only in A: %s" % (", ".join(only_a) if only_a else "none"))
    print("only in B: %s" % (", ".join(only_b) if only_b else "none"))
    kinds = {}
    for name in sorted(set(ta) & set(tb)):
        nea, tya, a0, a1, n = ta[name]
        neb, tyb, b0, b1, nb = tb[name]
        kind = name.split(".", 2)[2] if name.startswith("blk.") else name
        same = (nea == neb and tya == tyb) and bytes_equal(ra.m, a0, a1, rb.m, b0, b1)
        line = "%-40s type=%d ne=%s bytes_identical=%s" % (name, tya, nea, same)
        detail = tya == 0 and tyb == 0 and nea == neb and any(k in name for k in ("attn_q", "attn_k", "attn_v"))
        changed = None
        if detail:
            changed, maxabs = f32_stats(ra.m, a0, rb.m, b0, n)
            line += " changed=%d/%d max_abs=%.6e" % (changed, n, maxabs)
        print(line)
        k = kinds.setdefault(kind, [0, 0])
        k[0] += 1
        k[1] += 0 if same else 1
    print("SUMMARY kind / tensors / tensors_changed")
    for kind in sorted(kinds):
        print("SUMMARY %-28s %4d %4d" % (kind, kinds[kind][0], kinds[kind][1]))


if __name__ == "__main__":
    main()
