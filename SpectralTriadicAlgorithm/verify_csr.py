#!/usr/bin/env python3
# verify_csr.py (NumPy required)
import numpy as np, os, json, random, sys

OUT_DIR = "gpu_data"
with open(os.path.join(OUT_DIR, "meta.json")) as f: meta = json.load(f)
n = meta["n"]; m = meta["m"]
indptr = np.load(os.path.join(OUT_DIR, "indptr.npy"))
indices = np.load(os.path.join(OUT_DIR, "indices.npy"))
degree = np.load(os.path.join(OUT_DIR, "degree.npy"))

print("Loaded CSR: n =", n, "m =", m)
print("directed entries:", len(indices), "sum degrees:", degree.sum())

# basic checks
assert indptr.shape[0] == n+1, "indptr length != n+1"
assert indptr[0] == 0 and indptr[-1] == len(indices), "indptr endpoints invalid"
assert np.all(indptr[1:] >= indptr[:-1]), "indptr not nondecreasing"
deg_from_csr = indptr[1:] - indptr[:-1]
assert np.array_equal(degree, deg_from_csr), "degree mismatch"

# no self loops and bounds
self_loops = 0
for u in range(n):
    for v in indices[indptr[u]:indptr[u+1]]:
        if u == v: self_loops += 1
        if v < 0 or v >= n: raise SystemExit("invalid neighbor index")
assert self_loops == 0, f"self loops found: {self_loops}"

# symmetry check
sym_errors = 0
for u in range(n):
    neigh = indices[indptr[u]:indptr[u+1]]
    for v in neigh:
        row_v = indices[indptr[v]:indptr[v+1]]
        i = np.searchsorted(row_v, u)
        if i >= len(row_v) or row_v[i] != u:
            sym_errors += 1
assert sym_errors == 0, f"missing reverse edges: {sym_errors}"

# duplicates
dup_count = 0
for u in range(n):
    neigh = indices[indptr[u]:indptr[u+1]]
    if len(neigh) > 1:
        dup_count += np.sum(neigh[1:] == neigh[:-1])
assert dup_count == 0, f"duplicate entries: {dup_count}"

print("All checks passed.")
# sample neighborhoods
for _ in range(3):
    u = random.randint(0, n-1)
    print(f"node {u} deg {len(indices[indptr[u]:indptr[u+1]])} neighbors (first10):",
          indices[indptr[u]:indptr[u+1]][:10])
