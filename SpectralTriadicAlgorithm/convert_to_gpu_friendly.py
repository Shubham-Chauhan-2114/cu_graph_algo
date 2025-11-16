#!/usr/bin/env python3
# convert_to_gpu_friendly.py  (NumPy required)
import numpy as np
import os, json, sys

DATA_PATH = "sampleDataset.txt"
OUT_DIR = "gpu_data"
os.makedirs(OUT_DIR, exist_ok=True)

def read_edge_list(path):
    edges = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.lower().startswith('source'):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except:
                u = parts[0]; v = parts[1]
            edges.append((u, v))
    return edges

print("Reading edges...")
edges_raw = read_edge_list(DATA_PATH)
print("Raw edges lines:", len(edges_raw))

# Map to compact integer ids
nodes = {}
inv_nodes = []
def get_id(x):
    if x in nodes: return nodes[x]
    idx = len(inv_nodes)
    nodes[x] = idx; inv_nodes.append(x); return idx

for (u,v) in edges_raw:
    get_id(u); get_id(v)
n = len(inv_nodes)
print("Unique nodes:", n)

# canonicalize unique undirected edges (no self-loops)
edge_set = set()
for (u0,v0) in edges_raw:
    u = nodes[u0]; v = nodes[v0]
    if u == v: continue
    a,b = (u,v) if u <= v else (v,u)
    edge_set.add((a,b))

m = len(edge_set)
print("Unique undirected edges:", m)

# create directed adjacency entries (both directions)
directed = np.empty((2*m,2), dtype=np.int64)
i = 0
for (a,b) in edge_set:
    directed[i,0] = a; directed[i,1] = b; i+=1
    directed[i,0] = b; directed[i,1] = a; i+=1

src = directed[:,0]; dst = directed[:,1]

# build CSR
deg = np.bincount(src, minlength=n).astype(np.int64)
indptr = np.empty(n+1, dtype=np.int64); indptr[0]=0; np.cumsum(deg, out=indptr[1:])
indices = np.empty(indptr[-1], dtype=np.int64)
offsets = indptr[:-1].copy()
for s,d in zip(src, dst):
    idx = offsets[s]; indices[idx] = d; offsets[s] += 1

# sort neighbor lists
for u in range(n):
    a = indptr[u]; b = indptr[u+1]
    if b - a > 1:
        indices[a:b].sort()

degree = (indptr[1:] - indptr[:-1]).astype(np.int64)

# Save binary artifacts
np.save(os.path.join(OUT_DIR, "indptr.npy"), indptr)
np.save(os.path.join(OUT_DIR, "indices.npy"), indices)
np.save(os.path.join(OUT_DIR, "degree.npy"), degree)
np.save(os.path.join(OUT_DIR, "nodes.npy"), np.array(inv_nodes, dtype=object))
np.save(os.path.join(OUT_DIR, "edges.npy"), np.array(sorted(edge_set), dtype=np.int64))
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump({"n":int(n),"m":int(m)}, f, indent=2)

print("Saved to", OUT_DIR)
print("n:", n, "m:", m)
