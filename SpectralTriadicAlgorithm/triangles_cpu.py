#!/usr/bin/env python3
# triangles_cpu.py (NumPy required) - creates edge->triangles lists
import numpy as np, os, json
OUT_DIR = "gpu_data"
with open(os.path.join(OUT_DIR, "meta.json")) as f: meta = json.load(f)
n = meta["n"]
indptr = np.load(os.path.join(OUT_DIR, "indptr.npy"))
indices = np.load(os.path.join(OUT_DIR, "indices.npy"))
edges = np.load(os.path.join(OUT_DIR, "edges.npy"))  # undirected unique edges (a,b) with a<b

# Build a mapping from (u,v) directed edge -> edge_id
# We will create a dictionary for quick lookup of directed-edge index
# We'll create an edge id for the directed adjacency position index
# For convenience: we create an edge_id for each directed entry (position in indices)
edge_pos_to_edgeid = {}
edge_count_directed = len(indices)
# map (u, neighbor_index_index) -> edge_id
eid = 0
for u in range(n):
    for pos in range(indptr[u], indptr[u+1]):
        edge_pos_to_edgeid[(u, pos)] = eid
        eid += 1

# Create a map from (min(u,v), max(u,v)) -> some undirected edge id (from edges.npy)
undirected_edge_to_index = {}
for idx, (a,b) in enumerate(edges):
    undirected_edge_to_index[(a,b)] = idx

# We'll store triangles by undirected-edge index (as paper suggests)
tri_thirds = []      # concatenated third vertices
tri_off = [0] * (len(edges)+1)  # offsets per undirected edge
# Approach: iterate over u, for each pair v>w in neighbors(u) find intersection via binary search
# Simpler: for each edge (a,b) with a < b, intersect neighbor lists of a and b for w > b to avoid duplicates
total = 0
for i,(a,b) in enumerate(edges):
    # intersect neighbors of a and b
    neigh_a = indices[indptr[a]:indptr[a+1]]
    neigh_b = indices[indptr[b]:indptr[b+1]]
    # classic merge intersection
    p = q = 0
    cnt = 0
    while p < len(neigh_a) and q < len(neigh_b):
        x = neigh_a[p]; y = neigh_b[q]
        if x == y:
            if x != a and x != b:
                tri_thirds.append(int(x))
                cnt += 1
            p += 1; q += 1
        elif x < y:
            p += 1
        else:
            q += 1
    total += cnt
    tri_off[i+1] = len(tri_thirds)

tri_thirds = np.array(tri_thirds, dtype=np.int64)
tri_off = np.array(tri_off, dtype=np.int64)
np.save(os.path.join(OUT_DIR, "tri_off.npy"), tri_off)
np.save(os.path.join(OUT_DIR, "tri_thirds.npy"), tri_thirds)
print("Triangles stored. total third-entries:", tri_thirds.shape[0], "triangle_off len:", tri_off.shape[0])
