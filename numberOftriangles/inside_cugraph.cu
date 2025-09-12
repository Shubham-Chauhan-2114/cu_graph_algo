// Triangle counting requires checking if (u,v,w) forms a cycle of length 3.

// cuGraph implementation:

    // Works on CSR adjacency.

    // For each edge (u,v), find intersection of adjacency lists of u and v.

    // Intersections computed using merge-based parallel algorithms.

    // Optimized with warp-level primitives (__shfl_sync, etc.) and bitset intersections.

    // Supports per-vertex triangle count or global count.