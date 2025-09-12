// Classical Dijkstra = priority queue → not GPU-friendly.

// cuGraph Dijkstra:

    // Implemented using a Δ-stepping algorithm (bucket-based relaxation).

    // Vertices are grouped into buckets by distance range.

    // Each bucket processed in parallel.

    // Relax edges (u → v) with atomicMin on distance array.

    // Uses parallel frontier expansion, similar to BFS but with weights.

// Much faster than CPU due to warp-parallel edge relaxations.