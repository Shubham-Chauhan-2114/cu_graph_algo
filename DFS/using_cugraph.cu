// DFS is sequential by nature, harder to parallelize.

// cuGraph DFS:

    // Not a simple stack recursion → instead uses parallel edge relaxation + masking.

    // Maintains a GPU-side stack frontier but processed in waves (like BFS).

    // Each warp explores neighbor chains as long as possible until divergence.

    // Focus is on traversal marking (discover time, finish time) rather than recursive structure.