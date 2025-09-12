// Two main variants in cuGraph:

// Weakly Connected Components (WCC):

    // Implemented with parallel label propagation.

    // Each vertex starts with its own ID.

    // Iteratively updates vertex label to minimum of itself and its neighbors.

    // Uses parallel min-reductions until convergence.

    // Optimized with atomicMin operations in CUDA.

// Strongly Connected Components (SCC):

    // Based on parallel Kosaraju/Tarjan-like methods.

    // Leverages BFS/DFS sweeps on the graph and its transpose.

    // Uses GPU-level graph condensation.