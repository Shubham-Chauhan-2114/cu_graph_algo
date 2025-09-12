// Bellman-Ford is iterative edge relaxation.

// cuGraph implementation:

    // For V-1 iterations (or until no update):

    // Launch parallel kernel to relax all edges simultaneously.

        // Use atomicMin for updating distances.

        // Early-stopping if no distance update detected.

    // More suitable for negative weight edges (unlike Dijkstra).

// Heavily parallelized with CUDA atomics + device-wide reductions.