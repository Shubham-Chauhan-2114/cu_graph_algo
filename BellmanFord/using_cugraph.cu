// ❌ cuGraph does not provide Bellman-Ford.

// GPUs usually avoid Bellman-Ford (too sequential, O(VE)).

// Instead, cuGraph recommends SSSP (delta-stepping).