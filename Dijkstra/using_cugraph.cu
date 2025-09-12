// dijkstra_cugraph_usage.cpp
#include <cugraph/algorithms.hpp>

auto [distances, predecessors] =
    cugraph::sssp<int32_t, int32_t, float>(
        handle, graph, 0 /*source*/);

