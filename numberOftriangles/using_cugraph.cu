// triangle_cugraph_usage.cpp
#include <cugraph/algorithms.hpp>

auto result = cugraph::triangle_count<int32_t, int32_t>(
    handle, graph, false /*is_directed*/);

std::cout << "Triangle count: " << result << std::endl;
