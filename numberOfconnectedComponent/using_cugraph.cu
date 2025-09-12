// cc_cugraph_usage.cpp
#include <cugraph/algorithms.hpp>

auto [labels, num_components] =
    cugraph::connected_components<int32_t, int32_t, float>(
        handle, graph, cugraph::components_type_t::WEAK);
