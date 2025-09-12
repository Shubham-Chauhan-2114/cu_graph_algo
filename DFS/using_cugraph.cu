// bfs_cugraph_usage.cpp
#include <raft/core/handle.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/algorithms.hpp>
#include <rmm/device_uvector.hpp>
#include <raft/util/cuda_utils.cuh>
#include <iostream>

int main() {
    raft::handle_t handle;

    // Graph edges: 0->1, 0->2, 1->3, 2->4
    std::vector<int32_t> h_src{0, 0, 1, 2};
    std::vector<int32_t> h_dst{1, 2, 3, 4};

    rmm::device_uvector<int32_t> d_src(h_src.size(), handle.get_stream());
    rmm::device_uvector<int32_t> d_dst(h_dst.size(), handle.get_stream());
    raft::update_device(d_src.data(), h_src.data(), h_src.size(), handle.get_stream());
    raft::update_device(d_dst.data(), h_dst.data(), h_dst.size(), handle.get_stream());

    // Build graph (CSR)
    auto [graph, edge_weights, renumber_map] =
        cugraph::create_graph_from_edgelist<int32_t, int32_t, float>(
            handle, d_src.data(), d_dst.data(), nullptr,
            h_src.size(), 5, false, true);

    // Run BFS
    auto [distances, predecessors] =
        cugraph::bfs<int32_t, int32_t, float>(handle, graph, 0 /*start*/, false);

    std::cout << "BFS executed successfully." << std::endl;
}
