#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/algorithms.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <iostream>
#include <vector>

// Utility for CUDA error checking
#define CUDA_TRY(call)                                                           
  do {                                                                           
    cudaError_t status = call;                                                   
    if (status != cudaSuccess) {                                                 
      std::cerr << "CUDA error: " << cudaGetErrorString(status)                  
                << " at line " << __LINE__ << std::endl;                         
      exit(1);                                                                   
    }                                                                            
  } while (0)

int main(int argc, char** argv) {
  raft::handle_t handle; // cuGraph needs a handle (manages streams, resources)
  auto stream = handle.get_stream();

  using vertex_t = int32_t;
  using edge_t   = int32_t;
  using weight_t = float;

  // -------------------------------
  // 1. Define edge list on host
  // Example: simple undirected chain graph
  //   0 --1--> 1 --1--> 2 --1--> 3
  std::vector<vertex_t> h_src{0, 1, 2};
  std::vector<vertex_t> h_dst{1, 2, 3};
  std::vector<weight_t> h_wgt{1.0, 1.0, 1.0};

  edge_t num_edges = static_cast<edge_t>(h_src.size());
  vertex_t num_vertices = 4;

  // -------------------------------
  // 2. Copy to device
  rmm::device_uvector<vertex_t> d_src(num_edges, stream);
  rmm::device_uvector<vertex_t> d_dst(num_edges, stream);
  rmm::device_uvector<weight_t> d_wgt(num_edges, stream);

  CUDA_TRY(cudaMemcpyAsync(d_src.data(), h_src.data(),
                           num_edges * sizeof(vertex_t),
                           cudaMemcpyHostToDevice, stream));

  CUDA_TRY(cudaMemcpyAsync(d_dst.data(), h_dst.data(),
                           num_edges * sizeof(vertex_t),
                           cudaMemcpyHostToDevice, stream));

  CUDA_TRY(cudaMemcpyAsync(d_wgt.data(), h_wgt.data(),
                           num_edges * sizeof(weight_t),
                           cudaMemcpyHostToDevice, stream));

  CUDA_TRY(cudaStreamSynchronize(stream));

  // -------------------------------
  // 3. Build graph
  // NOTE: store_transposed::NO => adjacency in CSR format
  cugraph::graph_t<vertex_t, edge_t, weight_t, false, true> graph(handle);

  // Construct graph from edge list
  std::tie(graph, std::ignore) = cugraph::from_edgelist<vertex_t, edge_t, weight_t>(
      handle,
      d_src.data(),
      d_dst.data(),
      d_wgt.data(),
      num_edges,
      cugraph::store_transposed::NO,
      false // multi-GPU = false
  );

  // -------------------------------
  // 4. Print graph properties
  auto num_verts = graph.number_of_vertices();
  auto num_edges_in_graph = graph.number_of_edges();

  std::cout << "Graph successfully created!" << std::endl;
  std::cout << "Number of vertices: " << num_verts << std::endl;
  std::cout << "Number of edges: " << num_edges_in_graph << std::endl;

  return 0;
}
