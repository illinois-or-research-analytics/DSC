#pragma GCC optimize("Ofast")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <set>
#include <utility>
#include <cstring>
#include <queue>
#include <numeric>
#include <iomanip> // For std::setprecision

#if PARALLEL
#include <omp.h>
#endif

using namespace std;

vector<bool> fractional_peeling(
    const vector<vector<pair<int, int>>> &Adj,
    vector<double> &b,
    const vector<double> &x,
    long long n,
    long long m)
{
  /*
  Performs fractional peeling on the graph with load vector b and solution vector x.
  Arguments:
  - Adj: Adjacency list of the graph.
  - b: Load vector for each node.
  - x: Solution vector.
  - n: Number of nodes in the graph.
  - m: Number of edges in the graph.
  Returns:
  - A vector of booleans indicating which nodes are in the densest subgraph found.
  */

  // Initialize priority queue for fractional peeling
  // The priority queue will store pairs of (b[i], i) for each node i
  // The queue is ordered by load in ascending order
  // so that we can always process the node with the smallest load first
  priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
  for (int i = 0; i < n; ++i)
  {
    pq.push({b[i], i});
  }

  // Create a vector to keep track of deleted nodes
  vector<bool> deleted(n, false);

  // Initialize the running number of nodes and edges
  long long N = n;
  long long M = m;

  // Initialize the density of the graph
  double density = (double)(1.0 * M / N);

  // Create a vector to keep track of nodes in the current subset
  // Initially, all nodes are considered part of the current subset
  vector<bool> current_subset(n, true);

  // Iterate the priority queue until it is empty
  // This loop will peel off nodes with the smallest load and update the graph accordingly
  while (!pq.empty())
  {
    // Get the node i with the smallest load (d = b[i]) from the priority queue
    // and remove it from the queue
    auto [d, i] = pq.top();
    pq.pop();

    // If the node i has already been deleted, skip it
    if (deleted[i])
      continue;

    // Iterate through the neighbors of node i in the adjacency list
    for (auto &[j, e_idx] : Adj[i])
    {
      // For each neighbor j with edge (i, j) indexed by e_idx,

      // Find the index of the sister edge (the other direction of the edge)
      int sister_idx = e_idx % 2 ? e_idx - 1 : e_idx + 1;

      // Check if the neighbor j has not been deleted
      if (!deleted[j])
      {
        // If not deleted
        // Remove the edge and update the load of neighbor j

        // Update the load of neighbor j
        // by subtracting the solution value corresponding to the sister edge
        b[j] -= x[sister_idx];

        // Push the updated load of neighbor j into the priority queue
        // This is to update the position of neighbor j in the queue
        pq.push({b[j], j});

        // Decrease the number of edges M by 1
        M--;
      }
    }

    // Remove the node i from the graph

    // Mark the node i as deleted
    deleted[i] = true;

    // Decrease the number of nodes N by 1
    N--;

    // Check if there are still nodes left in the graph
    if (N > 0)
    {
      // If there are nodes left

      // Check if the current density (M / N) is greater than the previous density
      if ((1.0 * M / N) > density)
      {
        // If the current density is greater

        // Update the density to the current density
        density = 1.0 * M / N;

        // Update the current subset to reflect the nodes that are not deleted
        for (int i = 0; i < n; i++)
        {
          current_subset[i] = !deleted[i];
        }
      }
    }
  }
  return current_subset;
}

pair<vector<bool>, vector<double>> fista_frac(
    const vector<vector<pair<int, int>>> &Adj,
    long long iterations,
    long long n,
    long long m)
{
  /*
  FISTA algorithm implementation for network optimization.
  Arguments:
  - Adj: Adjacency list of the network, where each node has a list of its neighbors and the corresponding edge indices.
  - iterations: Number of iterations to run the FISTA algorithm.
  - n: Number of nodes in the network.
  - m: Number of edges in the network.
  Returns:
  - A vector of doubles representing the load on each node after running the FISTA algorithm.
  */

  // Create a mapping from directed edge index to its reverse
  // For each edge (i, j) in the adjacency list, we have two directed edges:
  // (i -> j) and (j -> i). The reverse edge index is the index of the edge in the opposite direction.
  int *reverse_edge_idx = new int[2 * m];
  int *edge_src_indices = new int[2 * m];
  // Iterate over all nodes
#pragma omp parallel for
  for (int i = 0; i < n; ++i)
  {
    // Iterate over all edges of the current node
    for (int j = 0; j < Adj[i].size(); ++j)
    {
      // sister_idx is the index of the directed edge (j -> i)
      int sister_idx = Adj[i][j].second;
      // idx is the index of the directed edge (i -> j)
      int idx = (sister_idx % 2 == 0 ? sister_idx + 1 : sister_idx - 1);

      // Source node of the idx edge is the current node i
      edge_src_indices[idx] = i;
      // sister_idx is the index of the reverse edge of the idx edge
      reverse_edge_idx[idx] = sister_idx;
    }
  }

  // Learning rate is set based on the maximum degree of the graph
  double max_degree = (*std::max_element(
                           Adj.begin(), Adj.end(),
                           [](auto &a, auto &b)
                           { return a.size() < b.size(); }))
                          .size();
  double learning_rate = 0.5 / max_degree;

  // Allocate FISTA variables: x = current iterate, y = momentum mix, z = gradient step
  vector<double> x(2 * m, 0.5);
  vector<double> y(2 * m, 0.5);
  vector<double> z(2 * m, 0.0);

  // Load vector b initialized to 0
  vector<double> b(n, 0);

  // Momentum parameter tk initialized to 1.0
  double tk = 1.0;

  // Buffers for new updates, allocated once
  double *new_xuv = new double[2 * m];
  double *new_y = new double[2 * m];

  // Iterate for the specified number of iterations
  for (int t = 0; t < iterations; ++t)
  {
    // Gradient computation: b[u] = sum of y over edges outgoing from u
    std::fill(b.begin(), b.end(), 0.0); // Reset b to zero
    for (int i = 0; i < 2 * m; ++i)
      b[edge_src_indices[i]] += y[i];

    // Gradient step: z_{uv} = y_{uv} - 2 * lr * b_u
#if PARALLEL
#pragma omp parallel for
#endif
    for (int i = 0; i < 2 * m; ++i)
      z[i] = y[i] - 2.0 * learning_rate * b[edge_src_indices[i]];

    // Update momentum scalar via FISTA formula
    // tk' = (1 + sqrt(1 + 4 * tk^2)) / 2
    double tknew = (1.0 + std::sqrt(1.0 + 4.0 * tk * tk)) / 2.0;

    // Proximal / projection step to satisfy x_uv + x_vu = 1
    // new_xuv[i] = clamp((z[i] - z[reverse_edge] + 1)/2, [0,1])
#if PARALLEL
#pragma omp parallel for
#endif
    for (int i = 0; i < 2 * m; ++i)
      new_xuv[i] = std::clamp((z[i] - z[reverse_edge_idx[i]] + 1.0) / 2.0, 0.0, 1.0);

    // FISTA momentum update: new_y = new_x + ((tk-1)/tknew)*(new_x - x_old) + (tk/tknew)*(new_x - y_old)
#if PARALLEL
#pragma omp parallel for
#endif
    for (int i = 0; i < 2 * m; ++i)
      new_y[i] = new_xuv[i] + ((tk - 1.0) / tknew) * (new_xuv[i] - x[i]) + (tk / tknew) * (new_xuv[i] - y[i]);

    // Copy updates back into x and y
#if PARALLEL
#pragma omp parallel for
#endif
    for (int i = 0; i < 2 * m; ++i)
    {
      x[i] = new_xuv[i];
      y[i] = new_y[i];
    }

    // Update the momentum scalar tk for the next iteration
    tk = tknew;

    // Calculate and store stats
    std::fill(b.begin(), b.end(), 0.0);
    for (int i = 0; i < 2 * m; ++i)
      b[edge_src_indices[i]] += x[i];
  }

  // Clean up allocated memory
  delete[] reverse_edge_idx;
  delete[] edge_src_indices;
  delete[] new_xuv;
  delete[] new_y;

  vector<double> b_to_return = b; // Copy b before it's modified by peeling
  vector<bool> peeling_result = fractional_peeling(Adj, b, x, n, m);
  return {peeling_result, b_to_return};
}

char get_delimiter(string filepath)
{
  /*
  Function to detect the delimiter used in the edgelist file.
  Arguments:
  - filepath: The path to the edgelist file.
  Returns:
  - The character used as a delimiter in the edgelist file.
  */
  // Open the edgelist file
  ifstream edgelist(filepath);
  // Placeholder for the first line
  string line;
  // Read the first line of the edgelist file
  getline(edgelist, line);
  // Check for common delimiters (comma, tab, space) in the line
  // and return the appropriate delimiter character
  if (line.find(',') != string::npos)
  {
    return ',';
  }
  else if (line.find('\t') != string::npos)
  {
    return '\t';
  }
  else if (line.find(' ') != string::npos)
  {
    return ' ';
  }
  // If no known delimiter is found, throw an error
  throw invalid_argument("Could not detect filetype for " + filepath);
}

map<int, string> reverse_node_mapping(map<string, int> original_to_integer_map)
{
  /*
  Function to create a mapping from original node IDs (as strings) to integer IDs.
  Arguments:
  - original_to_integer_map: A map where keys are original node IDs (as strings) and values are integer IDs.
  Returns:
  - A map where keys are integer IDs and values are original node IDs (as strings).
  */
  // Create a mapping from integer IDs back to original node IDs
  map<int, string> integer_to_original_map;
  // Iterate through the original to integer mapping
  for (auto const &[original_node_id, integer_node_id] : original_to_integer_map)
  {
    // For each original node ID, map the integer ID back to the original node ID
    integer_to_original_map[integer_node_id] = original_node_id;
  }
  // Return the mapping from integer IDs to original node IDs
  return integer_to_original_map;
}

map<string, int> get_node_mapping(string filepath, char delimiter)
{
  /*
  Function to create a mapping from original node IDs (as strings) to integer IDs.
  Arguments:
  - filepath: The path to the edgelist file containing the network data.
  - delimiter: The character used to separate node IDs in the edgelist file.
  Returns:
  - A map where keys are original node IDs (as strings) and values are integer IDs.
  */
  // Create a mapping from original node IDs (as strings) to integer IDs
  map<string, int> original_to_integer_map;
  // Open the edgelist file
  ifstream edgelist(filepath);
  // Create a placeholder for each line in the edgelist
  string line;
  // Initialize a counter for new node IDs
  int current_new_node_id = 0;
  // Iterate through each line in the edgelist
  while (getline(edgelist, line))
  {
    // Create a stringstream to split the line by the delimiter
    stringstream ss(line);
    // Create a placeholder for the original node ID currently being processed
    string current_original_node_id;
    // Iterate through each node ID in the line (split by the delimiter)
    while (getline(ss, current_original_node_id, delimiter))
    {
      // Check if the current original node ID is already in the map
      if (original_to_integer_map.find(current_original_node_id) == original_to_integer_map.end())
      {
        // If not, add it to the map with the current new node ID
        original_to_integer_map[current_original_node_id] = current_new_node_id;
        // Increment the new node ID counter for the next unique node
        current_new_node_id++;
      }
    }
  }
  // Return the mapping from original node IDs to integer IDs
  return original_to_integer_map;
}

vector<pair<int, int>> read_network_edgelist(string filepath, char delimiter, map<string, int> original_to_integer_map)
{
  /*
  Function to read the network edgelist from a file and create a vector of edges.
  Arguments:
  - filepath: The path to the edgelist file.
  - delimiter: The character used to separate node IDs in the edgelist file.
  - original_to_integer_map: A map where keys are original node IDs (as strings) and values are integer IDs.
  Returns:
  - A vector of pairs representing the edges in the network.
  */
  // Create a vector to hold the edges of the network
  vector<pair<int, int>> vector_edgelist;
  // Open the edgelist file
  ifstream edgelist(filepath);
  // Create a placeholder for each line in the edgelist
  string line;
  // Iterate through each line in the edgelist
  while (getline(edgelist, line))
  {
    // Create a stringstream to split the line by the delimiter
    stringstream ss(line);
    // Create a placeholder for the current node ID being processed
    string current_node;
    // Create a vector to hold the current nodes in the line
    vector<string> current_nodes;
    // Iterate through each node ID in the line (split by the delimiter)
    while (getline(ss, current_node, delimiter))
    {
      // Add the current node ID to the vector of current nodes
      current_nodes.push_back(current_node);
    }
    // Add the edge to the vector of edges using the integer IDs from the mapping
    vector_edgelist.push_back({original_to_integer_map[current_nodes[0]], original_to_integer_map[current_nodes[1]]});
  }
  // Return the vector of edges representing the network
  return vector_edgelist;
}

template <typename T>
void write_vprop_vector(
    string &prop_vec,
    long long num_nodes,
    map<int, string> &integer_to_original_map,
    vector<T> &b,
    char delimiter = '\t')
{
  ofstream ostream(prop_vec);
  for (int i = 0; i < num_nodes; i++)
  {
    ostream << integer_to_original_map[i] << delimiter << b[i] << '\n';
  }
}

map<int, int> get_subgraph_mapping(const vector<bool> &nodes_in_subgraph, int n, vector<int> &subgraph_to_original_map)
{
  map<int, int> original_to_subgraph_map;
  int current_idx = 0;
  subgraph_to_original_map.clear();
  for (int i = 0; i < n; ++i)
  {
    if (nodes_in_subgraph[i])
    {
      original_to_subgraph_map[i] = current_idx;
      subgraph_to_original_map.push_back(i);
      current_idx++;
    }
  }
  return original_to_subgraph_map;
}

vector<vector<pair<int, int>>> get_subgraph_adj(
    const vector<vector<pair<int, int>>> &Adj,
    const map<int, int> &original_to_subgraph_map,
    long long &subgraph_m)
{
  int subgraph_n = original_to_subgraph_map.size();
  vector<vector<pair<int, int>>> subgraph_adj(subgraph_n);
  subgraph_m = 0;
  int e_idx = 0;

  for (const auto &pair : original_to_subgraph_map)
  {
    int u_original = pair.first;
    int u_subgraph = pair.second;
    for (const auto &edge : Adj[u_original])
    {
      int v_original = edge.first;
      // Check if neighbor is also in the subgraph
      auto it = original_to_subgraph_map.find(v_original);
      if (it != original_to_subgraph_map.end())
      {
        int v_subgraph = it->second;
        if (u_original < v_original)
        { // Process each undirected edge once
          subgraph_adj[u_subgraph].push_back({v_subgraph, e_idx});
          subgraph_adj[v_subgraph].push_back({u_subgraph, e_idx + 1});
          e_idx += 2;
          subgraph_m++;
        }
      }
    }
  }
  return subgraph_adj;
}

int main(int argc, char **argv)
{
  // Disable synchronization with C-style I/O
  ios_base::sync_with_stdio(0);
  // Untie cin from cout
  cin.tie(0);

  auto start_total = chrono::high_resolution_clock::now();

  // Start timing the input reading process
  auto start = std::chrono::high_resolution_clock::now();

  // Get command line arguments
  // Get the number of iterations
  int iters = atoi(argv[1]);
  // Get the file path to the input network edgelist
  string network_filepath = argv[2];
  // Get the file path for outputting the clustering results
  string output_filepath = argv[3];
  // Get the file path for outputting the density results
  string density_filepath = argv[4];

  // Process the input network edgelist
  // Determine the delimiter used in the edgelist file (comma, tab, or space)
  char delimiter = get_delimiter(network_filepath);
  // Read the network edgelist and create mappings from original node IDs to integer IDs
  map<string, int> original_to_integer_map = get_node_mapping(network_filepath, delimiter);
  // Reverse the mapping to get integer IDs back to original node IDs
  map<int, string> integer_to_original_map = reverse_node_mapping(original_to_integer_map);
  // Read the network edgelist into a vector of pairs (edges)
  vector<pair<int, int>> vector_edgelist = read_network_edgelist(network_filepath, delimiter, original_to_integer_map);

  // Compute the number of nodes
  long long num_nodes = original_to_integer_map.size();
  // Compute the number of edges
  long long num_edges = vector_edgelist.size();
  // Build the adjacency list, where each node has a list of
  // its neighbors and the corresponding edge indices
  vector<vector<pair<int, int>>> Adj(num_nodes);
  // Edge index counter
  int e_idx = 0;
  // Iterate through the edges and populate the adjacency list
  for (int e = 0; e < num_edges; ++e)
  {
    // Get the nodes connected by the edge
    int i = vector_edgelist[e].first;
    int j = vector_edgelist[e].second;
    // Add the edge to the adjacency list for both nodes
    // Each edge is represented by two entries in the adjacency list
    // (one for each direction, with the edge index)
    Adj[i].push_back({j, e_idx});
    Adj[j].push_back({i, e_idx + 1});
    // Increment the edge index by 2 (since each edge is represented twice)
    e_idx += 2;
  }

  // End timing the input reading process
  auto end = std::chrono::high_resolution_clock::now();
  // Output the time taken to read the input
  cout << "[TIME] Reading input: " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

  // Keep track of remaining nodes
  vector<bool> remaining_nodes(num_nodes, true);
  long long remaining_node_count = num_nodes;

  // Initialize cluster assignments and final b values
  vector<long long> cluster_assignments(num_nodes, -1);
  vector<double> final_b_values(num_nodes, 0.0);

  // Initialize the subgraph ID counter
  int subgraph_id_counter = 0;

  while (remaining_node_count > 0)
  {
    // First, find and process any newly isolated vertices
    vector<int> isolated_nodes;
    for (int i = 0; i < num_nodes; ++i) {
        if (remaining_nodes[i]) {
            bool is_isolated = true;
            for (const auto& edge : Adj[i]) {
                if (remaining_nodes[edge.first]) {
                    is_isolated = false;
                    break;
                }
            }
            if (is_isolated) {
                isolated_nodes.push_back(i);
            }
        }
    }

    if (!isolated_nodes.empty()) {
        cout << "\nFound " << isolated_nodes.size() << " isolated vertices. Assigning them to singleton clusters." << endl;
        for (int isolated_node_idx : isolated_nodes) {
            cluster_assignments[isolated_node_idx] = subgraph_id_counter++;
            remaining_nodes[isolated_node_idx] = false;
            remaining_node_count--;
        }
    }

    if(remaining_node_count == 0) {
        break;
    }

    cout << "\n--- Finding Subgraph " << subgraph_id_counter << " ---" << endl;

    vector<int> subgraph_to_original_map;
    map<int, int> original_to_subgraph_map = get_subgraph_mapping(remaining_nodes, num_nodes, subgraph_to_original_map);

    long long subgraph_m;
    vector<vector<pair<int, int>>> subgraph_adj = get_subgraph_adj(Adj, original_to_subgraph_map, subgraph_m);
    long long subgraph_n = original_to_subgraph_map.size();

    cout << "Considering subgraph of " << subgraph_n << " nodes and " << subgraph_m << " edges." << endl;

    // If no nodes are left in the subgraph, break the loop
    if (subgraph_n == 0) {
        break;
    }

    // If no edges are left, assign all remaining nodes to their own singleton clusters.
    if (subgraph_m == 0) {
        for (int original_node_idx : subgraph_to_original_map) {
            cluster_assignments[original_node_idx] = subgraph_id_counter++;
        }
        break;
    }

    auto start_fista = chrono::high_resolution_clock::now();

    pair<vector<bool>, vector<double>> fista_result = fista_frac(subgraph_adj, iters, subgraph_n, subgraph_m);
    vector<bool> densest_subgraph_in_current = fista_result.first;
    vector<double> b_subgraph = fista_result.second;

    auto end_fista = chrono::high_resolution_clock::now();
    cout << "[TIME] FISTA on subgraph: " << chrono::duration_cast<chrono::milliseconds>(end_fista - start_fista).count() << " ms" << endl;

    for (int j = 0; j < subgraph_n; ++j)
    {
      int original_node_idx = subgraph_to_original_map[j];
      final_b_values[original_node_idx] = b_subgraph[j];
    }

    int nodes_in_densest = 0;
    for (int j = 0; j < subgraph_n; ++j)
    {
      if (densest_subgraph_in_current[j])
      {
        int original_node_idx = subgraph_to_original_map[j];
        if (remaining_nodes[original_node_idx])
        {
          remaining_nodes[original_node_idx] = false;
          cluster_assignments[original_node_idx] = subgraph_id_counter;
          nodes_in_densest++;
        }
      }
    }

    remaining_node_count -= nodes_in_densest;
    cout << "Found and removed " << nodes_in_densest << " nodes. " << remaining_node_count << " nodes remaining." << endl;
    subgraph_id_counter++;
  }

  auto start_write = chrono::high_resolution_clock::now();
  write_vprop_vector<long long>(output_filepath, cluster_assignments.size(), integer_to_original_map, cluster_assignments, '\t');
  cout << "\n[TIME] Writing final assignments: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write).count() << " ms" << endl;

  auto start_write_density = chrono::high_resolution_clock::now();
  write_vprop_vector<double>(density_filepath, final_b_values.size(), integer_to_original_map, final_b_values, '\t');
  cout << "[TIME] Writing final density vector: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write_density).count() << " ms" << endl;

  auto end_total = chrono::high_resolution_clock::now();
  cout << "[TIME] Total execution: " << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() << " ms" << endl;

  return 0;
}