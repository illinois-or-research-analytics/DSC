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
#include <utility>
#include <cstring>

#if PARALLEL
#include <omp.h>
#endif

using namespace std;

vector<double> fista(vector<vector<pair<int, int>>> &Adj,
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
  double *x = new double[2 * m];
  double *y = new double[2 * m];
  double *z = new double[2 * m];

  // Load vector b initialized to 0
  vector<double> b(n, 0);

  // Initialize x and y to 0.5 for all directed edges
#if PARALLEL
#pragma omp parallel for
#endif
  for (int i = 0; i < 2 * m; ++i)
  {
    x[i] = 0.5;
    y[i] = 0.5;
  }

  // Initialize z to 0
  std::memset(z, 0, 2 * m * sizeof(double));

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
  delete[] x;
  delete[] y;
  delete[] z;
  delete[] new_xuv;
  delete[] new_y;

  // Return the final load vector b
  return b;
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

int main(int argc, char **argv)
{
  // Disable synchronization with C-style I/O
  ios_base::sync_with_stdio(0);
  // Untie cin from cout
  cin.tie(0);

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
  string output_density = argv[4];

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

  // Start timing the FISTA algorithm execution
  start = std::chrono::high_resolution_clock::now();

  // Run the FISTA algorithm with the given parameters to get the load vector b
  vector<double> b = fista(Adj, iters, num_nodes, num_edges);

  // End timing the FISTA algorithm execution
  end = std::chrono::high_resolution_clock::now();
  // Output the time taken to complete the FISTA step
  cout << "[TIME] FISTA: " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

  // == Density ==

  // Start timing the density writing process
  start = std::chrono::high_resolution_clock::now();

  // Write the load vector results to the specified output file
  write_vprop_vector<double>(output_density, num_nodes, integer_to_original_map, b);

  // End timing the density writing process
  end = std::chrono::high_resolution_clock::now();
  // Output the time taken to write the density results
  cout << "[TIME] Writing density: " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

  // == Clustering ==

  // Start timing the clustering writing process
  start = std::chrono::high_resolution_clock::now();

  // Create a clustering vector to hold the cluster IDs for each node
  vector<long long> clustering(num_nodes);
  // Iterate through each node to assign it to a cluster based on the load vector b
  for (int i = 0; i < num_nodes; ++i)
  {
    // Assign each node to a cluster based on the load vector b
    // The cluster ID is determined by rounding the load value to the nearest integer
    clustering[i] = static_cast<long long>(round(b[i]));
  }
  // Write the clustering results to the specified output file
  write_vprop_vector<long long>(output_filepath, num_nodes, integer_to_original_map, clustering);

  // End timing the clustering writing process
  end = std::chrono::high_resolution_clock::now();
  // Output the time taken to write the clustering results
  cout << "[TIME] Writing clustering: " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
}