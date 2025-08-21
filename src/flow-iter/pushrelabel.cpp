// High-performance push-relabel DSP solver
extern "C"
{
#include "external/exactDSP-cpp/hi_pr.h"
}

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "pushrelabel.h"

using namespace std;
const int INF = (int)1e9;

pair<vector<char>, double> find_densest_subgraph(const vector<pair<int, int>> &edges,
                                                 long long n,
                                                 long long m,
                                                 int max_iter,
                                                 int accuracy)
{
  if (n == 0)
  {
    return {{}, 0.0};
  }

  vector<char> subg(n, 1);
  double density = (n > 0) ? (1.0 * m / n) : 0.0;
  double prev_density = -1.0;

  int max_flow_nodes_alloc = n + m + 2;
  long max_flow_arcs_alloc = 2L * n + 6L * m + 100;

  vector<long> deg(max_flow_nodes_alloc);
  vector<long> cur(max_flow_nodes_alloc);
  vector<node> nodes(max_flow_nodes_alloc + 1);
  vector<arc> arcs(max_flow_arcs_alloc);
  vector<cType> cap(max_flow_arcs_alloc);

  node *nodes_ptr = nodes.data();
  ::sentinelNode = nodes_ptr + max_flow_nodes_alloc;

  for (int iter = 0; iter < max_iter; ++iter)
  {
    prev_density = density;

    vector<int> old2new(n, -1);
    vector<int> new2old;
    new2old.reserve(n);
    int n1 = 0;
    for (int u = 0; u < n; ++u)
    {
      if (subg[u])
      {
        old2new[u] = n1;
        n1++;
        new2old.push_back(u);
      }
    }

    if (n1 == 0)
    {
      density = 0.0;
      if (abs(density - prev_density) < 1e-12)
        break;
      continue;
    }

    vector<pair<int, int>> edges1;
    edges1.reserve(m);
    for (const auto &e : edges)
    {
      int u_new = old2new[e.first];
      int v_new = old2new[e.second];
      if (u_new != -1 && v_new != -1)
      {
        edges1.emplace_back(u_new, v_new);
      }
    }

    int m1 = edges1.size();
    int SRC = n1 + m1, SNK = SRC + 1, NND = SNK + 1;

    fill(deg.begin(), deg.begin() + NND, 0L);
    for (int i = 0; i < n1; ++i)
      deg[i] = 1;
    for (const auto &e1 : edges1)
    {
      deg[e1.first]++;
      deg[e1.second]++;
    }
    for (int j = 0; j < m1; ++j)
      deg[n1 + j] = 3;
    deg[SRC] = n1;
    deg[SNK] = m1;

    for (int i = 1; i < NND; ++i)
      deg[i] += deg[i - 1];
    long tot_arcs = (NND > 0) ? deg[NND - 1] : 0;

    if (tot_arcs > max_flow_arcs_alloc)
    {
      cerr << "Error: Total arcs required (" << tot_arcs << ") exceeds allocation (" << max_flow_arcs_alloc << ")." << endl;
      return {subg, density};
    }

    if (NND > 0)
      cur[0] = 0;
    for (int i = 1; i < NND; ++i)
      cur[i] = deg[i - 1];
    for (int i = 0; i < NND; ++i)
      nodes_ptr[i].first = arcs.data() + cur[i];

    auto add_arc = [&](int u_arc, int v_arc, cType capacity_val)
    {
      long pu = cur[u_arc]++;
      long pv = cur[v_arc]++;

      arcs[pu].head = &nodes_ptr[v_arc];
      arcs[pu].rev = &arcs[pv];
      cap[pu] = capacity_val;

      arcs[pv].head = &nodes_ptr[u_arc];
      arcs[pv].rev = &arcs[pu];
      cap[pv] = 0;
    };

    cType src_u_cap_val = static_cast<cType>(density * accuracy);
    if (src_u_cap_val < 0)
      src_u_cap_val = 0;

    for (int u_new = 0; u_new < n1; ++u_new)
      add_arc(SRC, u_new, src_u_cap_val);
    for (int j = 0; j < m1; ++j)
    {
      add_arc(edges1[j].first, n1 + j, INF);
      add_arc(edges1[j].second, n1 + j, INF);
      add_arc(n1 + j, SNK, static_cast<cType>(accuracy));
    }

    if (NND > 0 && tot_arcs > 0 && SRC < NND && SNK < NND && SRC != SNK)
    {
      min_cut(NND, tot_arcs / 2, nodes_ptr, arcs.data(), cap.data(), &nodes_ptr[SRC], &nodes_ptr[SNK], 0);
    }

    fill(subg.begin(), subg.end(), (char)0);
    long src_arc_base = (NND > 0 && nodes_ptr[SRC].first) ? (nodes_ptr[SRC].first - arcs.data()) : -1;

    int vcount = 0;
    if (src_arc_base != -1)
    {
      for (int idx = 0; idx < n1; ++idx)
      {
        if (nodes_ptr[idx].d < NND && cap[src_arc_base + idx] > 0)
        {
          subg[new2old[idx]] = 1;
          vcount++;
        }
      }
    }

    int ecount = 0;
    for (const auto &e : edges)
    {
      if (subg[e.first] && subg[e.second])
      {
        ecount++;
      }
    }

    density = (vcount > 0) ? (static_cast<double>(ecount) / vcount) : 0.0;

    if (prev_density >= 0.0 && abs(density - prev_density) < 1e-12)
      break;
  }
  return {subg, density};
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

inline vector<pair<int, int>> read_network_edgelist(string filepath, char delimiter, const map<string, int> &original_to_integer_map)
{
  vector<pair<int, int>> vector_edgelist;
  ifstream edgelist(filepath);
  string line;
  while (getline(edgelist, line))
  {
    stringstream ss(line);
    string u_node_str, v_node_str;
    getline(ss, u_node_str, delimiter);
    getline(ss, v_node_str, delimiter);
    if (!u_node_str.empty() && !v_node_str.empty())
    {
      vector_edgelist.push_back({original_to_integer_map.at(u_node_str), original_to_integer_map.at(v_node_str)});
    }
  }
  return vector_edgelist;
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

extern "C" int run_pushrelabel(int argc, char **argv)
{
  // Disable synchronization with C-style I/O
  ios_base::sync_with_stdio(0);
  // Untie cin from cout
  cin.tie(0);

  auto start_total = chrono::high_resolution_clock::now();

  if (argc < 6)
  {
    cerr << "Usage: " << argv[0] << " <ACCURACY> <maxIter> <graph_path> <output_path> <density_path>" << endl;
    return 1;
  }

  // Get the accuracy parameter
  int accuracy = atoi(argv[1]);
  // Get the number of iterations
  int max_iter = atoi(argv[2]);
  // Get the file path to the input network edgelist
  string network_filepath = argv[3];
  // Get the file path for outputting the clustering results
  string output_filepath = argv[4];
  // Get the file path for outputting the density results
  string density_filepath = argv[5];

  // Process the input network edgelist
  // Determine the delimiter used in the edgelist file (comma, tab, or space)
  char delimiter = get_delimiter(network_filepath);
  // Read the network edgelist and create mappings from original node IDs to integer IDs
  map<string, int> original_to_integer_map = get_node_mapping(network_filepath, delimiter);
  // Reverse the mapping to get integer IDs back to original node IDs
  map<int, string> integer_to_original_map = reverse_node_mapping(original_to_integer_map);

  vector<pair<int, int>> all_edges = read_network_edgelist(network_filepath, delimiter, original_to_integer_map);

  long long n_total = original_to_integer_map.size();
  vector<bool> remaining_nodes(n_total, true);
  long long remaining_node_count = n_total;

  vector<long long> cluster_assignments(n_total, 0);
  vector<double> final_densities(n_total, 0.0);
  long long cluster_id_counter = 0;

  while (remaining_node_count > 0)
  {
    vector<int> isolated_nodes;
    map<int, vector<int>> adj_list;
    for (const auto &edge : all_edges)
    {
      if (remaining_nodes[edge.first] && remaining_nodes[edge.second])
      {
        adj_list[edge.first].push_back(edge.second);
        adj_list[edge.second].push_back(edge.first);
      }
    }
    for (int i = 0; i < n_total; ++i)
    {
      if (remaining_nodes[i] && adj_list.find(i) == adj_list.end())
      {
        isolated_nodes.push_back(i);
      }
    }

    if (!isolated_nodes.empty())
    {
      cout << "Found and removing " << isolated_nodes.size() << " isolated vertices." << endl;
      for (int node_idx : isolated_nodes)
      {
        if (remaining_nodes[node_idx])
        {
          cluster_assignments[node_idx] = cluster_id_counter++;
          final_densities[node_idx] = 0.0;
          remaining_nodes[node_idx] = false;
          remaining_node_count--;
        }
      }
    }

    if (remaining_node_count == 0)
      break;

    map<int, int> original_to_subgraph_map;
    vector<int> subgraph_to_original_map;
    int current_subgraph_idx = 0;
    for (int i = 0; i < n_total; ++i)
    {
      if (remaining_nodes[i])
      {
        original_to_subgraph_map[i] = current_subgraph_idx;
        subgraph_to_original_map.push_back(i);
        current_subgraph_idx++;
      }
    }

    vector<pair<int, int>> subgraph_edges;
    for (const auto &edge : all_edges)
    {
      if (remaining_nodes[edge.first] && remaining_nodes[edge.second])
      {
        subgraph_edges.push_back({original_to_subgraph_map[edge.first], original_to_subgraph_map[edge.second]});
      }
    }

    long long subgraph_n = subgraph_to_original_map.size();
    long long subgraph_m = subgraph_edges.size();

    cout << "\nProcessing subgraph of " << subgraph_n << " nodes and " << subgraph_m << " edges." << endl;

    if (subgraph_n == 0)
      break;
    if (subgraph_m == 0)
    {
      cout << "No edges remain. Assigning singletons." << endl;
      for (int node_idx : subgraph_to_original_map)
      {
        cluster_assignments[node_idx] = cluster_id_counter++;
        final_densities[node_idx] = 0.0;
      }
      break;
    }

    pair<vector<char>, double> result = find_densest_subgraph(subgraph_edges, subgraph_n, subgraph_m, max_iter, accuracy);
    vector<char> densest_nodes_mask = result.first;
    double found_density = result.second;

    int nodes_in_cluster = 0;
    for (int i = 0; i < subgraph_n; ++i)
    {
      if (densest_nodes_mask[i])
      {
        int original_node_idx = subgraph_to_original_map[i];
        if (remaining_nodes[original_node_idx])
        {
          cluster_assignments[original_node_idx] = cluster_id_counter;
          final_densities[original_node_idx] = found_density;
          remaining_nodes[original_node_idx] = false;
          remaining_node_count--;
          nodes_in_cluster++;
        }
      }
    }

    if (nodes_in_cluster == 0)
    {
      cout << "Could not extract a dense cluster. Halting." << endl;
      break;
    }

    cout << "Found cluster " << cluster_id_counter << " with " << nodes_in_cluster << " nodes and density " << found_density << endl;
    cluster_id_counter++;
  }

  auto start_write = chrono::high_resolution_clock::now();
  write_vprop_vector<long long>(output_filepath, cluster_assignments.size(), integer_to_original_map, cluster_assignments, '\t');
  cout << "\n[TIME] Writing final assignments: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write).count() << " ms" << endl;

  auto start_write_density = chrono::high_resolution_clock::now();
  write_vprop_vector<double>(density_filepath, final_densities.size(), integer_to_original_map, final_densities, '\t');
  cout << "[TIME] Writing final density vector: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write_density).count() << " ms" << endl;

  auto end_total = chrono::high_resolution_clock::now();
  cout << "[TIME] Total execution: " << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() << " ms" << endl;

  return 0;
}