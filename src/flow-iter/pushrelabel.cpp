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
#include <queue> // Added for BFS
#include "pushrelabel.h"

using namespace std;

// --- TYPE DEFINITIONS FOR LARGE NETWORKS ---
using NodeID = long long;
using EdgeID = long long;

const int INF = (int)1e9;

// --- HELPER: Find connected components within the extracted densest subset ---
vector<vector<NodeID>> get_components_in_subgraph(
    NodeID n,
    const vector<pair<NodeID, NodeID>> &edges,
    const vector<char> &subset_mask)
{
  // Build adjacency list restricted to the subset nodes
  // Note: We only build this temporarily for the component search
  vector<vector<NodeID>> adj(n);
  for (const auto &e : edges)
  {
    if (subset_mask[e.first] && subset_mask[e.second])
    {
      adj[e.first].push_back(e.second);
      adj[e.second].push_back(e.first);
    }
  }

  vector<bool> visited(n, false);
  vector<vector<NodeID>> components;

  for (NodeID i = 0; i < n; ++i)
  {
    // Only process nodes that are part of the densest subset
    if (subset_mask[i] && !visited[i])
    {
      vector<NodeID> component;
      queue<NodeID> q;

      visited[i] = true;
      q.push(i);
      component.push_back(i);

      while (!q.empty())
      {
        NodeID u = q.front();
        q.pop();

        for (NodeID v : adj[u])
        {
          if (!visited[v])
          {
            visited[v] = true;
            q.push(v);
            component.push_back(v);
          }
        }
      }
      components.push_back(component);
    }
  }
  return components;
}

pair<vector<char>, double> find_densest_subgraph(const vector<pair<NodeID, NodeID>> &edges,
                                                 NodeID n,
                                                 EdgeID m,
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

  long long max_flow_nodes_alloc = n + m + 2;
  long long max_flow_arcs_alloc = 2LL * n + 6LL * m + 100;

  vector<long long> deg(max_flow_nodes_alloc);
  vector<long long> cur(max_flow_nodes_alloc);
  vector<node> nodes(max_flow_nodes_alloc + 1);
  vector<arc> arcs(max_flow_arcs_alloc);
  vector<cType> cap(max_flow_arcs_alloc);

  node *nodes_ptr = nodes.data();
  ::sentinelNode = nodes_ptr + max_flow_nodes_alloc;

  for (int iter = 0; iter < max_iter; ++iter)
  {
    prev_density = density;

    vector<long long> old2new(n, -1);
    vector<long long> new2old;
    new2old.reserve(n);
    long long n1 = 0;
    for (long long u = 0; u < n; ++u)
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

    vector<pair<long long, long long>> edges1;
    edges1.reserve(m);
    for (const auto &e : edges)
    {
      long long u_new = old2new[e.first];
      long long v_new = old2new[e.second];
      if (u_new != -1 && v_new != -1)
      {
        edges1.emplace_back(u_new, v_new);
      }
    }

    long long m1 = edges1.size();
    long long SRC = n1 + m1, SNK = SRC + 1, NND = SNK + 1;

    fill(deg.begin(), deg.begin() + NND, 0L);
    for (long long i = 0; i < n1; ++i)
      deg[i] = 1;
    for (const auto &e1 : edges1)
    {
      deg[e1.first]++;
      deg[e1.second]++;
    }
    for (long long j = 0; j < m1; ++j)
      deg[n1 + j] = 3;
    deg[SRC] = n1;
    deg[SNK] = m1;

    for (long long i = 1; i < NND; ++i)
      deg[i] += deg[i - 1];
    long long tot_arcs = (NND > 0) ? deg[NND - 1] : 0;

    if (tot_arcs > max_flow_arcs_alloc)
    {
      cerr << "Error: Total arcs required (" << tot_arcs << ") exceeds allocation (" << max_flow_arcs_alloc << ")." << endl;
      return {subg, density};
    }

    if (NND > 0)
      cur[0] = 0;
    for (long long i = 1; i < NND; ++i)
      cur[i] = deg[i - 1];
    for (long long i = 0; i < NND; ++i)
      nodes_ptr[i].first = arcs.data() + cur[i];

    auto add_arc = [&](long long u_arc, long long v_arc, cType capacity_val)
    {
      long long pu = cur[u_arc]++;
      long long pv = cur[v_arc]++;

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

    for (long long u_new = 0; u_new < n1; ++u_new)
      add_arc(SRC, u_new, src_u_cap_val);
    for (long long j = 0; j < m1; ++j)
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
    long long src_arc_base = (NND > 0 && nodes_ptr[SRC].first) ? (nodes_ptr[SRC].first - arcs.data()) : -1;

    long long vcount = 0;
    if (src_arc_base != -1)
    {
      for (long long idx = 0; idx < n1; ++idx)
      {
        if (nodes_ptr[idx].d < NND && cap[src_arc_base + idx] > 0)
        {
          subg[new2old[idx]] = 1;
          vcount++;
        }
      }
    }

    long long ecount = 0;
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

inline char get_delimiter(string filepath)
{
  ifstream edgelist(filepath);
  string line;
  if (!getline(edgelist, line))
    throw runtime_error("Empty edgelist file.");
  if (line.find(',') != string::npos)
    return ',';
  else if (line.find('\t') != string::npos)
    return '\t';
  else if (line.find(' ') != string::npos)
    return ' ';
  throw invalid_argument("Could not detect filetype for " + filepath);
}

inline map<string, NodeID> get_node_mapping(string filepath, char delimiter)
{
  map<string, NodeID> original_to_integer_map;
  ifstream edgelist(filepath);
  string line;
  if (!getline(edgelist, line))
    throw runtime_error("Empty edgelist file.");
  NodeID current_new_node_id = 0;
  while (getline(edgelist, line))
  {
    if (line.empty())
      continue;
    stringstream ss(line);
    string current_original_node_id;
    while (getline(ss, current_original_node_id, delimiter))
    {
      if (original_to_integer_map.find(current_original_node_id) == original_to_integer_map.end())
      {
        original_to_integer_map[current_original_node_id] = current_new_node_id;
        current_new_node_id++;
      }
    }
  }
  return original_to_integer_map;
}

inline vector<pair<NodeID, NodeID>> read_network_edgelist(
    string filepath,
    char delimiter,
    const map<string, NodeID> &original_to_integer_map)
{
  vector<pair<NodeID, NodeID>> vector_edgelist;
  ifstream edgelist(filepath);
  string line;

  if (!getline(edgelist, line))
    throw runtime_error("Empty edgelist file: " + filepath);

  while (getline(edgelist, line))
  {
    if (line.empty())
      continue;

    stringstream ss(line);
    string current_node;
    vector<string> current_nodes;

    while (getline(ss, current_node, delimiter))
    {
      current_nodes.push_back(current_node);
    }

    if (current_nodes.size() >= 2)
    {
      vector_edgelist.push_back({original_to_integer_map.at(current_nodes[0]),
                                 original_to_integer_map.at(current_nodes[1])});
    }
  }
  return vector_edgelist;
}

inline map<NodeID, string> reverse_node_mapping(map<string, NodeID> original_to_integer_map)
{
  map<NodeID, string> integer_to_original_map;
  for (auto const &[original_node_id, integer_node_id] : original_to_integer_map)
  {
    integer_to_original_map[integer_node_id] = original_node_id;
  }
  return integer_to_original_map;
}

// --- OUTPUT FUNCTION FOR DENSITY ---
void write_density_values(
    string filepath,
    NodeID num_nodes,
    map<NodeID, string> &integer_to_original_map,
    vector<double> &density_values)
{
  ofstream ostream(filepath);
  ostream << "node_id,value\n"; // Write Header
  for (NodeID i = 0; i < num_nodes; i++)
  {
    ostream << integer_to_original_map[i] << ',' << density_values[i] << '\n';
  }
}

// --- OUTPUT FUNCTION FOR CLUSTERS ---
void write_cluster_assignments(
    string filepath,
    NodeID num_nodes,
    map<NodeID, string> &integer_to_original_map,
    vector<long long> &cluster_assignments)
{
  ofstream ostream(filepath);
  ostream << "node_id,cluster_id\n"; // Write Header
  for (NodeID i = 0; i < num_nodes; i++)
  {
    // Skip noise/singleton clusters (marked as -1)
    if (cluster_assignments[i] == -1)
      continue;

    ostream << integer_to_original_map[i] << ',' << cluster_assignments[i] << '\n';
  }
}

extern "C" int run_pushrelabel(int argc, char **argv)
{
  // Disable synchronization with C-style I/O
  ios_base::sync_with_stdio(0);
  // Untie cin from cout
  cin.tie(0);

  auto start_total = chrono::high_resolution_clock::now();

  if (argc < 7)
  {
    cerr << "Usage: " << argv[0] << " <ACCURACY> <maxIter> <graph_path> <output_path> <density_path> <density_threshold>" << endl;
    return 1;
  }

  int accuracy = atoi(argv[1]);
  int max_iter = atoi(argv[2]);
  string network_filepath = argv[3];
  string output_filepath = argv[4];
  string density_filepath = argv[5];
  double density_threshold = stod(argv[6]);

  char delimiter = get_delimiter(network_filepath);
  map<string, NodeID> original_to_integer_map = get_node_mapping(network_filepath, delimiter);
  map<NodeID, string> integer_to_original_map = reverse_node_mapping(original_to_integer_map);

  vector<pair<NodeID, NodeID>> all_edges = read_network_edgelist(network_filepath, delimiter, original_to_integer_map);

  NodeID n_total = original_to_integer_map.size();
  vector<bool> remaining_nodes(n_total, true);
  long long remaining_node_count = n_total;

  vector<long long> cluster_assignments(n_total, -1);
  vector<double> final_densities(n_total, 0.0);
  long long cluster_id_counter = 0;

  while (remaining_node_count > 0)
  {
    vector<NodeID> isolated_nodes;
    map<NodeID, vector<NodeID>> adj_list;

    // Build temp adjacency list for finding isolated nodes
    for (const auto &edge : all_edges)
    {
      if (remaining_nodes[edge.first] && remaining_nodes[edge.second])
      {
        adj_list[edge.first].push_back(edge.second);
        adj_list[edge.second].push_back(edge.first);
      }
    }

    for (NodeID i = 0; i < n_total; ++i)
    {
      if (remaining_nodes[i] && adj_list.find(i) == adj_list.end())
      {
        isolated_nodes.push_back(i);
      }
    }

    if (!isolated_nodes.empty())
    {
      cout << "Found and removing " << isolated_nodes.size() << " isolated vertices." << endl;
      for (NodeID node_idx : isolated_nodes)
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

    map<NodeID, NodeID> original_to_subgraph_map;
    vector<NodeID> subgraph_to_original_map;
    NodeID current_subgraph_idx = 0;

    for (NodeID i = 0; i < n_total; ++i)
    {
      if (remaining_nodes[i])
      {
        original_to_subgraph_map[i] = current_subgraph_idx;
        subgraph_to_original_map.push_back(i);
        current_subgraph_idx++;
      }
    }

    vector<pair<NodeID, NodeID>> subgraph_edges;
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
      for (NodeID node_idx : subgraph_to_original_map)
      {
        cluster_assignments[node_idx] = cluster_id_counter++;
        final_densities[node_idx] = 0.0;
      }
      break;
    }

    // Run push-relabel to find the densest subset
    pair<vector<char>, double> result = find_densest_subgraph(subgraph_edges, subgraph_n, subgraph_m, max_iter, accuracy);
    vector<char> densest_nodes_mask = result.first;
    double found_density = result.second;

    if (found_density < density_threshold)
    {
      cout << "Densest subset density " << found_density << " is below threshold " << density_threshold << ". Stopping iteration." << endl;
      break;
    }

    vector<vector<NodeID>> components = get_components_in_subgraph(subgraph_n, subgraph_edges, densest_nodes_mask);

    if (components.empty())
    {
      cout << "Could not extract a dense cluster (empty). Halting." << endl;
      break;
    }

    cout << "Densest subset density: " << found_density << ". Decomposed into " << components.size() << " component(s)." << endl;

    NodeID nodes_removed_this_step = 0;
    for (const auto &comp : components)
    {
      for (NodeID sub_u : comp)
      {
        NodeID original_node_idx = subgraph_to_original_map[sub_u];
        if (remaining_nodes[original_node_idx])
        {
          cluster_assignments[original_node_idx] = cluster_id_counter;
          final_densities[original_node_idx] = found_density;
          remaining_nodes[original_node_idx] = false;
          nodes_removed_this_step++;
        }
      }
      cluster_id_counter++;
    }
    remaining_node_count -= nodes_removed_this_step;
    cout << "Removed " << nodes_removed_this_step << " nodes. " << remaining_node_count << " nodes remaining." << endl;
  }

  // --- POST-PROCESSING: Filter Singletons ---
  map<long long, int> cluster_sizes;
  for (NodeID i = 0; i < n_total; ++i)
  {
    if (cluster_assignments[i] != -1)
    {
      cluster_sizes[cluster_assignments[i]]++;
    }
  }

  long long singleton_count = 0;
  for (NodeID i = 0; i < n_total; i++)
  {
    long long cid = cluster_assignments[i];
    // Check if it's a valid cluster and has size 1
    if (cid != -1 && cluster_sizes[cid] == 1)
    {
      cluster_assignments[i] = -1; // Mark as noise
      singleton_count++;
    }
  }
  cout << "\nRemoved " << singleton_count << " singleton clusters (set to -1) before writing." << endl;

  // --- OUTPUT WRITING ---
  auto start_write = chrono::high_resolution_clock::now();
  write_cluster_assignments(output_filepath, n_total, integer_to_original_map, cluster_assignments);
  cout << "[TIME] Writing final assignments: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write).count() << " ms" << endl;

  auto start_write_density = chrono::high_resolution_clock::now();
  write_density_values(density_filepath, n_total, integer_to_original_map, final_densities);
  cout << "[TIME] Writing final density vector: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write_density).count() << " ms" << endl;

  auto end_total = chrono::high_resolution_clock::now();
  cout << "[TIME] Total execution: " << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() << " ms" << endl;

  return 0;
}