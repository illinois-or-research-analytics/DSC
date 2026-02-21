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
#include <iomanip>

#if PARALLEL
#include <omp.h>
#endif

using namespace std;

// --- TYPE DEFINITIONS ---
using NodeID = long long;
using EdgeID = long long;

// --- HELPER: Find connected components within a subset ---
vector<vector<NodeID>> get_components_in_subset(
    NodeID n_total,
    const vector<vector<pair<NodeID, EdgeID>>> &all_adj,
    const vector<NodeID> &subset_nodes)
{
  // Mask for fast lookup
  vector<bool> in_subset(n_total, false);
  for (NodeID u : subset_nodes)
    in_subset[u] = true;

  vector<bool> visited(n_total, false);
  vector<vector<NodeID>> components;

  for (NodeID root : subset_nodes)
  {
    if (!visited[root])
    {
      vector<NodeID> component;
      queue<NodeID> q;

      visited[root] = true;
      q.push(root);
      component.push_back(root);

      while (!q.empty())
      {
        NodeID u = q.front();
        q.pop();

        for (const auto &edge : all_adj[u])
        {
          NodeID v = edge.first;
          if (in_subset[v] && !visited[v])
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

vector<bool> fractional_peeling(
    const vector<vector<pair<NodeID, EdgeID>>> &Adj,
    vector<double> &b,
    const vector<double> &x,
    NodeID n,
    EdgeID m)
{
  /*
  Performs fractional peeling on the subgraph.
  */

  priority_queue<pair<double, NodeID>, vector<pair<double, NodeID>>, greater<pair<double, NodeID>>> pq;
  for (NodeID i = 0; i < n; ++i)
  {
    pq.push({b[i], i});
  }

  vector<bool> deleted(n, false);

  NodeID N = n;
  EdgeID M = m;

  double density = (N > 0) ? (double)(1.0 * M / N) : 0.0;

  vector<bool> current_subset(n, true);

  while (!pq.empty())
  {
    auto [d, i] = pq.top();
    pq.pop();

    if (deleted[i])
      continue;

    for (auto &[j, e_idx] : Adj[i])
    {
      EdgeID sister_idx = e_idx % 2 ? e_idx - 1 : e_idx + 1;

      if (!deleted[j])
      {
        b[j] -= x[sister_idx];
        pq.push({b[j], j});
        M--;
      }
    }

    deleted[i] = true;
    N--;

    if (N > 0)
    {
      if ((1.0 * M / N) > density)
      {
        density = 1.0 * M / N;
        for (NodeID k = 0; k < n; k++)
        {
          current_subset[k] = !deleted[k];
        }
      }
    }
  }
  return current_subset;
}

pair<vector<bool>, vector<double>> fista_frac(
    const vector<vector<pair<NodeID, EdgeID>>> &Adj,
    long long iterations,
    NodeID n,
    EdgeID m)
{
  vector<EdgeID> reverse_edge_idx(2 * m);
  vector<NodeID> edge_src_indices(2 * m);

#pragma omp parallel for
  for (NodeID i = 0; i < n; ++i)
  {
    for (size_t j = 0; j < Adj[i].size(); ++j)
    {
      EdgeID sister_idx = Adj[i][j].second;
      EdgeID idx = (sister_idx % 2 == 0 ? sister_idx + 1 : sister_idx - 1);

      edge_src_indices[idx] = i;
      reverse_edge_idx[idx] = sister_idx;
    }
  }

  double max_degree = 0;
  for (const auto &neighbors : Adj)
  {
    if (neighbors.size() > max_degree)
      max_degree = neighbors.size();
  }

  double learning_rate = (max_degree > 0) ? (0.5 / max_degree) : 0.0;

  vector<double> x(2 * m, 0.5);
  vector<double> y(2 * m, 0.5);
  vector<double> z(2 * m, 0.0);
  vector<double> b(n, 0);

  double tk = 1.0;

  vector<double> new_xuv(2 * m);
  vector<double> new_y(2 * m);

  for (int t = 0; t < iterations; ++t)
  {
    std::fill(b.begin(), b.end(), 0.0);
    for (EdgeID i = 0; i < 2 * m; ++i)
      b[edge_src_indices[i]] += y[i];

#if PARALLEL
#pragma omp parallel for
#endif
    for (EdgeID i = 0; i < 2 * m; ++i)
      z[i] = y[i] - 2.0 * learning_rate * b[edge_src_indices[i]];

    double tknew = (1.0 + std::sqrt(1.0 + 4.0 * tk * tk)) / 2.0;

#if PARALLEL
#pragma omp parallel for
#endif
    for (EdgeID i = 0; i < 2 * m; ++i)
      new_xuv[i] = std::clamp((z[i] - z[reverse_edge_idx[i]] + 1.0) / 2.0, 0.0, 1.0);

#if PARALLEL
#pragma omp parallel for
#endif
    for (EdgeID i = 0; i < 2 * m; ++i)
      new_y[i] = new_xuv[i] + ((tk - 1.0) / tknew) * (new_xuv[i] - x[i]) + (tk / tknew) * (new_xuv[i] - y[i]);

#if PARALLEL
#pragma omp parallel for
#endif
    for (EdgeID i = 0; i < 2 * m; ++i)
    {
      x[i] = new_xuv[i];
      y[i] = new_y[i];
    }

    tk = tknew;

    std::fill(b.begin(), b.end(), 0.0);
    for (EdgeID i = 0; i < 2 * m; ++i)
      b[edge_src_indices[i]] += x[i];
  }

  vector<double> b_to_return = b;
  vector<bool> peeling_result = fractional_peeling(Adj, b, x, n, m);
  return {peeling_result, b_to_return};
}

char get_delimiter(string filepath)
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

map<NodeID, string> reverse_node_mapping(map<string, NodeID> original_to_integer_map)
{
  map<NodeID, string> integer_to_original_map;
  for (auto const &[original_node_id, integer_node_id] : original_to_integer_map)
  {
    integer_to_original_map[integer_node_id] = original_node_id;
  }
  return integer_to_original_map;
}

map<string, NodeID> get_node_mapping(string filepath, char delimiter)
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

vector<pair<NodeID, NodeID>> read_network_edgelist(
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

map<NodeID, NodeID> get_subgraph_mapping(const vector<bool> &nodes_in_subgraph, NodeID n, vector<NodeID> &subgraph_to_original_map)
{
  map<NodeID, NodeID> original_to_subgraph_map;
  NodeID current_idx = 0;
  subgraph_to_original_map.clear();
  for (NodeID i = 0; i < n; ++i)
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

vector<vector<pair<NodeID, EdgeID>>> get_subgraph_adj(
    const vector<vector<pair<NodeID, EdgeID>>> &Adj,
    const map<NodeID, NodeID> &original_to_subgraph_map,
    EdgeID &subgraph_m)
{
  NodeID subgraph_n = original_to_subgraph_map.size();
  vector<vector<pair<NodeID, EdgeID>>> subgraph_adj(subgraph_n);
  subgraph_m = 0;
  EdgeID e_idx = 0;

  for (const auto &pair : original_to_subgraph_map)
  {
    NodeID u_original = pair.first;
    NodeID u_subgraph = pair.second;
    for (const auto &edge : Adj[u_original])
    {
      NodeID v_original = edge.first;
      auto it = original_to_subgraph_map.find(v_original);
      if (it != original_to_subgraph_map.end())
      {
        NodeID v_subgraph = it->second;
        if (u_original < v_original)
        {
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
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  auto start_total = chrono::high_resolution_clock::now();

  if (argc < 5)
  {
    cerr << "Usage: " << argv[0] << " <iters> <network_file> <output_clusters> <output_density>" << endl;
    return 1;
  }

  int iters = atoi(argv[1]);
  string network_filepath = argv[2];
  string output_filepath = argv[3];
  string density_filepath = argv[4];

  auto start = std::chrono::high_resolution_clock::now();

  char delimiter = get_delimiter(network_filepath);
  map<string, NodeID> original_to_integer_map = get_node_mapping(network_filepath, delimiter);
  map<NodeID, string> integer_to_original_map = reverse_node_mapping(original_to_integer_map);
  vector<pair<NodeID, NodeID>> vector_edgelist = read_network_edgelist(network_filepath, delimiter, original_to_integer_map);

  NodeID num_nodes = original_to_integer_map.size();
  EdgeID num_edges = vector_edgelist.size();

  vector<vector<pair<NodeID, EdgeID>>> Adj(num_nodes);
  EdgeID e_idx = 0;
  for (EdgeID e = 0; e < num_edges; ++e)
  {
    NodeID i = vector_edgelist[e].first;
    NodeID j = vector_edgelist[e].second;
    Adj[i].push_back({j, e_idx});
    Adj[j].push_back({i, e_idx + 1});
    e_idx += 2;
  }

  auto end = std::chrono::high_resolution_clock::now();
  cout << "[TIME] Reading input: " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

  vector<bool> remaining_nodes(num_nodes, true);
  NodeID remaining_node_count = num_nodes;

  vector<long long> cluster_assignments(num_nodes, -1);
  vector<double> final_b_values(num_nodes, 0.0);

  long long cluster_id_counter = 0;

  while (remaining_node_count > 0)
  {
    // 1. Identify and process isolated vertices (Singletons)
    // These are assigned a cluster ID but will be filtered out during output
    vector<NodeID> isolated_nodes;
    for (NodeID i = 0; i < num_nodes; ++i)
    {
      if (remaining_nodes[i])
      {
        bool is_isolated = true;
        for (const auto &edge : Adj[i])
        {
          if (remaining_nodes[edge.first])
          {
            is_isolated = false;
            break;
          }
        }
        if (is_isolated)
        {
          isolated_nodes.push_back(i);
        }
      }
    }

    if (!isolated_nodes.empty())
    {
      for (NodeID u : isolated_nodes)
      {
        cluster_assignments[u] = cluster_id_counter++;
        remaining_nodes[u] = false;
        remaining_node_count--;
      }
    }

    if (remaining_node_count == 0)
      break;

    // 2. Build Subgraph of remaining nodes
    vector<NodeID> subgraph_to_original_map;
    map<NodeID, NodeID> original_to_subgraph_map = get_subgraph_mapping(remaining_nodes, num_nodes, subgraph_to_original_map);

    EdgeID subgraph_m;
    vector<vector<pair<NodeID, EdgeID>>> subgraph_adj = get_subgraph_adj(Adj, original_to_subgraph_map, subgraph_m);
    NodeID subgraph_n = original_to_subgraph_map.size();

    if (subgraph_n == 0)
      break;

    if (subgraph_m == 0)
    {
      // If remaining nodes have no edges between them, they are singletons
      for (NodeID u : subgraph_to_original_map)
      {
        cluster_assignments[u] = cluster_id_counter++;
        remaining_nodes[u] = false;
        remaining_node_count--;
      }
      continue;
    }

    // 3. Run FISTA + Peeling on Subgraph
    auto start_fista = chrono::high_resolution_clock::now();
    pair<vector<bool>, vector<double>> fista_result = fista_frac(subgraph_adj, iters, subgraph_n, subgraph_m);
    vector<bool> densest_subgraph_mask = fista_result.first;
    vector<double> b_subgraph = fista_result.second;
    auto end_fista = chrono::high_resolution_clock::now();

    // Store density values
    for (NodeID j = 0; j < subgraph_n; ++j)
    {
      final_b_values[subgraph_to_original_map[j]] = b_subgraph[j];
    }

    // 4. Extract Nodes in the Densest Peeling Set (Global IDs)
    vector<NodeID> dense_subset_global;
    for (NodeID j = 0; j < subgraph_n; ++j)
    {
      if (densest_subgraph_mask[j])
      {
        dense_subset_global.push_back(subgraph_to_original_map[j]);
      }
    }

    // 5. Split this "Dense Subset" into Connected Components
    // Even if peeling finds a set, it might consist of disjoint components.
    // We treat each component as a distinct cluster.
    vector<vector<NodeID>> components = get_components_in_subset(num_nodes, Adj, dense_subset_global);

    NodeID nodes_removed_this_step = 0;

    for (const auto &comp : components)
    {
      // Assign one cluster ID per component
      for (NodeID u : comp)
      {
        cluster_assignments[u] = cluster_id_counter;
        if (remaining_nodes[u])
        {
          remaining_nodes[u] = false;
          nodes_removed_this_step++;
        }
      }
      cluster_id_counter++;
    }

    cout << "Iter: Found " << components.size() << " components ("
         << nodes_removed_this_step << " nodes) in "
         << chrono::duration_cast<chrono::milliseconds>(end_fista - start_fista).count() << " ms. "
         << remaining_node_count - nodes_removed_this_step << " remaining." << endl;

    remaining_node_count -= nodes_removed_this_step;
  }

  // --- OUTPUT WRITING ---

  // 1. Identify Singleton Clusters and Mark as -1 (Noise)
  map<long long, int> cluster_sizes;
  for (NodeID i = 0; i < num_nodes; ++i)
  {
    if (cluster_assignments[i] != -1)
    {
      cluster_sizes[cluster_assignments[i]]++;
    }
  }

  long long singleton_count = 0;
  for (NodeID i = 0; i < num_nodes; i++)
  {
    long long cid = cluster_assignments[i];
    if (cid != -1 && cluster_sizes[cid] == 1)
    {
      cluster_assignments[i] = -1; // Mark as noise
      singleton_count++;
    }
  }
  cout << "\nRemoved " << singleton_count << " singleton clusters (set to -1) before writing." << endl;

  // 2. Write Clusters using specialized function
  auto start_write = chrono::high_resolution_clock::now();
  write_cluster_assignments(output_filepath, num_nodes, integer_to_original_map, cluster_assignments);
  cout << "[TIME] Writing clusters: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write).count() << " ms" << endl;

  // 3. Write Density using specialized function
  auto start_write_density = chrono::high_resolution_clock::now();
  write_density_values(density_filepath, num_nodes, integer_to_original_map, final_b_values);
  cout << "[TIME] Writing density: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write_density).count() << " ms" << endl;

  auto end_total = chrono::high_resolution_clock::now();
  cout << "[TIME] Total execution: " << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() << " ms" << endl;

  return 0;
}