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
#include <queue> // Added for BFS
#include <iomanip>

#if PARALLEL
#include <omp.h>
#endif

using namespace std;

// --- TYPE DEFINITIONS FOR LARGE NETWORKS ---
using NodeID = long long;
using EdgeID = long long;

// --- FISTA ALGORITHM ---
vector<double> fista(const vector<vector<pair<NodeID, EdgeID>>> &Adj,
                     long long iterations,
                     NodeID n,
                     EdgeID m)
{
  /*
  FISTA algorithm implementation for network optimization.
  */

  // Create a mapping from directed edge index to its reverse
  // Using vectors for safer memory management with large types
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

  // Allocate FISTA variables
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

  return b;
}

// --- HELPER: Find connected components within a subset of nodes ---
vector<vector<NodeID>> get_components_in_subset(
    NodeID n,
    const vector<vector<pair<NodeID, EdgeID>>> &adj,
    const vector<NodeID> &subset_nodes)
{
  vector<bool> in_subset(n, false);
  for (NodeID u : subset_nodes)
  {
    in_subset[u] = true;
  }

  vector<bool> visited(n, false);
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

        for (const auto &edge : adj[u])
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

int main(int argc, char **argv)
{
  ios_base::sync_with_stdio(0);
  cin.tie(0);

  auto start = std::chrono::high_resolution_clock::now();

  if (argc < 5)
  {
    cerr << "Usage: " << argv[0] << " <iters> <network_file> <output_clusters> <output_density>" << endl;
    return 1;
  }

  int iters = atoi(argv[1]);
  string network_filepath = argv[2];
  string output_filepath = argv[3];
  string output_density = argv[4];

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

  start = std::chrono::high_resolution_clock::now();

  // Run FISTA
  vector<double> b = fista(Adj, iters, num_nodes, num_edges);

  end = std::chrono::high_resolution_clock::now();
  cout << "[TIME] FISTA: " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

  // == Density Output ==
  start = std::chrono::high_resolution_clock::now();
  write_density_values(output_density, num_nodes, integer_to_original_map, b);
  end = std::chrono::high_resolution_clock::now();
  cout << "[TIME] Writing density: " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

  // == Clustering Processing & Output ==
  start = std::chrono::high_resolution_clock::now();

  vector<long long> cluster_assignments(num_nodes, -1); // Initialize all as noise (-1)

  // 1. Group nodes by rounded value
  map<long long, vector<NodeID>> groups;
  for (NodeID i = 0; i < num_nodes; ++i)
  {
    long long rounded_val = static_cast<long long>(round(b[i]));
    groups[rounded_val].push_back(i);
  }

  long long final_cluster_id = 0;
  long long total_clustered_nodes = 0;

  // 2. Process each group to find connected components
  for (auto const &[val, nodes_in_group] : groups)
  {
    if (nodes_in_group.empty())
      continue;

    // Find connected components within this group
    vector<vector<NodeID>> components = get_components_in_subset(num_nodes, Adj, nodes_in_group);

    for (const auto &comp : components)
    {
      // Filter out singleton clusters (size == 1)
      if (comp.size() > 1)
      {
        for (NodeID u : comp)
        {
          cluster_assignments[u] = final_cluster_id;
        }
        final_cluster_id++;
        total_clustered_nodes += comp.size();
      }
    }
  }

  // 3. Write Clusters using the specific function
  write_cluster_assignments(output_filepath, num_nodes, integer_to_original_map, cluster_assignments);

  end = std::chrono::high_resolution_clock::now();
  cout << "[TIME] Writing clustering: " << std::chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
  cout << "Final: Written " << final_cluster_id << " clusters containing " << total_clustered_nodes << " nodes." << endl;

  return 0;
}