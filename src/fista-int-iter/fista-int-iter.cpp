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

// --- TYPE DEFINITIONS FOR LARGE NETWORKS ---
using NodeID = long long;
using EdgeID = long long;

// --- HELPER: Find connected components within the extracted densest subset ---
vector<vector<NodeID>> get_components_in_subgraph(
    NodeID n,
    const vector<vector<pair<NodeID, EdgeID>>> &adj,
    const vector<bool> &subset_mask)
{
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

                for (const auto &edge : adj[u])
                {
                    NodeID v = edge.first;
                    // Traverse only if neighbor is also in the subset
                    if (subset_mask[v] && !visited[v])
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

// --- EXTRACT BY ROUNDING ---
vector<bool> extract_by_rounding(
    const vector<double> &b,
    NodeID n)
{
    vector<bool> current_subset(n, false);
    if (n == 0)
        return current_subset;

    long long max_rounded_val = -1;

    // Pass 1: Find the maximum rounded integer value
    for (double val : b)
    {
        long long rounded = std::llround(val);
        if (rounded > max_rounded_val)
        {
            max_rounded_val = rounded;
        }
    }

    // Pass 2: Select nodes matching the max value
    for (NodeID i = 0; i < n; ++i)
    {
        if (std::llround(b[i]) == max_rounded_val)
        {
            current_subset[i] = true;
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
    // Using vectors instead of new/delete for safety with large types
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
    // Compute max degree
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
    vector<bool> result_subset = extract_by_rounding(b, n);

    return {result_subset, b_to_return};
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
    auto start = std::chrono::high_resolution_clock::now();

    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <iters> <network_file> <output_clusters> <output_density>" << endl;
        return 1;
    }

    int iters = atoi(argv[1]);
    string network_filepath = argv[2];
    string output_filepath = argv[3];
    string density_filepath = argv[4];

    char delimiter = get_delimiter(network_filepath);
    // Updated map to use NodeID (long long)
    map<string, NodeID> original_to_integer_map = get_node_mapping(network_filepath, delimiter);
    map<NodeID, string> integer_to_original_map = reverse_node_mapping(original_to_integer_map);
    vector<pair<NodeID, NodeID>> vector_edgelist = read_network_edgelist(network_filepath, delimiter, original_to_integer_map);

    NodeID num_nodes = original_to_integer_map.size();
    EdgeID num_edges = vector_edgelist.size();

    // Adjacency list with NodeID and EdgeID
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

    long long subgraph_id_counter = 0;

    while (remaining_node_count > 0)
    {
        // 1. Process isolated vertices
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
            cout << "\nFound " << isolated_nodes.size() << " isolated vertices. Assigning them to singleton clusters." << endl;
            for (NodeID isolated_node_idx : isolated_nodes)
            {
                cluster_assignments[isolated_node_idx] = subgraph_id_counter++;
                remaining_nodes[isolated_node_idx] = false;
                remaining_node_count--;
            }
        }

        if (remaining_node_count == 0)
            break;

        cout << "\n--- Finding Subgraph Iteration " << subgraph_id_counter << " (approximation) ---" << endl;

        vector<NodeID> subgraph_to_original_map;
        map<NodeID, NodeID> original_to_subgraph_map = get_subgraph_mapping(remaining_nodes, num_nodes, subgraph_to_original_map);

        EdgeID subgraph_m;
        vector<vector<pair<NodeID, EdgeID>>> subgraph_adj = get_subgraph_adj(Adj, original_to_subgraph_map, subgraph_m);
        NodeID subgraph_n = original_to_subgraph_map.size();

        cout << "Considering subgraph of " << subgraph_n << " nodes and " << subgraph_m << " edges." << endl;

        if (subgraph_n == 0)
            break;

        if (subgraph_m == 0)
        {
            // Remaining nodes have no edges between themselves -> all are singletons
            for (NodeID original_node_idx : subgraph_to_original_map)
            {
                cluster_assignments[original_node_idx] = subgraph_id_counter++;
            }
            break;
        }

        auto start_fista = chrono::high_resolution_clock::now();

        // 2. Run FISTA to find the "densest" nodes
        pair<vector<bool>, vector<double>> fista_result = fista_frac(subgraph_adj, iters, subgraph_n, subgraph_m);
        vector<bool> densest_subgraph_in_current = fista_result.first;
        vector<double> b_subgraph = fista_result.second;

        auto end_fista = chrono::high_resolution_clock::now();
        cout << "[TIME] FISTA on subgraph: " << chrono::duration_cast<chrono::milliseconds>(end_fista - start_fista).count() << " ms" << endl;

        // Save density values
        for (NodeID j = 0; j < subgraph_n; ++j)
        {
            NodeID original_node_idx = subgraph_to_original_map[j];
            final_b_values[original_node_idx] = b_subgraph[j];
        }

        // 3. Find Connected Components within the densest subset
        // This splits the "densest" result into actual physically connected clusters
        vector<vector<NodeID>> components = get_components_in_subgraph(subgraph_n, subgraph_adj, densest_subgraph_in_current);

        NodeID nodes_removed_this_step = 0;

        if (components.empty())
        {
            cout << "Warning: Extraction step found no nodes. Breaking." << endl;
            break;
        }

        cout << "Densest subset contains " << components.size() << " connected component(s)." << endl;

        for (const auto &comp : components)
        {
            for (NodeID u_sub : comp)
            {
                NodeID original_node_idx = subgraph_to_original_map[u_sub];

                // Assign cluster ID
                cluster_assignments[original_node_idx] = subgraph_id_counter;

                // Mark as removed
                if (remaining_nodes[original_node_idx])
                {
                    remaining_nodes[original_node_idx] = false;
                    nodes_removed_this_step++;
                }
            }
            // Increment cluster ID for the next component
            subgraph_id_counter++;
        }

        remaining_node_count -= nodes_removed_this_step;
        cout << "Removed " << nodes_removed_this_step << " nodes. " << remaining_node_count << " nodes remaining." << endl;
    }

    // --- NEW: Identify Singleton Clusters and Mark as -1 ---
    map<long long, size_t> cluster_counts;
    for (long long c_id : cluster_assignments)
    {
        if (c_id != -1)
            cluster_counts[c_id]++;
    }
    long long singleton_count = 0;
    for (long long &c_id : cluster_assignments)
    {
        if (c_id != -1 && cluster_counts[c_id] == 1)
        {
            c_id = -1; // Set to noise/unassigned
            singleton_count++;
        }
    }
    cout << "\nRemoved " << singleton_count << " singleton clusters (set to -1) before writing." << endl;

    auto start_write = chrono::high_resolution_clock::now();
    // Use specific function for writing clusters (filters -1)
    write_cluster_assignments(output_filepath, cluster_assignments.size(), integer_to_original_map, cluster_assignments);
    cout << "\n[TIME] Writing final assignments: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write).count() << " ms" << endl;

    auto start_write_density = chrono::high_resolution_clock::now();
    // Use specific function for writing density
    write_density_values(density_filepath, final_b_values.size(), integer_to_original_map, final_b_values);
    cout << "[TIME] Writing final density vector: " << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_write_density).count() << " ms" << endl;

    auto end_total = chrono::high_resolution_clock::now();
    cout << "[TIME] Total execution: " << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() << " ms" << endl;

    return 0;
}