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

// --- HELPER: Find connected components within a specific subset of nodes ---
// This ensures that nodes with the same density value are only clustered together
// if they are actually connected in the graph.
vector<vector<NodeID>> get_components_in_subset(
    NodeID n_total,
    const vector<pair<NodeID, NodeID>> &all_edges,
    const vector<NodeID> &subset_nodes)
{
    // 1. Create a fast lookup for the subset
    // Mapping global NodeID -> Local Index (0..subset_size-1)
    map<NodeID, int> global_to_local;
    for (size_t i = 0; i < subset_nodes.size(); ++i)
    {
        global_to_local[subset_nodes[i]] = i;
    }

    // 2. Build Adjacency for the subset
    int subset_size = subset_nodes.size();
    vector<vector<int>> adj(subset_size);

    for (const auto &edge : all_edges)
    {
        auto it1 = global_to_local.find(edge.first);
        auto it2 = global_to_local.find(edge.second);

        // If both endpoints are in this density group
        if (it1 != global_to_local.end() && it2 != global_to_local.end())
        {
            adj[it1->second].push_back(it2->second);
            adj[it2->second].push_back(it1->second);
        }
    }

    // 3. BFS to find components
    vector<bool> visited(subset_size, false);
    vector<vector<NodeID>> components;

    for (int i = 0; i < subset_size; ++i)
    {
        if (!visited[i])
        {
            vector<NodeID> component;
            queue<int> q;

            visited[i] = true;
            q.push(i);
            component.push_back(subset_nodes[i]); // Store Global ID

            while (!q.empty())
            {
                int u_local = q.front();
                q.pop();

                for (int v_local : adj[u_local])
                {
                    if (!visited[v_local])
                    {
                        visited[v_local] = true;
                        q.push(v_local);
                        component.push_back(subset_nodes[v_local]); // Store Global ID
                    }
                }
            }
            components.push_back(component);
        }
    }
    return components;
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

inline map<NodeID, string> reverse_node_mapping(map<string, NodeID> original_to_integer_map)
{
    map<NodeID, string> integer_to_original_map;
    for (auto const &[original_node_id, integer_node_id] : original_to_integer_map)
    {
        integer_to_original_map[integer_node_id] = original_node_id;
    }
    return integer_to_original_map;
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
    cin.tie(0);

    if (argc < 5)
    {
        cerr << "Usage: " << argv[0] << " <ACCURACY> [maxIter] <graph_path> <output_path> <density_path>" << endl;
        return 1;
    }
    int ACCURACY = atoi(argv[1]);
    if (ACCURACY <= 0)
    {
        cerr << "Warning: ACCURACY is non-positive or invalid (" << argv[1] << "), using 100000." << endl;
        ACCURACY = 100000;
    }

    // Adjust indices for arguments
    int maxIter = 100;
    if (argc > 2)
        maxIter = atoi(argv[2]);
    if (maxIter <= 0)
        maxIter = 100;

    std::string graph_path = std::string(argv[3]);
    const char *output_path = argv[4];
    const char *density_path = argv[5];

    char delimiter = get_delimiter(graph_path);
    map<string, NodeID> original_to_integer_map = get_node_mapping(graph_path, delimiter);
    map<NodeID, string> integer_to_original_map = reverse_node_mapping(original_to_integer_map);
    vector<pair<NodeID, NodeID>> edges = read_network_edgelist(graph_path, delimiter, original_to_integer_map);

    NodeID n = original_to_integer_map.size();
    EdgeID m = edges.size();

    // Initialize with initial density
    vector<double> max_density(n, (n > 0 ? 1.0 * m / n : 0.0));

    if (n <= 0 && m > 0)
    {
        cerr << "Error: n <= 0 but m > 0. (n=" << n << ", m=" << m << ")" << endl;
        return 1;
    }

    vector<char> subg(n, 1); // 1 for true, 0 for false

    double density = (n > 0) ? (1.0 * m / n) : 0.0;
    std::cout << "Initial density = " << density << std::endl;

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

    vector<NodeID> new2old;
    new2old.reserve(n);
    vector<pair<NodeID, NodeID>> edges1;
    edges1.reserve(m);

    for (int iter = 0; iter < maxIter; ++iter)
    {
        prev_density = density;

        vector<NodeID> old2new(n, -1);
        new2old.clear();

        NodeID n1 = 0;
        for (NodeID u = 0; u < n; ++u)
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
        }

        edges1.clear();
        for (const auto &e : edges)
        {
            NodeID u_new = (n > 0 && e.first < n) ? old2new[e.first] : -1;
            NodeID v_new = (n > 0 && e.second < n) ? old2new[e.second] : -1;
            if (u_new != -1 && v_new != -1)
            {
                edges1.emplace_back(u_new, v_new);
            }
        }

        EdgeID m1 = edges1.size();
        NodeID SRC = n1 + m1, SNK = SRC + 1, NND = SNK + 1;

        if (NND > max_flow_nodes_alloc)
        {
            cerr << "Error: NND " << NND << " exceeds allocation " << max_flow_nodes_alloc << endl;
            return 1;
        }

        fill(deg.begin(), deg.begin() + NND, 0L);

        // Calculate degrees
        for (NodeID i = 0; i < n1; ++i)
            deg[i] = 1; // Nodes to Sink
        for (const auto &e1 : edges1)
        {
            deg[e1.first]++;
            deg[e1.second]++;
        }
        for (EdgeID j = 0; j < m1; ++j)
            deg[n1 + j] = 3; // Source to Edge, Edge to u, Edge to v
        deg[SRC] = n1;
        deg[SNK] = m1;

        // Prefix sum for current pointers
        for (NodeID i = 1; i < NND; ++i)
            deg[i] += deg[i - 1];
        long long tot_structural_arcs = (NND > 0) ? deg[NND - 1] : 0;

        if (tot_structural_arcs > max_flow_arcs_alloc)
        {
            cerr << "Error: tot_structural_arcs " << tot_structural_arcs << " exceeds allocation " << max_flow_arcs_alloc << endl;
            return 1;
        }

        if (NND > 0)
            cur[0] = 0;
        for (NodeID i = 1; i < NND; ++i)
            cur[i] = deg[i - 1];
        for (NodeID i = 0; i < NND; ++i)
            nodes_ptr[i].first = arcs.data() + cur[i];

        auto add_arc = [&](NodeID u_arc, NodeID v_arc, cType capacity_val)
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

        cType src_u_cap_val = static_cast<cType>(density * ACCURACY);
        if (src_u_cap_val < 0)
            src_u_cap_val = 0;

        for (NodeID u_new = 0; u_new < n1; ++u_new)
            add_arc(SRC, u_new, src_u_cap_val);

        for (EdgeID j = 0; j < m1; ++j)
        {
            const auto &e1 = edges1[j];
            add_arc(e1.first, n1 + j, INF);
            add_arc(e1.second, n1 + j, INF);
            add_arc(n1 + j, SNK, static_cast<cType>(ACCURACY));
        }

        if (NND > 0 && tot_structural_arcs > 0 && SRC < NND && SNK < NND && SRC != SNK)
        {
            min_cut(NND, tot_structural_arcs / 2, nodes_ptr, arcs.data(), cap.data(),
                    &nodes_ptr[SRC], &nodes_ptr[SNK], 0);
        }

        fill(subg.begin(), subg.end(), (char)0);
        long long src_arc_base_offset = (NND > 0 && SRC < NND && nodes_ptr[SRC].first != nullptr) ? (nodes_ptr[SRC].first - arcs.data()) : -1;

        long long vcount = 0;
        if (src_arc_base_offset != -1)
        {
            for (NodeID idx = 0; idx < n1; ++idx)
            {
                // Nodes on source side of cut
                if (nodes_ptr[idx].d < NND && cap[src_arc_base_offset + idx] > 0)
                {
                    subg[new2old[idx]] = 1;
                    vcount++;
                }
            }
        }

        long long ecount = 0;
        for (const auto &edge_pair_new_indices : edges1)
        {
            if (subg[new2old[edge_pair_new_indices.first]] && subg[new2old[edge_pair_new_indices.second]])
            {
                ecount++;
            }
        }

        density = (vcount > 0) ? (static_cast<double>(ecount) / vcount) : 0.0;

        // Update max density seen for these nodes
        for (NodeID u = 0; u < n; ++u)
        {
            if (subg[u])
            {
                max_density[u] = max(max_density[u], density);
            }
        }

        cerr << "Iter " << iter << ": dens=" << density << " (V=" << vcount << " E=" << ecount << ")\n";

        if (prev_density >= 0.0 && abs(density - prev_density) < 1e-12)
            break;
    }

    // --- POST PROCESSING: Group by Density + Connected Components ---

    vector<long long> cluster_assignments(n, -1); // Initialize all to -1 (noise)

    // 1. Group nodes by their max_density
    map<double, vector<NodeID>> density_groups;
    for (NodeID u = 0; u < n; ++u)
    {
        if (max_density[u] > 0)
        { // Only consider positive density
            density_groups[max_density[u]].push_back(u);
        }
    }

    long long final_cluster_id = 0;
    long long total_clustered_nodes = 0;

    // 2. For each density group, split into connected components
    // Iterate in reverse order (highest density first)
    for (auto it = density_groups.rbegin(); it != density_groups.rend(); ++it)
    {
        const vector<NodeID> &nodes_in_group = it->second;

        // Separate this density tier into connected components
        vector<vector<NodeID>> components = get_components_in_subset(n, edges, nodes_in_group);

        for (const auto &comp : components)
        {
            // Remove singleton clusters (size == 1)
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

    cout << "Final: " << final_cluster_id << " clusters processed (" << total_clustered_nodes << " nodes)." << endl;

    // 3. Write Cluster Assignments
    write_cluster_assignments(output_path, n, integer_to_original_map, cluster_assignments);

    // 4. Write Density Values
    write_density_values(density_path, n, integer_to_original_map, max_density);

    return 0;
}