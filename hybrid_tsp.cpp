#include "hybrid_tsp.h"

// ============================================================================
// Global Random Generator
// ============================================================================

std::random_device rd;
std::mt19937 gen(rd());

// ============================================================================
// Constants
// ============================================================================

const double PI = 3.14159265358979323846;
const double RRR = 6378.388;  // Earth radius for GEO coordinates

// ============================================================================
// Helper Functions
// ============================================================================

double convert_to_radians(double x) {
    int deg = (int)x;
    double min = x - deg;
    return PI * (deg + 5.0 * min / 3.0) / 180.0;
}

int nint(double x) {
    return (int)(x + 0.5);
}

// ============================================================================
// TSPProblem Implementation
// ============================================================================

std::string TSPProblem::parse_header_value(const std::string& line) {
    size_t colon_pos = line.find(':');
    return (colon_pos != std::string::npos) ? line.substr(colon_pos + 1) : "";
}

void TSPProblem::compute_nearest_neighbors() {
    nearest_neighbors.resize(N);

    for (int i = 0; i < N; ++i) {
        std::vector<std::pair<double, int>> distances;
        distances.reserve(N - 1);

        for (int j = 0; j < N; ++j) {
            if (i != j) {
                distances.push_back({weights[i][j], j});
            }
        }

        std::sort(distances.begin(), distances.end());

        nearest_neighbors[i].resize(M_neighbors);
        for (int k = 0; k < M_neighbors && k < (int)distances.size(); ++k) {
            nearest_neighbors[i][k] = distances[k].second;
        }
    }
}

void TSPProblem::compute_weights() {
    weights.assign(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dist = 0.0;
            if (edge_weight_type == "EUC_2D") {
                double xd = coords[i].x - coords[j].x;
                double yd = coords[i].y - coords[j].y;
                dist = nint(std::sqrt(xd * xd + yd * yd));
            } else if (edge_weight_type == "GEO") {
                double lat1 = convert_to_radians(coords[i].x);
                double lon1 = convert_to_radians(coords[i].y);
                double lat2 = convert_to_radians(coords[j].x);
                double lon2 = convert_to_radians(coords[j].y);
                double q1 = std::cos(lon1 - lon2);
                double q2 = std::cos(lat1 - lat2);
                double q3 = std::cos(lat1 + lat2);
                dist = (int)(RRR * std::acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
            } else {
                double xd = coords[i].x - coords[j].x;
                double yd = coords[i].y - coords[j].y;
                dist = std::sqrt(xd * xd + yd * yd);
            }
            weights[i][j] = weights[j][i] = dist;
        }
    }
}

bool TSPProblem::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::string line;
    bool coord_section = false;

    while (std::getline(file, line)) {
        if (line.find("EOF") != std::string::npos) break;

        if (!coord_section) {
            if (line.find("NAME") != std::string::npos) name = parse_header_value(line);
            else if (line.find("DIMENSION") != std::string::npos) {
                N = std::stoi(parse_header_value(line));
                coords.reserve(N);
            } else if (line.find("EDGE_WEIGHT_TYPE") != std::string::npos) {
                edge_weight_type = parse_header_value(line);
                edge_weight_type.erase(std::remove_if(edge_weight_type.begin(),
                    edge_weight_type.end(), ::isspace), edge_weight_type.end());
            } else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                coord_section = true;
            }
        } else {
            std::stringstream ss(line);
            int id;
            double x, y;
            if (ss >> id >> x >> y) coords.push_back({id, x, y});
        }
    }
    file.close();

    if (coords.size() != (size_t)N && !coords.empty()) N = coords.size();
    compute_weights();
    return true;
}

void TSPProblem::set_M_neighbors(int M) {
    M_neighbors = std::min(M, N - 1);
    compute_nearest_neighbors();
}

void TSPProblem::set_from_matrix(const DistMatrix& weight_matrix) {
    N = weight_matrix.size();
    weights = weight_matrix;
    edge_weight_type = "EXPLICIT";
    name = "sub_tsp";
    coords.clear();
}

// ============================================================================
// Tour Conversion Functions
// ============================================================================

std::vector<int> tour_matrix_to_list(const TourMatrix& matrix) {
    int N = matrix.size();
    std::vector<int> tour_list;
    tour_list.reserve(N);
    std::vector<bool> visited(N, false);

    int current = 0;
    tour_list.push_back(current);
    visited[current] = true;

    for (int step = 0; step < N - 1; ++step) {
        bool found = false;
        for (int next = 0; next < N; ++next) {
            if (matrix[current][next] == 1 && !visited[next]) {
                current = next;
                tour_list.push_back(current);
                visited[current] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            for (int k = 0; k < N; ++k) {
                if (!visited[k]) {
                    current = k;
                    tour_list.push_back(k);
                    visited[k] = true;
                    break;
                }
            }
        }
    }
    return tour_list;
}

TourMatrix list_to_tour_matrix(const std::vector<int>& tour_list, int N) {
    TourMatrix matrix(N, std::vector<int>(N, 0));
    for (size_t i = 0; i < tour_list.size(); ++i) {
        int u = tour_list[i];
        int v = tour_list[(i + 1) % N];
        matrix[u][v] = matrix[v][u] = 1;
    }
    return matrix;
}

TourMatrix generate_random_tour(int N) {
    std::vector<int> nodes(N);
    std::iota(nodes.begin(), nodes.end(), 0);
    std::shuffle(nodes.begin(), nodes.end(), gen);
    return list_to_tour_matrix(nodes, N);
}

// ============================================================================
// 2-Opt Move Functions
// ============================================================================

void apply_move_inplace(TourMatrix& matrix, int i, int j, int k, int l) {
    matrix[i][j] = matrix[j][i] = 0;
    matrix[k][l] = matrix[l][k] = 0;
    matrix[i][k] = matrix[k][i] = 1;
    matrix[j][l] = matrix[l][j] = 1;
}

void apply_move_list(std::vector<int>& tour, int idx_start, int idx_end) {
    int N = tour.size();
    int dist = (idx_end - idx_start + N) % N;
    int swaps = (dist + 1) / 2;

    for (int s = 0; s < swaps; ++s) {
        int l = (idx_start + s) % N;
        int r = (idx_end - s + N) % N;
        std::swap(tour[l], tour[r]);
    }
}

// ============================================================================
// Energy Calculation Functions
// ============================================================================

double calculate_tour_energy(const std::vector<int>& tour, const DistMatrix& weights) {
    double energy = 0.0;
    int N = tour.size();
    for (int i = 0; i < N; ++i) {
        energy += weights[tour[i]][tour[(i + 1) % N]];
    }
    return energy;
}

double calculate_tour_energy_matrix(const TourMatrix& tour, const DistMatrix& weights) {
    double energy = 0.0;
    int N = tour.size();
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (tour[i][j]) energy += weights[i][j];
        }
    }
    return energy;
}

double calculate_total_energy(const DistMatrix& weights, const TrotterSlices& tours, double J_perp) {
    double H = 0.0;
    int P = tours.size();
    int N = tours[0].size();

    for (const auto& tour : tours) {
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                if (tour[i][j]) H += weights[i][j];
            }
        }
    }

    if (P > 1 && J_perp != 0.0) {
        for (int p = 0; p < P; ++p) {
            int prev_p = (p - 1 + P) % P;
            for (int r = 0; r < N; ++r) {
                for (int c = r + 1; c < N; ++c) {
                    int s_curr = 2 * tours[p][r][c] - 1;
                    int s_prev = 2 * tours[prev_p][r][c] - 1;
                    H -= J_perp * s_curr * s_prev;
                }
            }
        }
    }
    return H;
}

// ============================================================================
// Tour Validation
// ============================================================================

bool tour_valid(const std::vector<int>& tour, int N) {
    if ((int)tour.size() != N) return false;

    std::vector<bool> visited(N, false);
    for (int city : tour) {
        if (city < 0 || city >= N || visited[city]) return false;
        visited[city] = true;
    }
    return true;
}

// ============================================================================
// Simulated Annealing Optimization
// ============================================================================

void simulated_annealing_optimization(
    const TSPProblem& problem,
    std::vector<int>& tour_list,
    std::vector<int>& pos_in_tour,
    TourMatrix& tour_matrix,
    double T_start,
    double T_end,
    int sa_steps
) {
    int N = problem.N;
    int M = std::min(problem.M_neighbors, N - 1);

    std::uniform_int_distribution<> dist_idx(0, N - 1);
    std::uniform_int_distribution<> dist_neighbor(0, M - 1);
    std::uniform_real_distribution<> dist_real(0.0, 1.0);

    int moves_per_step = M * N;

    for (int step = 0; step < sa_steps; ++step) {
        double T_current = T_start - (T_start - T_end) * ((double)step / sa_steps);

        for (int attempt = 0; attempt < moves_per_step; ++attempt) {
            int idx_i = dist_idx(gen);
            int i = tour_list[idx_i];
            int j = tour_list[(idx_i + 1) % N];

            int neighbor_idx = dist_neighbor(gen);
            int l = problem.nearest_neighbors[j][neighbor_idx];

            int idx_l = pos_in_tour[l];
            int idx_k = (idx_l - 1 + N) % N;
            int k = tour_list[idx_k];

            if (idx_k == idx_i || idx_k == (idx_i + 1) % N ||
                idx_l == idx_i || idx_l == (idx_i + 1) % N) {
                continue;
            }

            double dH = (problem.weights[i][k] + problem.weights[j][l]) -
                        (problem.weights[i][j] + problem.weights[k][l]);

            if (dH < 0 || std::exp(-dH / T_current) > dist_real(gen)) {
                apply_move_inplace(tour_matrix, i, j, k, l);

                int start_rev = (idx_i + 1) % N;
                int end_rev = idx_k;
                apply_move_list(tour_list, start_rev, end_rev);

                int dist = (end_rev - start_rev + N) % N;
                for (int s = 0; s <= dist; ++s) {
                    int idx = (start_rev + s) % N;
                    pos_in_tour[tour_list[idx]] = idx;
                }
            }
        }
    }
}

void local_search_2opt(
    const TSPProblem& problem,
    std::vector<int>& tour_list,
    std::vector<int>& pos_in_tour,
    TourMatrix& tour_matrix,
    int max_iterations
) {
    // Use SA with very low temperature (essentially greedy local search)
    simulated_annealing_optimization(
        problem, tour_list, pos_in_tour, tour_matrix,
        0.01, 0.001, max_iterations
    );
}

// ============================================================================
// Step 1: Generate Classical Pool
// ============================================================================

std::vector<PoolEntry> generate_classical_pool(
    const TSPProblem& problem,
    int N_I,
    int local_search_steps
) {
    std::vector<PoolEntry> pool;
    pool.reserve(N_I);

    for (int i = 0; i < N_I; ++i) {
        // Generate random tour
        TourMatrix tour_matrix = generate_random_tour(problem.N);
        std::vector<int> tour_list = tour_matrix_to_list(tour_matrix);

        // Build position lookup table
        std::vector<int> pos_in_tour(problem.N);
        for (int idx = 0; idx < problem.N; ++idx) {
            pos_in_tour[tour_list[idx]] = idx;
        }

        // Apply local search
        simulated_annealing_optimization(
            problem, tour_list, pos_in_tour, tour_matrix,
            100.0, 0.1, local_search_steps
        );

        // Calculate energy and add to pool
        double energy = calculate_tour_energy(tour_list, problem.weights);
        pool.push_back({tour_list, energy, 0});
    }

    // Sort by energy
    std::sort(pool.begin(), pool.end());

    return pool;
}

// ============================================================================
// Step 2: Analyze Edge Frequencies
// ============================================================================

EdgeAnalysis analyze_edge_frequencies(
    const std::vector<PoolEntry>& pool,
    int N_S,
    double threshold,
    int N
) {
    EdgeAnalysis result;
    result.frequency_matrix.assign(N, std::vector<double>(N, 0.0));

    // Randomly sample N_S solutions from the pool
    int sample_size = std::min(N_S, (int)pool.size());

    // Create indices and shuffle to get random sample
    std::vector<int> indices(pool.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);

    // Count edge frequencies from random sample
    for (int i = 0; i < sample_size; ++i) {
        const auto& tour = pool[indices[i]].tour;
        for (size_t j = 0; j < tour.size(); ++j) {
            int a = tour[j];
            int b = tour[(j + 1) % tour.size()];
            result.frequency_matrix[a][b] += 1.0;
            result.frequency_matrix[b][a] += 1.0;
        }
    }

    // Normalize and identify fixed edges
    result.num_fixed = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            result.frequency_matrix[i][j] /= sample_size;
            result.frequency_matrix[j][i] = result.frequency_matrix[i][j];

            if (result.frequency_matrix[i][j] >= threshold) {
                result.fixed_edges.push_back({i, j});
                result.num_fixed++;
            }
        }
    }

    return result;
}

// ============================================================================
// Step 3: Contract Graph
// ============================================================================

ContractionResult contract_graph(
    const std::vector<std::pair<int,int>>& fixed_edges,
    int N
) {
    ContractionResult result;
    result.city_to_super_node.assign(N, -1);

    // Build adjacency list for fixed edges only
    std::vector<std::vector<int>> adj(N);
    for (const auto& edge : fixed_edges) {
        adj[edge.first].push_back(edge.second);
        adj[edge.second].push_back(edge.first);
    }

    // Find degree of each node in the fixed edge graph
    std::vector<int> degree(N, 0);
    for (int i = 0; i < N; ++i) {
        degree[i] = adj[i].size();
    }

    // Find chains by starting from endpoints (degree 1) or isolated edges
    std::vector<bool> visited(N, false);
    int super_node_id = 0;

    // First pass: find chains starting from degree-1 nodes (endpoints)
    for (int start = 0; start < N; ++start) {
        if (degree[start] == 1 && !visited[start]) {
            // Start a new chain from this endpoint
            SuperNode sn;
            sn.id = super_node_id++;

            int current = start;
            while (current != -1) {
                visited[current] = true;
                sn.cities.push_back(current);
                result.city_to_super_node[current] = sn.id;

                // Find unvisited neighbor
                int next = -1;
                for (int neighbor : adj[current]) {
                    if (!visited[neighbor]) {
                        next = neighbor;
                        break;
                    }
                }
                current = next;
            }

            if (sn.cities.size() >= 2) {
                sn.endpoint_A = sn.cities.front();
                sn.endpoint_B = sn.cities.back();
                result.super_nodes.push_back(sn);
            } else {
                // Single city that was degree 1 but neighbor was already visited
                // This shouldn't happen in practice, treat as free city
                result.city_to_super_node[sn.cities[0]] = -1;
                super_node_id--;
            }
        }
    }

    // Second pass: find cycles (all degree 2, no endpoints visited yet)
    for (int start = 0; start < N; ++start) {
        if (degree[start] == 2 && !visited[start]) {
            // This is part of a cycle
            SuperNode sn;
            sn.id = super_node_id++;

            int current = start;
            while (!visited[current]) {
                visited[current] = true;
                sn.cities.push_back(current);
                result.city_to_super_node[current] = sn.id;

                // Find unvisited neighbor
                for (int neighbor : adj[current]) {
                    if (!visited[neighbor]) {
                        current = neighbor;
                        break;
                    }
                }
            }

            // For a cycle, we can pick any two adjacent nodes as endpoints
            // We'll break the cycle at the last edge
            if (sn.cities.size() >= 2) {
                sn.endpoint_A = sn.cities.front();
                sn.endpoint_B = sn.cities.back();
                result.super_nodes.push_back(sn);
            } else {
                // Single city "cycle" - reset to free city (same fix as first pass)
                for (int city : sn.cities) {
                    result.city_to_super_node[city] = -1;
                }
                super_node_id--;
            }
        }
    }

    // Collect free cities (degree 0 in fixed edges, not part of any chain)
    for (int i = 0; i < N; ++i) {
        if (result.city_to_super_node[i] == -1) {
            result.free_cities.push_back(i);
        }
    }

    result.N_sub = result.free_cities.size() + result.super_nodes.size();

    return result;
}

// ============================================================================
// Step 4: Build Sub-TSP
// ============================================================================

SubTSP build_sub_tsp(
    const TSPProblem& original_problem,
    const ContractionResult& contraction
) {
    SubTSP result;
    result.N_sub = contraction.N_sub;

    // Build entity map: first free cities, then super-nodes
    result.entity_map.reserve(result.N_sub);

    for (int city : contraction.free_cities) {
        result.entity_map.push_back({false, city});
    }

    for (const auto& sn : contraction.super_nodes) {
        result.entity_map.push_back({true, sn.id});
    }

    // Build sub-weights matrix
    result.sub_weights.assign(result.N_sub, std::vector<double>(result.N_sub, 0.0));

    for (int i = 0; i < result.N_sub; ++i) {
        for (int j = i + 1; j < result.N_sub; ++j) {
            double dist = std::numeric_limits<double>::infinity();

            const auto& entity_i = result.entity_map[i];
            const auto& entity_j = result.entity_map[j];

            if (!entity_i.is_super_node && !entity_j.is_super_node) {
                // Free city to free city
                dist = original_problem.weights[entity_i.original_id][entity_j.original_id];
            }
            else if (!entity_i.is_super_node && entity_j.is_super_node) {
                // Free city to super-node
                int city = entity_i.original_id;
                const SuperNode& sn = contraction.super_nodes[entity_j.original_id];
                dist = std::min(
                    original_problem.weights[city][sn.endpoint_A],
                    original_problem.weights[city][sn.endpoint_B]
                );
            }
            else if (entity_i.is_super_node && !entity_j.is_super_node) {
                // Super-node to free city
                const SuperNode& sn = contraction.super_nodes[entity_i.original_id];
                int city = entity_j.original_id;
                dist = std::min(
                    original_problem.weights[sn.endpoint_A][city],
                    original_problem.weights[sn.endpoint_B][city]
                );
            }
            else {
                // Super-node to super-node
                const SuperNode& sn1 = contraction.super_nodes[entity_i.original_id];
                const SuperNode& sn2 = contraction.super_nodes[entity_j.original_id];
                dist = std::min({
                    original_problem.weights[sn1.endpoint_A][sn2.endpoint_A],
                    original_problem.weights[sn1.endpoint_A][sn2.endpoint_B],
                    original_problem.weights[sn1.endpoint_B][sn2.endpoint_A],
                    original_problem.weights[sn1.endpoint_B][sn2.endpoint_B]
                });
            }

            result.sub_weights[i][j] = result.sub_weights[j][i] = dist;
        }
    }

    return result;
}

// ============================================================================
// Step 5: PIMC Quantum Annealing
// ============================================================================

std::pair<std::vector<int>, double> quantum_annealing(
    const TSPProblem& problem,
    int steps,
    double G_start,
    double T,
    int P,
    int sa_preannealing_steps
) {
    // Generate random initial tour
    TourMatrix initial_tour = generate_random_tour(problem.N);
    TrotterSlices tours(P, initial_tour);

    std::vector<std::vector<int>> tour_lists(P);
    std::vector<std::vector<int>> pos_in_tour(P, std::vector<int>(problem.N));

    // Initialize tour_lists and pos_in_tour
    for (int p = 0; p < P; ++p) {
        tour_lists[p] = tour_matrix_to_list(tours[p]);
        for (int idx = 0; idx < problem.N; ++idx) {
            pos_in_tour[p][tour_lists[p][idx]] = idx;
        }
    }

    // Pre-annealing with classical SA
    if (sa_preannealing_steps > 0) {
        double T0 = P * T;
        double T_start = 5.0 * T0;
        double T_end = T0;

        for (int p = 0; p < P; ++p) {
            simulated_annealing_optimization(
                problem, tour_lists[p], pos_in_tour[p], tours[p],
                T_start, T_end, sa_preannealing_steps
            );
        }
    }

    // PIMC main loop
    int M = std::min(problem.M_neighbors, problem.N - 1);
    std::uniform_int_distribution<> dist_idx(0, problem.N - 1);
    std::uniform_int_distribution<> dist_neighbor(0, M - 1);
    std::uniform_real_distribution<> dist_real(0.0, 1.0);

    int moves_per_step = M * problem.N;

    for (int step = 0; step < steps; ++step) {
        double G = G_start * (1.0 - (double)step / steps);
        if (G < 1e-5) G = 1e-5;

        double arg = std::tanh(G / (P * T));
        if (arg <= 0) arg = 1e-10;
        double J_perp = (-P * T / 2.0) * std::log(arg);

        for (int p = 0; p < P; ++p) {
            for (int attempt = 0; attempt < moves_per_step; ++attempt) {
                int N = problem.N;

                int idx_i = dist_idx(gen);
                int i = tour_lists[p][idx_i];
                int j = tour_lists[p][(idx_i + 1) % N];

                int neighbor_idx = dist_neighbor(gen);
                int l = problem.nearest_neighbors[j][neighbor_idx];

                int idx_l = pos_in_tour[p][l];
                int idx_k = (idx_l - 1 + N) % N;
                int k = tour_lists[p][idx_k];

                if (idx_k == idx_i || idx_k == (idx_i + 1) % N ||
                    idx_l == idx_i || idx_l == (idx_i + 1) % N) {
                    continue;
                }

                // Delta potential energy
                double dH_pot = (problem.weights[i][k] + problem.weights[j][l]) -
                                (problem.weights[i][j] + problem.weights[k][l]);

                // Delta kinetic energy
                double dH_kin = 0.0;
                int prev_p = (p - 1 + P) % P;
                int next_p = (p + 1) % P;

                auto interaction_delta = [&](int u, int v, int old_bit, int new_bit) {
                    int s_new = 2 * new_bit - 1;
                    int s_old = 2 * old_bit - 1;
                    int ds = s_new - s_old;

                    int s_prev = 2 * tours[prev_p][u][v] - 1;
                    int s_next = 2 * tours[next_p][u][v] - 1;

                    return -J_perp * ds * (s_prev + s_next);
                };

                dH_kin += interaction_delta(i, j, 1, 0);
                dH_kin += interaction_delta(k, l, 1, 0);
                dH_kin += interaction_delta(std::min(i,k), std::max(i,k), 0, 1);
                dH_kin += interaction_delta(std::min(j,l), std::max(j,l), 0, 1);

                double total_delta = dH_pot + dH_kin;

                if (total_delta < 0 || std::exp(-total_delta / (P * T)) > dist_real(gen)) {
                    apply_move_inplace(tours[p], i, j, k, l);

                    int start_rev = (idx_i + 1) % N;
                    int end_rev = idx_k;
                    apply_move_list(tour_lists[p], start_rev, end_rev);

                    int dist = (end_rev - start_rev + N) % N;
                    for (int s = 0; s <= dist; ++s) {
                        int idx = (start_rev + s) % N;
                        pos_in_tour[p][tour_lists[p][idx]] = idx;
                    }
                }
            }
        }
    }

    // Find best solution among all Trotter slices
    double best_E = std::numeric_limits<double>::infinity();
    std::vector<int> best_tour;

    for (int p = 0; p < P; ++p) {
        double energy = calculate_tour_energy(tour_lists[p], problem.weights);
        if (energy < best_E) {
            best_E = energy;
            best_tour = tour_lists[p];
        }
    }

    return {best_tour, best_E};
}

std::vector<int> solve_sub_tsp_pimc(
    const SubTSP& sub_tsp,
    const HybridConfig& config
) {
    // Create a TSPProblem from the sub-TSP weights
    TSPProblem sub_problem;
    sub_problem.set_from_matrix(sub_tsp.sub_weights);

    int M = std::min(config.M_neighbors, sub_tsp.N_sub - 1);
    sub_problem.set_M_neighbors(M);

    // Run PIMC
    auto [tour, energy] = quantum_annealing(
        sub_problem,
        config.QA_steps,
        config.QA_G0,
        config.QA_T,
        config.QA_P,
        config.sa_preannealing
    );

    return tour;
}

// ============================================================================
// Step 6: Expand Tour
// ============================================================================

std::vector<int> expand_tour(
    const std::vector<int>& sub_tour,
    const SubTSP& sub_tsp,
    const ContractionResult& contraction,
    const TSPProblem& original_problem
) {
    std::vector<int> full_tour;
    full_tour.reserve(original_problem.N);

    int N_sub = sub_tour.size();
    if (N_sub == 0) return full_tour;

    // Step 1: Rotate sub_tour to start from a free city (if any exists)
    // This avoids the circular dependency problem for the first element
    int start_offset = 0;
    for (int i = 0; i < N_sub; ++i) {
        const auto& entity = sub_tsp.entity_map[sub_tour[i]];
        if (!entity.is_super_node) {
            start_offset = i;
            break;
        }
    }

    // Create rotated tour starting from free city
    std::vector<int> rotated_tour(N_sub);
    for (int i = 0; i < N_sub; ++i) {
        rotated_tour[i] = sub_tour[(i + start_offset) % N_sub];
    }

    // Step 2: Expand with correct tracking of prev_exit
    // prev_exit is ALWAYS the last city added to full_tour
    for (int pos = 0; pos < N_sub; ++pos) {
        int entity_idx = rotated_tour[pos];
        const auto& entity = sub_tsp.entity_map[entity_idx];

        if (!entity.is_super_node) {
            // Free city: just add it
            full_tour.push_back(entity.original_id);
        } else {
            // Super-node: determine direction based on last inserted city
            const SuperNode& sn = contraction.super_nodes[entity.original_id];

            if (full_tour.empty()) {
                // First element is a super-node (shouldn't happen after rotation, but handle it)
                // Just insert in default order
                for (int city : sn.cities) {
                    full_tour.push_back(city);
                }
            } else {
                // prev_exit is simply the last city we added
                int prev_exit = full_tour.back();

                // Determine which endpoint of super-node to enter from
                double dist_to_A = original_problem.weights[prev_exit][sn.endpoint_A];
                double dist_to_B = original_problem.weights[prev_exit][sn.endpoint_B];

                if (dist_to_A <= dist_to_B) {
                    // Enter from endpoint_A, traverse in normal order
                    for (int city : sn.cities) {
                        full_tour.push_back(city);
                    }
                } else {
                    // Enter from endpoint_B, traverse in reverse order
                    for (int i = sn.cities.size() - 1; i >= 0; --i) {
                        full_tour.push_back(sn.cities[i]);
                    }
                }
            }
        }
    }

    return full_tour;
}

// ============================================================================
// Step 7: Update Pool
// ============================================================================

void update_pool(
    std::vector<PoolEntry>& pool,
    const std::vector<int>& new_tour,
    double new_energy,
    int N_I
) {
    // Increment age of all entries
    for (auto& entry : pool) {
        entry.age++;
    }

    // Add new tour
    pool.push_back({new_tour, new_energy, 0});

    // Sort by energy
    std::sort(pool.begin(), pool.end());

    // Keep only top N_I
    if ((int)pool.size() > N_I) {
        pool.resize(N_I);
    }
}

// ============================================================================
// Print Functions
// ============================================================================

void print_config(const HybridConfig& config) {
    std::cout << "=== Hybrid TSP Configuration ===" << std::endl;
    std::cout << "Pool size (N_I): " << config.N_I << std::endl;
    std::cout << "Sample size (N_S): " << config.N_S << std::endl;
    std::cout << "Edge threshold: " << config.threshold << std::endl;
    std::cout << "Convergence patience (K): " << config.K << std::endl;
    std::cout << "Local search steps: " << config.sa_local_search_steps << std::endl;
    std::cout << "Stagnation threshold: " << std::fixed << std::setprecision(2)
              << (config.stagnation_threshold * 100) << "%" << std::endl;
    std::cout << "--- PIMC Parameters ---" << std::endl;
    std::cout << "QA_G0: " << config.QA_G0 << std::endl;
    std::cout << "QA_T: " << config.QA_T << " (PT=" << config.QA_P * config.QA_T << ")" << std::endl;
    std::cout << "QA_P: " << config.QA_P << std::endl;
    std::cout << "QA_steps: " << config.QA_steps << std::endl;
    std::cout << "M_neighbors: " << config.M_neighbors << std::endl;
    std::cout << "SA pre-annealing: " << config.sa_preannealing << std::endl;
    std::cout << "================================" << std::endl;
}

void print_pool_stats(const std::vector<PoolEntry>& pool) {
    if (pool.empty()) return;

    double sum = 0, min_e = pool[0].energy, max_e = pool[0].energy;
    for (const auto& entry : pool) {
        sum += entry.energy;
        min_e = std::min(min_e, entry.energy);
        max_e = std::max(max_e, entry.energy);
    }
    double avg = sum / pool.size();

    std::cout << "Pool: min=" << min_e << ", avg=" << std::fixed << std::setprecision(1)
              << avg << ", max=" << max_e << std::endl;
}

void print_contraction_stats(const ContractionResult& contraction, int original_N) {
    std::cout << "Contraction: " << contraction.super_nodes.size() << " super-nodes, "
              << contraction.free_cities.size() << " free cities -> N_sub="
              << contraction.N_sub << " (from " << original_N << ")" << std::endl;
}

// ============================================================================
// Main Hybrid Solver
// ============================================================================

double hybrid_solve(
    const TSPProblem& problem,
    HybridConfig& config,
    bool verbose
) {
    if (verbose) {
        std::cout << "\n=== Starting Hybrid Classical + PIMC Solver ===" << std::endl;
        std::cout << "Problem: " << problem.name << " (N=" << problem.N << ")" << std::endl;
        print_config(config);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    long long pimc_time_ms = 0;           // Accumulate PIMC time
    long long local_search_time_ms = 0;   // Accumulate local search time

    // Step 1: Generate initial pool
    if (verbose) std::cout << "\n[Step 1] Generating initial pool of " << config.N_I << " solutions..." << std::endl;

    auto ls_start = std::chrono::high_resolution_clock::now();
    std::vector<PoolEntry> pool = generate_classical_pool(
        problem, config.N_I, config.sa_local_search_steps
    );
    auto ls_end = std::chrono::high_resolution_clock::now();
    local_search_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(ls_end - ls_start).count();

    if (verbose) print_pool_stats(pool);

    double best_energy = pool[0].energy;
    std::vector<int> best_tour = pool[0].tour;
    int stagnation_counter = 0;
    int iteration = 0;

    // Main loop
    while (stagnation_counter < config.K) {
        iteration++;
        if (verbose) {
            std::cout << "\n--- Iteration " << iteration
                      << " (stagnation: " << stagnation_counter << "/" << config.K << ") ---" << std::endl;
        }

        // Step 1.5: Refine all solutions in the pool with local search 
        if (verbose) {
            std::cout << "[Step 1.5] Refining pool with local search..." << std::endl;
        }

        auto ls_refine_start = std::chrono::high_resolution_clock::now();

        bool pool_improved = false;
        int refinement_steps = std::max(1, config.sa_local_search_steps / 2);

        for (auto& entry : pool) {
            double old_energy = entry.energy;

            // Reconstruct tour_matrix and pos_in_tour from the list representation
            TourMatrix tour_matrix = list_to_tour_matrix(entry.tour, problem.N);
            std::vector<int> pos_in_tour(problem.N);
            for (int i = 0; i < (int)entry.tour.size(); ++i) {
                pos_in_tour[entry.tour[i]] = i;
            }

            // Apply local search
            local_search_2opt(problem, entry.tour, pos_in_tour, tour_matrix, refinement_steps);

            // Recalculate energy
            entry.energy = calculate_tour_energy(entry.tour, problem.weights);

            if (entry.energy < old_energy) {
                pool_improved = true;
            }
        }

        auto ls_refine_end = std::chrono::high_resolution_clock::now();
        local_search_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(ls_refine_end - ls_refine_start).count();

        // If pool improved, re-sort and check for new global best
        if (pool_improved) {
            std::sort(pool.begin(), pool.end());

            if (pool[0].energy < best_energy) {
                double improvement_pct = (best_energy - pool[0].energy) / best_energy;
                double old_best = best_energy;
                best_energy = pool[0].energy;
                best_tour = pool[0].tour;

                // Only reset stagnation if improvement exceeds threshold
                if (improvement_pct >= config.stagnation_threshold) {
                    stagnation_counter = 0;
                    if (verbose) {
                        std::cout << "*** Local search found new best: " << best_energy
                                  << " (improved " << std::fixed << std::setprecision(2)
                                  << (improvement_pct * 100) << "%) ***" << std::endl;
                    }
                } else {
                    stagnation_counter++;
                    if (verbose) {
                        std::cout << "*** Local search found marginal improvement: " << old_best
                                  << " -> " << best_energy << " (" << std::fixed << std::setprecision(2)
                                  << (improvement_pct * 100) << "% < " << (config.stagnation_threshold * 100)
                                  << "% threshold, stagnation++) ***" << std::endl;
                    }
                }
            }
        }

        // Step 2: Analyze edge frequencies
        EdgeAnalysis edge_analysis = analyze_edge_frequencies(
            pool, config.N_S, config.threshold, problem.N
        );

        if (verbose) {
            std::cout << "[Step 2] Fixed edges: " << edge_analysis.num_fixed
                      << " (threshold=" << config.threshold << ")" << std::endl;
        }

        // Step 3: Contract graph
        ContractionResult contraction = contract_graph(edge_analysis.fixed_edges, problem.N);

        if (verbose) {
            print_contraction_stats(contraction, problem.N);
        }

        // Check if problem is too small or fully contracted
        if (contraction.N_sub < 3) {
            if (verbose) {
                std::cout << "Problem fully contracted (N_sub < 3). Solution found." << std::endl;
            }
            break;
        }

        // If no contraction happened, reduce threshold slightly
        if (contraction.N_sub == problem.N && edge_analysis.num_fixed == 0) {
            if (verbose) {
                std::cout << "No edges fixed. Reducing threshold from " << config.threshold;
            }
            config.threshold = std::max(0.5, config.threshold - 0.05);
            if (verbose) {
                std::cout << " to " << config.threshold << std::endl;
            }
            stagnation_counter++;
            continue;
        }

        // Step 4: Build sub-TSP
        SubTSP sub_tsp = build_sub_tsp(problem, contraction);

        if (verbose) {
            std::cout << "[Step 4] Built sub-TSP with N_sub=" << sub_tsp.N_sub << std::endl;
        }

        // Step 5: Solve sub-TSP with PIMC
        if (verbose) {
            std::cout << "[Step 5] Solving sub-TSP with PIMC..." << std::endl;
        }

        auto pimc_start = std::chrono::high_resolution_clock::now();
        std::vector<int> sub_tour = solve_sub_tsp_pimc(sub_tsp, config);
        auto pimc_end = std::chrono::high_resolution_clock::now();
        pimc_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(pimc_end - pimc_start).count();

        // Step 6: Expand tour
        std::vector<int> full_tour = expand_tour(sub_tour, sub_tsp, contraction, problem);

        // Validate tour
        if (!tour_valid(full_tour, problem.N)) {
            if (verbose) {
                std::cout << "Warning: Expanded tour is invalid. Skipping." << std::endl;

                // Debug: identify the specific problem
                std::cout << "  DEBUG: full_tour.size()=" << full_tour.size()
                          << ", expected N=" << problem.N << std::endl;

                if ((int)full_tour.size() != problem.N) {
                    std::cout << "  CAUSE: Size mismatch" << std::endl;
                    std::cout << "    - Free cities: " << contraction.free_cities.size() << std::endl;
                    std::cout << "    - Super-nodes: " << contraction.super_nodes.size() << std::endl;
                    int total_in_supernodes = 0;
                    for (const auto& sn : contraction.super_nodes) {
                        total_in_supernodes += sn.cities.size();
                    }
                    std::cout << "    - Cities in super-nodes: " << total_in_supernodes << std::endl;
                    std::cout << "    - Total accounted: " << (contraction.free_cities.size() + total_in_supernodes) << std::endl;
                } else {
                    // Check for duplicates or out-of-range
                    std::vector<int> count(problem.N, 0);
                    for (int city : full_tour) {
                        if (city >= 0 && city < problem.N) {
                            count[city]++;
                        }
                    }

                    std::vector<int> duplicates, missing;
                    for (int i = 0; i < problem.N; ++i) {
                        if (count[i] > 1) duplicates.push_back(i);
                        if (count[i] == 0) missing.push_back(i);
                    }

                    if (!duplicates.empty()) {
                        std::cout << "  CAUSE: Duplicate cities: ";
                        for (int d : duplicates) std::cout << d << " ";
                        std::cout << std::endl;
                    }
                    if (!missing.empty()) {
                        std::cout << "  CAUSE: Missing cities: ";
                        for (int m : missing) std::cout << m << " ";
                        std::cout << std::endl;
                    }
                }
            }
            stagnation_counter++;
            continue;
        }

        // Calculate energy
        double new_energy = calculate_tour_energy(full_tour, problem.weights);

        if (verbose) {
            std::cout << "[Step 6] Expanded tour energy: " << new_energy << std::endl;
        }

        // Step 7: Update pool
        update_pool(pool, full_tour, new_energy, config.N_I);

        if (verbose) {
            print_pool_stats(pool);
        }

        // Check for improvement
        if (pool[0].energy < best_energy) {
            double improvement_pct = (best_energy - pool[0].energy) / best_energy;
            double old_best = best_energy;
            best_energy = pool[0].energy;
            best_tour = pool[0].tour;

            // Only reset stagnation if improvement exceeds threshold
            if (improvement_pct >= config.stagnation_threshold) {
                stagnation_counter = 0;
                if (verbose) {
                    std::cout << "*** New best energy: " << best_energy
                              << " (improved " << std::fixed << std::setprecision(2)
                              << (improvement_pct * 100) << "%) ***" << std::endl;
                }
            } else {
                stagnation_counter++;
                if (verbose) {
                    std::cout << "*** Marginal improvement: " << old_best << " -> " << best_energy
                              << " (" << std::fixed << std::setprecision(2)
                              << (improvement_pct * 100) << "% < " << (config.stagnation_threshold * 100)
                              << "% threshold, stagnation++) ***" << std::endl;
                }
            }
        } else {
            stagnation_counter++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    long long classical_time_ms = total_time_ms - pimc_time_ms;
    long long other_time_ms = classical_time_ms - local_search_time_ms;

    // Always print timing summary (for parsing in quiet mode)
    double pimc_pct = total_time_ms > 0 ? (100.0 * pimc_time_ms / total_time_ms) : 0.0;
    double ls_pct = total_time_ms > 0 ? (100.0 * local_search_time_ms / total_time_ms) : 0.0;

    if (verbose) {
        std::cout << "\n=== Hybrid Solver Complete ===" << std::endl;
        std::cout << "Best energy found: " << best_energy << std::endl;
        std::cout << "Total iterations: " << iteration << std::endl;
        std::cout << "\nTiming:" << std::endl;
        std::cout << "  Total time:       " << total_time_ms << " ms" << std::endl;
        std::cout << "  PIMC time:        " << pimc_time_ms << " ms ("
                  << std::fixed << std::setprecision(1) << pimc_pct << "%)" << std::endl;
        std::cout << "  Classical time:   " << classical_time_ms << " ms ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * classical_time_ms / total_time_ms) << "%)" << std::endl;
        std::cout << "    - Local search: " << local_search_time_ms << " ms ("
                  << std::fixed << std::setprecision(1) << ls_pct << "%)" << std::endl;
        std::cout << "    - Other:        " << other_time_ms << " ms ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * other_time_ms / total_time_ms) << "%)" << std::endl;
        std::cout << "==============================" << std::endl;
    } else {
        // Compact summary for quiet mode (parseable)
        std::cout << "TIMING: " << total_time_ms << "ms PIMC:" << std::fixed << std::setprecision(1)
                  << pimc_pct << "% LS:" << ls_pct << "%" << std::endl;
    }

    return best_energy;
}

// ============================================================================
// Main Function
// ============================================================================

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " <tsp_file> [options]" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --pool-size N       Pool size (default: 50)" << std::endl;
    std::cout << "  --sample-size N     Sample size for frequency analysis (default: 30)" << std::endl;
    std::cout << "  --threshold T       Edge fixing threshold (default: 0.95)" << std::endl;
    std::cout << "  --patience K        Convergence patience (default: 10)" << std::endl;
    std::cout << "  --local-steps N     Local search steps (default: 100)" << std::endl;
    std::cout << "  --stagnation-pct P  Min improvement % to reset stagnation (default: 0)" << std::endl;
    std::cout << "  --qa-g0 G           Initial gamma (default: 300)" << std::endl;
    std::cout << "  --qa-t T            Temperature (default: 3.33)" << std::endl;
    std::cout << "  --qa-p P            Trotter slices (default: 30)" << std::endl;
    std::cout << "  --qa-steps S        QA MC steps (default: 1000)" << std::endl;
    std::cout << "  --m-neighbors M     Neighborhood pruning (default: 20)" << std::endl;
    std::cout << "  --sa-preannealing S SA pre-annealing steps (default: 100)" << std::endl;
    std::cout << "  --quiet             Suppress verbose output" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string filename = argv[1];
    HybridConfig config;
    bool verbose = true;

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--pool-size" && i + 1 < argc) {
            config.N_I = std::stoi(argv[++i]);
        } else if (arg == "--sample-size" && i + 1 < argc) {
            config.N_S = std::stoi(argv[++i]);
        } else if (arg == "--threshold" && i + 1 < argc) {
            config.threshold = std::stod(argv[++i]);
        } else if (arg == "--patience" && i + 1 < argc) {
            config.K = std::stoi(argv[++i]);
        } else if (arg == "--local-steps" && i + 1 < argc) {
            config.sa_local_search_steps = std::stoi(argv[++i]);
        } else if (arg == "--stagnation-pct" && i + 1 < argc) {
            config.stagnation_threshold = std::stod(argv[++i]) / 100.0;  // Convert % to fraction
        } else if (arg == "--qa-g0" && i + 1 < argc) {
            config.QA_G0 = std::stod(argv[++i]);
        } else if (arg == "--qa-t" && i + 1 < argc) {
            config.QA_T = std::stod(argv[++i]);
        } else if (arg == "--qa-p" && i + 1 < argc) {
            config.QA_P = std::stoi(argv[++i]);
        } else if (arg == "--qa-steps" && i + 1 < argc) {
            config.QA_steps = std::stoi(argv[++i]);
        } else if (arg == "--m-neighbors" && i + 1 < argc) {
            config.M_neighbors = std::stoi(argv[++i]);
        } else if (arg == "--sa-preannealing" && i + 1 < argc) {
            config.sa_preannealing = std::stoi(argv[++i]);
        } else if (arg == "--quiet") {
            verbose = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Load TSP problem
    TSPProblem problem;
    if (!problem.load_from_file(filename)) {
        std::cerr << "Failed to load TSP problem from " << filename << std::endl;
        return 1;
    }

    // Set M_neighbors
    problem.set_M_neighbors(config.M_neighbors);

    // Run hybrid solver
    double best_energy = hybrid_solve(problem, config, verbose);

    // Always print final result (for parsing)
    std::cout << "\nFinal Result: " << best_energy << std::endl;

    return 0;
}
