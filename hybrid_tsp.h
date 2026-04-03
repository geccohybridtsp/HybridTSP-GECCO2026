#ifndef HYBRID_TSP_H
#define HYBRID_TSP_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <set>
#include <map>
#include <utility>

// ============================================================================
// Type Definitions
// ============================================================================

using TourMatrix = std::vector<std::vector<int>>;
using DistMatrix = std::vector<std::vector<double>>;
using TrotterSlices = std::vector<TourMatrix>;

// ============================================================================
// Basic Structures 
// ============================================================================

struct Point {
    int id;
    double x;
    double y;
};

// ============================================================================
// Hybrid-Specific Structures
// ============================================================================

// Super-node representing a contracted chain of cities
struct SuperNode {
    int id;                         // Unique super-node identifier
    std::vector<int> cities;        // Ordered sequence of cities in the chain
    int endpoint_A;                 // First city (entry point A)
    int endpoint_B;                 // Last city (entry point B)

    // Internal distance (sum of edges within the chain)
    double internal_distance;
};

// Mapping entity for sub-TSP indices
struct SubTSPEntity {
    bool is_super_node;             // true = super-node, false = free city
    int original_id;                // City ID (if free) or SuperNode ID
};

// Pool entry with tour and its energy
struct PoolEntry {
    std::vector<int> tour;          // Tour as city list
    double energy;                  // Tour length
    int age;                        // Iteration count since creation

    bool operator<(const PoolEntry& other) const {
        return energy < other.energy;
    }
};

// Edge frequency analysis result
struct EdgeAnalysis {
    std::vector<std::vector<double>> frequency_matrix;  // N×N frequencies [0,1]
    std::vector<std::pair<int,int>> fixed_edges;        // Edges >= threshold
    int num_fixed;
};

// Graph contraction result
struct ContractionResult {
    std::vector<SuperNode> super_nodes;
    std::vector<int> free_cities;                       // Cities not in any super-node
    std::vector<int> city_to_super_node;                // -1 if free, else super-node ID
    int N_sub;                                          // Size of sub-problem
};

// Sub-TSP problem representation
struct SubTSP {
    int N_sub;
    std::vector<std::vector<double>> sub_weights;       // N_sub × N_sub distance matrix
    std::vector<SubTSPEntity> entity_map;               // Maps sub-TSP index to original
};

// Configuration parameters
struct HybridConfig {
    // Pool parameters
    int N_I = 50;                       // Pool size
    int N_S = 30;                       // Sample size for frequency analysis
    double threshold = 0.95;            // Edge fixing threshold
    int K = 10;                         // Convergence patience
    int sa_local_search_steps = 100;    // Steps for initial local search
    double stagnation_threshold = 0.0;  // Min % improvement to reset stagnation (0 = any improvement)

    // PIMC parameters
    double QA_G0 = 300.0;               // Initial gamma
    double QA_T = 3.33;                 // Temperature (PT = 100)
    int QA_P = 30;                      // Trotter slices
    int QA_steps = 1000;                // QA MC steps
    int M_neighbors = 20;               // Neighborhood pruning
    int sa_preannealing = 100;          // SA pre-annealing steps
};

// ============================================================================
// TSP Problem Class
// ============================================================================

class TSPProblem {
public:
    int N;
    std::string name;
    std::string edge_weight_type;
    DistMatrix weights;
    std::vector<Point> coords;
    std::vector<std::vector<int>> nearest_neighbors;
    int M_neighbors;

    TSPProblem() : N(0), M_neighbors(20) {}

    bool load_from_file(const std::string& filename);
    void set_M_neighbors(int M);
    void set_from_matrix(const DistMatrix& weight_matrix);

private:
    std::string parse_header_value(const std::string& line);
    void compute_nearest_neighbors();
    void compute_weights();
};

// ============================================================================
// Core TSP Functions
// ============================================================================

// Tour conversions
std::vector<int> tour_matrix_to_list(const TourMatrix& matrix);
TourMatrix list_to_tour_matrix(const std::vector<int>& tour_list, int N);
TourMatrix generate_random_tour(int N);

// 2-opt moves
void apply_move_inplace(TourMatrix& matrix, int i, int j, int k, int l);
void apply_move_list(std::vector<int>& tour, int idx_start, int idx_end);

// Energy calculations
double calculate_tour_energy(const std::vector<int>& tour, const DistMatrix& weights);
double calculate_tour_energy_matrix(const TourMatrix& tour, const DistMatrix& weights);

// ============================================================================
// Classical Optimization Functions
// ============================================================================

// Local search with 2-opt
void local_search_2opt(
    const TSPProblem& problem,
    std::vector<int>& tour_list,
    std::vector<int>& pos_in_tour,
    TourMatrix& tour_matrix,
    int max_iterations
);

// Simulated annealing (for pool generation)
void simulated_annealing_optimization(
    const TSPProblem& problem,
    std::vector<int>& tour_list,
    std::vector<int>& pos_in_tour,
    TourMatrix& tour_matrix,
    double T_start,
    double T_end,
    int sa_steps
);

// ============================================================================
// Hybrid Algorithm Functions
// ============================================================================

// Step 1: Generate initial pool of solutions
std::vector<PoolEntry> generate_classical_pool(
    const TSPProblem& problem,
    int N_I,
    int local_search_steps
);

// Step 2: Analyze edge frequencies
EdgeAnalysis analyze_edge_frequencies(
    const std::vector<PoolEntry>& pool,
    int N_S,
    double threshold,
    int N
);

// Step 3: Contract graph into super-nodes
ContractionResult contract_graph(
    const std::vector<std::pair<int,int>>& fixed_edges,
    int N
);

// Step 4: Build sub-TSP
SubTSP build_sub_tsp(
    const TSPProblem& original_problem,
    const ContractionResult& contraction
);

// Step 5: Solve sub-TSP with PIMC
std::vector<int> solve_sub_tsp_pimc(
    const SubTSP& sub_tsp,
    const HybridConfig& config
);

// Step 6: Expand tour back to full size
std::vector<int> expand_tour(
    const std::vector<int>& sub_tour,
    const SubTSP& sub_tsp,
    const ContractionResult& contraction,
    const TSPProblem& original_problem
);

// Step 7: Update pool
void update_pool(
    std::vector<PoolEntry>& pool,
    const std::vector<int>& new_tour,
    double new_energy,
    int N_I
);

// Main hybrid solver
double hybrid_solve(
    const TSPProblem& problem,
    HybridConfig& config,
    bool verbose = true
);

// ============================================================================
// PIMC Quantum Annealing 
// ============================================================================

double calculate_total_energy(
    const DistMatrix& weights,
    const TrotterSlices& tours,
    double J_perp
);

std::pair<std::vector<int>, double> quantum_annealing(
    const TSPProblem& problem,
    int steps,
    double G_start,
    double T,
    int P,
    int sa_preannealing_steps
);

// ============================================================================
// Utility Functions
// ============================================================================

// Validate tour
bool tour_valid(const std::vector<int>& tour, int N);

// Print functions
void print_config(const HybridConfig& config);
void print_pool_stats(const std::vector<PoolEntry>& pool);
void print_contraction_stats(const ContractionResult& contraction, int original_N);

#endif // HYBRID_TSP_H
