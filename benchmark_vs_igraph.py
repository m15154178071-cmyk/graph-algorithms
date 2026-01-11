import time
import networkx as nx
import igraph as ig
import os
import sys

# Ensure we can import the solution
sys.path.append(os.getcwd())
try:
    from solution_enhanced import minimum_cycle_basis_heuristic
except ImportError:
    print("Error: Could not import 'minimum_cycle_basis_heuristic' from solution_enhanced.py")
    sys.exit(1)

def nx_to_igraph(nx_graph):
    """Convert NetworkX graph to igraph for fair comparison."""
    # Mapping nodes to integers 0..n-1
    node_map = {n: i for i, n in enumerate(nx_graph.nodes())}
    g = ig.Graph(len(nx_graph), directed=False)
    edges = [(node_map[u], node_map[v]) for u, v in nx_graph.edges()]
    g.add_edges(edges)
    return g

def calculate_total_length_nodes(basis):
    """Calculate total length for node-based cycles (My Algo)."""
    total = 0
    for cyc in basis:
        # standardizing: usually cycles are closed [A, B, C, A], so len-1
        # but solution_enhanced usually returns [A, B, C] for C3.
        # Let's check the first cycle to see the format or just trust the helper from before.
        # solution_enhanced usually returns simple path list of nodes.
        if len(cyc) > 0 and cyc[0] == cyc[-1]:
            total += len(cyc) - 1
        else:
            total += len(cyc)
    return total

def calculate_total_length_edges(basis_edges):
    """Calculate total length for edge-index cycles (igraph)."""
    # igraph returns cycles as lists of edge IDs. length is just len(cycle).
    return sum(len(c) for c in basis_edges)

def run_benchmark(name, nx_graph):
    print(f"\n[{name}] Nodes: {len(nx_graph.nodes())}, Edges: {len(nx_graph.edges())}")

    # --- 1. My Algorithm (Python Optimized) ---
    t0 = time.time()
    my_basis = minimum_cycle_basis_heuristic(nx_graph)
    t_my = time.time() - t0
    w_my = calculate_total_length_nodes(my_basis)
    print(f"  > My Algo (Python) | Time: {t_my:.4f}s | Basis Size: {len(my_basis)} | Total Len: {w_my}")

    # --- 2. igraph (C Extension) ---
    # Convert first implies a slight overhead, but we time only the algo
    g_ig = nx_to_igraph(nx_graph)
    
    t0 = time.time()
    # igraph uses the "Paton" algorithm or similar for MCB
    # returns list of list of edge indices
    ig_basis = g_ig.minimum_cycle_basis(use_cycle_order=False) 
    t_ig = time.time() - t0
    w_ig = calculate_total_length_edges(ig_basis)
    
    print(f"  > igraph  (C Lib)  | Time: {t_ig:.4f}s | Basis Size: {len(ig_basis)} | Total Len: {w_ig}")

    # Comparison
    ratio = t_my / t_ig if t_ig > 0 else 0
    diff_str = " (Slower)" if ratio > 1 else " (Faster!)"
    print(f"  >>> Ratio: My Algo is {ratio:.2f}x the time of igraph{diff_str}")
    
    if w_my == w_ig:
         print("  ✅ Accuracy: Exact Match")
    else:
         diff_pct = abs(w_my - w_ig) / w_ig * 100
         print(f"  ⚠️ Accuracy: Diff {w_my - w_ig} ({diff_pct:.2f}%)")

if __name__ == "__main__":
    print("Benchmarking: My Python Heuristic vs igraph (C Extension)")
    print("---------------------------------------------------------")

    # Case 1: Grid Graph (Sparse, Regular)
    # Using larger grid to see performance
    G_grid = nx.grid_2d_graph(30, 30) # 900 nodes
    run_benchmark("Grid 30x30", G_grid)

    # Case 2: Wheel Graph (Dense Hub, Regular) - My Algo's speciality
    # igraph usually handles this well too, but let's see.
    G_wheel = nx.wheel_graph(1000)
    run_benchmark("Wheel n=1000", G_wheel)

    # Case 3: Random Graph (Unstructured)
    # Hard for heuristics, easier for raw C speed?
    G_random = nx.gnp_random_graph(200, 0.1, seed=42)
    run_benchmark("Random G(200, 0.1)", G_random)
    
    # Case 4: Larger Random
    # G_random_l = nx.gnp_random_graph(500, 0.05, seed=42)
    # run_benchmark("Random G(500, 0.05)", G_random_l)
