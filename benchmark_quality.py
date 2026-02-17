import time
import igraph as ig
import networkx as nx
import os
import sys
import subprocess
import statistics

# MCB-Pro script path
MCB_PRO_SCRIPT = "mcb_cycle_basis_comprehensive_backup.py"

def generate_graphs():
    graphs = []
    
    # 1. Grid 20x20 (Mesh structure, high locality)
    print("Generating Grid 20x20...", end="\r")
    G = nx.grid_2d_graph(20, 20)
    G = nx.relabel_nodes(G, {n: i for i, n in enumerate(G.nodes())})
    graphs.append(("Grid 20x20", G))
    
    # 2. Watts-Strogatz (Small World, like power grids)
    print("Generating Watts-Strogatz (N=400, k=4, p=0.1)...", end="\r")
    G = nx.watts_strogatz_graph(400, 4, 0.1)
    graphs.append(("Watts-Strogatz (N=400)", G))
    
    # 3. Barabasi-Albert (Scale Free, like internet/social)
    print("Generating Barabasi-Albert (N=300, m=2)...", end="\r")
    G = nx.barabasi_albert_graph(300, 2)
    graphs.append(("Scale-Free (N=300)", G))
    
    # 4. Random Geometric Graph (Spatial, like sensor networks)
    print("Generating Geometric (N=300, r=0.15)...", end="\r")
    # radius 0.15 usually ensures connectivity for N=300
    G = nx.random_geometric_graph(300, 0.15)
    # Ensure connected components handling or just take largest
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
        graphs[-1] = ("Geometric (N=300, Connected)", G)
    else:
        graphs.append(("Geometric (N=300)", G))

    return graphs

def run_benchmark(name, G):
    filename = "temp_bench_input.txt"
    outfile = "temp_bench_output.txt"
    
    # Save to file
    with open(filename, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
            
    print(f"\n[{name}] Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # --- 1. igraph (Ground Truth for Minimum Basis) ---
    t0 = time.time()
    edges = [(u, v) for u, v in G.edges()]
    g_ig = ig.Graph(G.number_of_nodes(), edges, directed=False)
    # minimum_cycle_basis returns list of vertex lists
    basis_ig = g_ig.minimum_cycle_basis(use_cycle_order=False)
    t_ig = time.time() - t0
    
    size_ig = len(basis_ig)
    weight_ig = sum(len(c) for c in basis_ig)
    max_len_ig = max(len(c) for c in basis_ig) if basis_ig else 0
    
    # --- 2. MCB-Pro ---
    t0 = time.time()
    size_pro = 0
    weight_pro = 0
    max_len_pro = 0
    
    try:
        if os.path.exists(outfile): os.remove(outfile)
        subprocess.run([sys.executable, MCB_PRO_SCRIPT, filename, outfile], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        
        t_pro = time.time() - t0
        
        # Parse Output File for precise stats
        # Output format: First line count, then lines of "length n1 n2 ..."
        with open(outfile, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                try:
                    # Filter out empty lines
                    cycle_lines = [l for l in lines[1:] if l.strip()]
                    size_pro = len(cycle_lines)
                    weights = []
                    for cl in cycle_lines:
                        parts = cl.strip().split()
                        # format: length n1 n2 ...
                        # Length is usually parts[0]
                        # But strictly, cycle length is number of nodes.
                        # Check format from script: f"{len(nodes)} {' '.join(str(x) for x in nodes)}"
                        # nodes contains closed path [v1...v1], so length is len(nodes)-1 ?
                        # No, let's look at script output format logic:
                        # "buffer.append(f"{len(nodes)} ...")" where nodes is path [v1...v1]
                        # Wait, reconstruct_cycle_path returns [v1, v2, ..., v1]
                        # So len(nodes) is Length + 1.
                        # Let's verify standard output usually implies edges.
                        # Actually cycle length = edges = nodes - 1 (for closed loop v1..v1)
                        # Let's parse the node list to be sure.
                        nodes_list = parts[1:]
                        # If first and last are same, it's closed.
                        # Length = len(nodes_list) - 1.
                        
                        # Wait, let's check comprehensive backup code logic:
                        # nodes = path # [v1, v2, ..., v1]
                        # buffer.append(f"{len(nodes)} ...")
                        # So the first number is (Cycle Length + 1).
                        l_val = int(parts[0])
                        cycle_len = l_val - 1
                        weights.append(cycle_len)
                        
                    weight_pro = sum(weights)
                    max_len_pro = max(weights) if weights else 0
                    
                except ValueError: 
                    # Fallback if parsing fails
                    pass
                    
    except subprocess.CalledProcessError as e:
        print(f"MCB-Pro Failed")
        t_pro = 0.0
        
    # --- Comparison Report ---
    print(f"{'Metric':<15} | {'igraph (Optimal)':<18} | {'MCB-Pro (Yours)':<18} | {'Diff':<10}")
    print("-" * 65)
    print(f"{'Time':<15} | {t_ig:.4f} s            | {t_pro:.4f} s            | {t_ig/t_pro if t_pro>0 else 0:.1f}x speedup")
    print(f"{'Basis Size':<15} | {size_ig:<18} | {size_pro:<18} | {size_pro - size_ig:+}")
    print(f"{'Total Weight':<15} | {weight_ig:<18} | {weight_pro:<18} | {weight_pro - weight_ig:+}")
    print(f"{'Max Cycle Len':<15} | {max_len_ig:<18} | {max_len_pro:<18} | {max_len_pro - max_len_ig:+}")
    
    # Clean up
    if os.path.exists(filename): os.remove(filename)
    if os.path.exists(outfile): os.remove(outfile)
    
if __name__ == "__main__":
    for name, G in generate_graphs():
        run_benchmark(name, G)
