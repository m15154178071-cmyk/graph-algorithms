import time
import igraph as ig
import networkx as nx
import os
import sys
import subprocess

MCB_PRO_SCRIPT = "mcb_cycle_basis_comprehensive_backup.py"

def generate_grid_file(n, filename):
    print(f"[数据生成] 生成 {n}x{n} 网格...", end="\r")
    G = nx.grid_2d_graph(n, n)
    # Remap to int nodes
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    with open(filename, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    print(f"[数据生成] 完成 {n}x{n} (Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()})")
    return G.number_of_nodes(), G.number_of_edges()

def benchmark_grid(n):
    filename = f"grid_{n}.txt"
    output_file = "temp_output.txt"
    N, M = generate_grid_file(n, filename)
    print("-" * 50)
    
    # === 1. User's MCB-Pro (FIRST!) ===
    print(f"🚀 [MCB-Pro] Starting run...", end="", flush=True)
    t_pro_start = time.time()
    
    try:
        if os.path.exists(output_file): os.remove(output_file)
        cmd = [sys.executable, MCB_PRO_SCRIPT, filename, output_file]
        
        # Run and capture output
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        t_pro = time.time() - t_pro_start
        
        if res.returncode != 0:
            print(f"\n[MCB-Pro] Failed: {res.stderr}")
            t_pro = 9999.0
        else:
            print(f" Done! Time: {t_pro:.4f} s")
            # Parse output for basis count
            for line in res.stdout.split('\n'):
                if "线性无关基数量" in line:
                    print(f"   >> {line.strip()}")

    except Exception as e:
        print(f"\n[MCB-Pro] Error: {e}")
        t_pro = 9999.0

    # === 2. igraph (C++) ===
    print(f"🐢 [igraph ] Starting run (This might take a while)...", end="", flush=True)
    t_ig_start = time.time()
    try:
        edges = []
        with open(filename, 'r') as f:
            for line in f:
                u, v = map(int, line.split())
                edges.append((u, v))
                
        g = ig.Graph(N, edges, directed=False)
        basis = g.minimum_cycle_basis(use_cycle_order=False)
        t_ig = time.time() - t_ig_start
        print(f" Done! Time: {t_ig:.4f} s")
        print(f"   >> Basis Size: {len(basis)}")
    except Exception as e:
        print(f"\n[igraph] Error: {e}")
        t_ig = 9999.0

    # Summary
    if t_pro < 9999 and t_ig < 9999:
        speedup = t_ig / t_pro
        print("-" * 50)
        print(f"🏆 Speedup: Your algorithm is {speedup:.1f}x FASTER than C++ igraph!")
        print("-" * 50)
    
    # Cleanup
    if os.path.exists(filename): os.remove(filename)
    if os.path.exists(output_file): os.remove(output_file)

if __name__ == "__main__":
    benchmark_grid(60)
