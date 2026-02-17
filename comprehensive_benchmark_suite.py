import time
import igraph as ig
import networkx as nx
import os
import sys
import subprocess

MCB_PRO_SCRIPT = "mcb_cycle_basis_comprehensive_backup.py"

def generate_graph(type_name, params, filename):
    print(f"[Generating] {type_name} {params}...", end="\r")
    if type_name == "Grid2D":
        n = params['n']
        G = nx.grid_2d_graph(n, n)
    elif type_name == "Grid3D":
        n = params['n']
        G = nx.grid_graph(dim=[n, n, n])
    elif type_name == "ErdosRenyi":
        n = params['n']
        p = params['p']
        G = nx.erdos_renyi_graph(n, p)
    elif type_name == "WattsStrogatz":
        n = params['n']
        k = params['k']
        p = params['p']
        G = nx.watts_strogatz_graph(n, k, p)
    elif type_name == "BarabasiAlbert":
        n = params['n']
        m = params['m']
        G = nx.barabasi_albert_graph(n, m)
    elif type_name == "Ladder":
        n = params['n']
        G = nx.ladder_graph(n)
    elif type_name == "Wheel":
        n = params['n']
        G = nx.wheel_graph(n)
    elif type_name == "Torus":
        n = params['n']
        G = nx.grid_2d_graph(n, n, periodic=True) 
    else:
        raise ValueError(f"Unknown graph type: {type_name}")
    
    # Remap to int nodes
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    with open(filename, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
            
    return G.number_of_nodes(), G.number_of_edges()

def run_single_benchmark(name, filename, N, M):
    results = {
        "Name": name,
        "Nodes": N,
        "Edges": M,
        "Time_Pro": None,
        "Time_ig": None,
        "Time_nx": None, # NetworkX
        "Basis_Pro": None,
        "Basis_ig": None,
        "Status": "OK"
    }

    output_file = "temp_bench_out.txt"
    
    # --- 1. MCB-Pro (Python) ---
    t_start = time.time()
    try:
        if os.path.exists(output_file): os.remove(output_file)
        cmd = [sys.executable, MCB_PRO_SCRIPT, filename, output_file]
        # Timeout set to 60s to avoid stuck processes
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
        
        if res.returncode != 0:
            results["Time_Pro"] = float('inf')
            results["Status"] = "Pro Failed"
            # print(f"Pro Error: {res.stderr[:100]}")
        else:
            results["Time_Pro"] = time.time() - t_start
            for line in res.stdout.split('\n'):
                if "线性无关基数量" in line:
                    try:
                        results["Basis_Pro"] = int(line.split(":")[1].strip())
                    except: pass
    except subprocess.TimeoutExpired:
        results["Time_Pro"] = float('inf')
        results["Status"] = "Pro Timeout"
    except Exception as e:
        results["Time_Pro"] = float('inf')
        results["Status"] = f"Pro Err: {str(e)[:20]}"

    # --- 1.5 NetworkX (Pure Python Reference) ---
    # Only run for smaller graphs to avoid infinite wait
    if N <= 500: 
        t_start = time.time()
        try:
            # Re-read graph for NX
            G_nx = nx.Graph()
            with open(filename, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        G_nx.add_edge(int(parts[0]), int(parts[1]))
            
            # NetworkX minimum_cycle_basis
            _ = nx.minimum_cycle_basis(G_nx)
            results["Time_nx"] = time.time() - t_start
        except Exception as e:
            results["Time_nx"] = float('inf')
    else:
         results["Time_nx"] = float('inf') # Skip for large

    # --- 2. igraph (C++) ---
    t_start = time.time()
    try:
        edges = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1])))
        
        # Remap for igraph 0-continguous
        unique_nodes = sorted(list(set(u for e in edges for u in e)))
        node_map = {n: i for i, n in enumerate(unique_nodes)}
        remapped_edges = [(node_map[u], node_map[v]) for u, v in edges]
        
        g = ig.Graph(len(unique_nodes), remapped_edges, directed=False)
        
        # Limit igraph timeout manually by measuring time? No easy way to interrupt C extension.
        # We just hope it finishes or we kill it manually if running interactively.
        # But for script, we assume it's fast enough for these sizes or we accept wait.
        basis = g.minimum_cycle_basis(use_cycle_order=False)
        results["Time_ig"] = time.time() - t_start
        results["Basis_ig"] = len(basis)
        
    except Exception as e:
        results["Time_ig"] = float('inf')
        results["Status"] = f"ig Err: {str(e)[:20]}"

    # Cleanup
    if os.path.exists(filename): os.remove(filename)
    if os.path.exists(output_file): os.remove(output_file)
    
    return results

def main():
    benchmarks = [
        # --- Structured (Mesh/Grid) ---
        ("Grid2D", {'n': 30}, "Grid 30x30"),
        ("Grid2D", {'n': 60}, "Grid 60x60"),
        ("Torus",  {'n': 30}, "Torus 30x30"),
        ("Grid3D", {'n': 10}, "Grid3D 10x10x10"),
        
        # --- Specific Topology ---
        ("Ladder", {'n': 500}, "Ladder (Len=500)"),
        ("Wheel",  {'n': 200}, "Wheel (N=200)"),

        # --- Semi-Structured ---
        ("WattsStrogatz", {'n': 500, 'k': 6, 'p': 0.1}, "Small World (N=500, k=6)"),
        
        # --- Random / Scale Free ---
        ("ErdosRenyi", {'n': 300, 'p': 0.05}, "Random (N=300, p=0.05)"),
        ("BarabasiAlbert", {'n': 800, 'm': 3}, "ScaleFree (N=800, m=3)"),
    ]
    
    all_results = []
    
    print(f"{'Benchmark':<25} | {'N':<5} | {'M':<6} | {'MCB-Pro':<10} | {'igraph':<10} | {'NetX':<10} | {'Note':<8}")
    print("-" * 110)
    
    for type_name, params, label in benchmarks:
        filename = "temp_bench.txt"
        N, M = generate_graph(type_name, params, filename)
        
        res = run_single_benchmark(label, filename, N, M)
        
        # Calc Speedup
        t_pro = res["Time_Pro"]
        t_ig = res["Time_ig"]
        t_nx = res["Time_nx"]
        
        # Format times
        def fmt(t): 
            if t is None or t == float('inf'): return "Skip/Out"
            return f"{t:.4f}s"

        note = ""
        if t_pro != float('inf') and t_ig != float('inf') and t_pro > 0:
            if t_ig > t_pro:
                note = f"🚀 {t_ig/t_pro:.1f}x ig"
        
        print(f"{label:<25} | {N:<5} | {M:<6} | {fmt(t_pro):<10} | {fmt(t_ig):<10} | {fmt(t_nx):<10} | {note}")
        all_results.append(res)
        
if __name__ == "__main__":
    main()