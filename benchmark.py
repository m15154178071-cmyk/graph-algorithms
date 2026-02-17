import time
import subprocess
import os
import sys
import networkx as nx

def generate_graph(type_name, params, filename):
    """Generates a graph based on type and params."""
    if type_name == "grid":
        G = nx.grid_2d_graph(params['rows'], params['cols'])
    elif type_name == "erdos":
        # Erdos-Renyi: n nodes, p probability
        G = nx.erdos_renyi_graph(params['n'], params['p'], seed=42)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
    elif type_name == "barabasi":
        # Barabasi-Albert: n nodes, m edges to attach
        G = nx.barabasi_albert_graph(params['n'], params['m'], seed=42)
    elif type_name == "watts":
        # Watts-Strogatz: n nodes, k neighbors, p rewiring
        G = nx.watts_strogatz_graph(params['n'], params['k'], params['p'], seed=42)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
    else:
        raise ValueError(f"Unknown graph type: {type_name}")

    G = nx.convert_node_labels_to_integers(G)
    
    with open(filename, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    return G

def run_single_benchmark(script_name, input_file, output_file, timeout_sec=20):
    start_time = time.time()
    try:
        cmd = [sys.executable, script_name, input_file, output_file]
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            text=True,
            timeout=timeout_sec
        )
        duration = time.time() - start_time
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown Error"
            return {
                "status": "Failed",
                "time": duration,
                "cycles": 0, "weight": 0, "max_len": 0,
                "error": error_msg.split('\n')[-1][:100]
            }

        if not os.path.exists(output_file):
            return {"status": "No Output", "time": duration, "cycles": 0, "weight": 0, "max_len": 0}

        try:
            with open(output_file, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            
            if not lines:
                 return {"status": "Empty", "time": duration, "cycles": 0, "weight": 0, "max_len": 0}
                 
            num_cycles = int(lines[0])
            total_weight = 0
            max_len = 0
            
            for line in lines[1:]:
                parts = list(map(int, line.split()))
                if len(parts) > 0:
                    length = parts[0]
                    total_weight += length
                    if length > max_len: max_len = length
                    
            return {
                "status": "Success",
                "time": duration,
                "cycles": num_cycles,
                "weight": total_weight,
                "max_len": max_len
            }
            
        except ValueError:
             return {"status": "ParseError", "time": duration, "cycles": 0, "weight": 0, "max_len": 0}

    except subprocess.TimeoutExpired:
        return {"status": "Timeout", "time": timeout_sec, "cycles": 0, "weight": 0, "max_len": 0}
    except Exception as e:
        return {"status": "Error", "time": 0, "cycles": 0, "weight": 0, "max_len": 0, "error": str(e)}

def main():
    # Test Scenarios
    scenarios = [
        {"name": "Grid 15x15", "type": "grid", "params": {"rows": 15, "cols": 15}, "timeout": 30},
        {"name": "Grid 30x30", "type": "grid", "params": {"rows": 30, "cols": 30}, "timeout": 60},
        {"name": "Erdos(N=300,p=0.03)", "type": "erdos", "params": {"n": 300, "p": 0.03}, "timeout": 60},
        {"name": "Barabasi(N=300,m=3)", "type": "barabasi", "params": {"n": 300, "m": 3}, "timeout": 60},
        {"name": "Watts(N=300,k=6,p=0.1)", "type": "watts", "params": {"n": 300, "k": 6, "p": 0.1}, "timeout": 60},
    ]
    
    scripts = [
        "mcb_cycle_basis_simple.py", 
        "mcb_cycle_basis_optimized.py", 
        "mcb_cycle_basis_comprehensive.py"
    ]
    
    input_file = "temp_bench_input.txt"
    output_temp = "temp_bench_output.txt"
    
    # Header with Quality Metrics
    print(f"{'Graph':<20} | {'Script':<20} | {'Status':<8} | {'Time(s)':<7} | {'Cycles':<6} | {'Weight':<7} | {'MaxLen':<6}")
    print("-" * 95)
    
    for scen in scenarios:
        name = scen["name"]
        
        # Generate Graph
        generate_graph(scen["type"], scen["params"], input_file)
        
        for script in scripts:
            if not os.path.exists(script): continue
            
            if os.path.exists(output_temp): os.remove(output_temp)
            
            res = run_single_benchmark(script, input_file, output_temp, scen["timeout"])
            
            status = res['status']
            if status == "Success":
                print(f"{name:<20} | {script:<20} | {status:<8} | {res['time']:.4f}  | {res['cycles']:<6} | {res['weight']:<7} | {res['max_len']:<6}")
            else:
                print(f"{name:<20} | {script:<20} | {status:<8} | -        | -      | -       | -     ")
                # print error if available
                # if res.get('error'): print(f"  Error: {res['error']}")

    if os.path.exists(input_file): os.remove(input_file)
    if os.path.exists(output_temp): os.remove(output_temp)

if __name__ == "__main__":
    main()
