import time
import networkx as nx
import importlib.util
import sys

# Import 01.py dynamically since it starts with a number
spec = importlib.util.spec_from_file_location("module_01", "01.py")
module_01 = importlib.util.module_from_spec(spec)
sys.modules["module_01"] = module_01
spec.loader.exec_module(module_01)

def get_basis_metrix(basis, name):
    total_len = 0
    for c in basis:
        l = len(c)
        if len(c) > 0 and c[0] == c[-1]:
            l -= 1
        total_len += l
    return total_len

def verify():
    print("--- Verifying 01.py (Final Check) ---")
    
    # 1. Wheel Graph 300 (Performance Check)
    # The optimization (heavy edge skip) must be active for this to be fast.
    n = 300
    G = nx.wheel_graph(n)
    print(f"\n[Wheel n={n}] Running...")
    
    t0 = time.time()
    basis = module_01.minimum_cycle_basis_heuristic(G)
    dt = time.time() - t0
    
    total_len = get_basis_metrix(basis, "01.py")
    expected_len = 2 * (n - 1) + (n - 1)  # C3s (len 3) * (n-1)? 
    # Actually for Wheel(n):
    # Edges: Ring (n-1), Spokes (n-1). Total 2n-2.
    # Basis size: E - V + 1 = (2n-2) - n + 1 = n - 1.
    # All are triangles (len 3). Total len = 3 * (n-1).
    expected_total_len = 3 * (n - 1)

    print(f"  Time: {dt:.4f}s")
    print(f"  Basis Size: {len(basis)} (Expected: {n-1})")
    print(f"  Total Length: {total_len} (Expected: {expected_total_len})")
    
    if dt > 1.0:
        print("  [FAIL] Too slow! Heavy edge optimization might be missing or disabled.")
    else:
        print("  [PASS] Speed is good.")
        
    if total_len == expected_total_len:
        print("  [PASS] Accuracy is perfect.")
    else:
        print(f"  [FAIL] Accuracy mismatch. Got {total_len}, expected {expected_total_len}")

if __name__ == "__main__":
    verify()
