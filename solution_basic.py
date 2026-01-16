import sys
import time
import random
import re
from collections import deque, defaultdict
import itertools

# Increase recursion depth just in case
sys.setrecursionlimit(20000)

class DataHelper:
    @staticmethod
    def read_edges(filepath):
        edges = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = re.split(r'[,\s]+', line)
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        if u != v:
                            edges.append((min(u, v), max(u, v)))
                    except:
                        pass
        return edges

class LinearBasis:
    """
    Maintains a linear basis of cycles using a pivot-based approach (Gaussian Elimination).
    Cycles are treated as vectors in GF(2), represented by sets of edge IDs.
    """
    def __init__(self, num_edges):
        self.basis = {} # pivot_edge_id -> cycle_as_set
        self.num_edges = num_edges
        self.basis_count = 0

    def insert(self, cycle_edges):
        """
        Tries to insert cycle. Returns True if inserted (independent), False if dependent.
        """
        temp = set(cycle_edges)
        
        while temp:
            pivot = max(temp)
            if pivot in self.basis:
                temp.symmetric_difference_update(self.basis[pivot])
            else:
                self.basis[pivot] = temp
                self.basis_count += 1
                return True
        return False

def bfs_with_detours(u, v, adj, edge_to_id, visited_limit=200, detours=5, max_path_len=None):
    """
    Returns a list of candidate cycles (each is a set of edge IDs).
    """
    candidates = []
    
    # Base Edge ID
    base_eid = edge_to_id.get((min(u, v), max(u, v)))
    if base_eid is None: return []

    # 1. Primary Shortest Path (Standard BFS)
    # parent: node -> (pred_node, edge_id_from_pred)
    parent = {u: (None, None)}
    dist = {u: 0}
    queue = deque([u])
    found = False
    nodes_visited = 0
    
    visited_set = {u}
    
    primary_path_len = 0
    
    while queue:
        curr = queue.popleft()
        nodes_visited += 1
        if nodes_visited > visited_limit:
            break
            
        if curr == v:
            found = True
            primary_path_len = dist[curr]
            break
        
        # Depth limit check
        if max_path_len is not None and dist[curr] >= max_path_len:
            continue
            
        for nb in adj[curr]:
            # Mask the direct edge (u, v)
            if curr == u and nb == v: continue
            if curr == v and nb == u: continue
            
            if nb not in visited_set:
                visited_set.add(nb)
                eid = edge_to_id.get((min(curr, nb), max(curr, nb)))
                parent[nb] = (curr, eid)
                dist[nb] = dist[curr] + 1
                queue.append(nb)
                
    primary_path_eids = []
    if found:
        # Reconstruct
        curr = v
        while curr != u:
            pred_info = parent[curr]
            if pred_info is None: break # Should not happen
            pred, eid = pred_info
            primary_path_eids.append(eid)
            curr = pred
        
        cycle = frozenset(primary_path_eids + [base_eid])
        candidates.append(cycle)
        
    # 2. Detour Logic (If Primary found)
    # To improve rank, we force the path to diverge.
    if found and detours > 0 and len(primary_path_eids) > 1:
        # Sample edges to block
        edges_to_try = primary_path_eids[:]
        if len(edges_to_try) > detours:
            edges_to_try = random.sample(edges_to_try, detours)
            
        for blocked_eid in edges_to_try:
            # Run BFS again with blocked_eid
            p_visited = {u}
            p_parent = {u: (None, None)}
            p_dist = {u: 0}
            p_queue = deque([u])
            p_found = False
            p_cnt = 0
            
            # Slightly higher limit for detours?
            detour_limit = visited_limit + 100
            
            while p_queue:
                curr = p_queue.popleft()
                p_cnt += 1
                if p_cnt > detour_limit:
                    break
                    
                if curr == v:
                    p_found = True
                    break
                
                if max_path_len is not None and p_dist[curr] >= max_path_len:
                    continue
                
                for nb in adj[curr]:
                    if curr == u and nb == v: continue
                    
                    eid = edge_to_id.get((min(curr, nb), max(curr, nb)))
                    if eid == blocked_eid: continue # BLOCKED
                    
                    if nb not in p_visited:
                        p_visited.add(nb)
                        p_parent[nb] = (curr, eid)
                        p_dist[nb] = p_dist[curr] + 1
                        p_queue.append(nb)
            
            if p_found:
                d_eids = []
                curr = v
                while curr != u:
                    pkg = p_parent[curr]
                    if pkg is None: break
                    pred, eid = pkg
                    d_eids.append(eid)
                    curr = pred
                
                d_cycle = frozenset(d_eids + [base_eid])
                candidates.append(d_cycle)


    return candidates

def get_fundamental_cycles(adj, edges, num_nodes):
    """
    Generates fundamental cycles from a BFS spanning tree.
    Ensures that we have enough candidates to cover the full rank.
    """
    # 1. Build Spanning Forest
    # handle disconnected components
    visited = set()
    parent = {} # child -> parent
    depth = {}
    
    # Store candidates
    candidates = []
    
    # Edge set for fast lookup? Not needed if we iterate 'edges' list.
    # But we need to know which are tree edges.
    tree_edges = set()
    
    all_nodes = sorted(list(adj.keys()))
    
    for root in all_nodes:
        if root in visited: continue
        
        # BFS from root
        q = deque([root])
        visited.add(root)
        parent[root] = None
        depth[root] = 0
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    q.append(v)
                    tree_edges.add(tuple(sorted((u, v))))
    
    # 2. Iterate all edges. If not int tree, it's a back-edge -> Fundamental Cycle.
    # We need edge_to_id mapping to return edge IDs.
    
    # We can pass edge_to_id or assume caller handles lookup if we return node paths?
    # Better to return edge IDs directly.
    return [] # Placeholder, implemented inside main or helper with full context
    
def get_fundamental_cycles_eids(adj, edge_to_id, num_nodes):
    candidates = []
    visited = set()
    parent = {} 
    depth = {}
    
    nodes = sorted(list(adj.keys()))
    
    # Build Tree
    for root in nodes:
        if root in visited: continue
        q = deque([root])
        visited.add(root)
        parent[root] = None
        depth[root] = 0
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    q.append(v)

    # Find Back Edges
    # Iterate all unique edges
    for (pair, eid) in edge_to_id.items():
        u, v = pair
        # Check if tree edge: parent[u] == v or parent[v] == u
        is_tree = False
        if u in parent and parent[u] == v: is_tree = True
        if v in parent and parent[v] == u: is_tree = True
        
        if not is_tree:
            # Back Edge. Form Cycle.
            # LCA approach or just trace up.
            # Cycle = Edge(u,v) + Path(u...LCA...v)
            
            p1 = []
            c1 = u
            while c1 is not None:
                p1.append(c1)
                c1 = parent.get(c1)
                
            p2 = []
            c2 = v
            while c2 is not None:
                p2.append(c2)
                c2 = parent.get(c2)
            
            # Truncate to LCA
            # Reverse to have root at start?
            # p1: [u, p(u), ... root]
            # p2: [v, p(v), ... root]
            
            # Find LCA
            lca = None
            # Set lookup
            p1_set = set(p1)
            for x in p2:
                if x in p1_set:
                    lca = x
                    break
            
            if lca is None: continue # Disconnected? Should not happen if in component.
            
            # Construct cycle eids
            cyc_eids = [eid]
            
            # Trace u up to lca
            curr = u
            while curr != lca:
                par = parent[curr]
                if par is None: break
                e = edge_to_id.get(tuple(sorted((curr, par))))
                if e is not None: cyc_eids.append(e)
                curr = par
                
            # Trace v up to lca
            curr = v
            while curr != lca:
                par = parent[curr]
                if par is None: break
                e = edge_to_id.get(tuple(sorted((curr, par))))
                if e is not None: cyc_eids.append(e)
                curr = par
                
            candidates.append(frozenset(cyc_eids))
            
    return candidates

def find_squares(adj, edge_to_id):
    """
    Finds all simple squares (4-cycles). 
    """
    squares = []
    covered_edges = set()
    adj_sets = {u: set(nbs) for u, nbs in adj.items()}
    sorted_nodes = sorted(list(adj.keys()))
    
    for u in sorted_nodes:
        # u is min node index to avoid rotation duplicates
        u_nbs = adj_sets[u]
        valid_v = [n for n in u_nbs if n > u]
        
        for v in valid_v:
            v_nbs = adj_sets[v]
            for w in v_nbs:
                if w <= u: continue
                if w == v: continue
                
                # Check intersection of N(u) and N(w)
                # We need x in N(u) and N(w).
                # To avoid u-v-w-x vs u-x-w-v, enforce x > v
                
                w_nbs = adj_sets[w]
                common = u_nbs.intersection(w_nbs)
                
                for x in common:
                    if x > v:
                        # Found square u-v-w-x-u
                        e1 = edge_to_id.get((min(u, v), max(u, v)))
                        e2 = edge_to_id.get((min(v, w), max(v, w)))
                        e3 = edge_to_id.get((min(w, x), max(w, x)))
                        e4 = edge_to_id.get((min(x, u), max(x, u)))
                        
                        if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                            sq = frozenset([e1, e2, e3, e4])
                            squares.append(sq)
                            covered_edges.update([e1, e2, e3, e4])
    return squares, covered_edges

def find_triangles(adj, edge_to_id):
    """
    Finds all triangles efficiently using set intersections.
    Returns:
        triangles: list of frozenset(eids)
        covered_edges: set of edge indices that are part of at least one triangle
    """
    triangles = []
    covered_edges = set()
    
    # Pre-convert to sets for intersection
    adj_sets = {u: set(nbs) for u, nbs in adj.items()}
    
    # Iterate all edges to find triangles
    # To avoid duplication, we enforce u < v < w order or iterate edges
    # Iterating edges (u, v) with u < v is cleaner.
    
    # edge_to_id keys are (min, max)
    sorted_edges = sorted(edge_to_id.keys())
    
    for (u, v) in sorted_edges:
        # Common neighbors
        common = adj_sets[u].intersection(adj_sets[v])
        
        for w in common:
            # Found triangle u-v-w
            # We only record it if u < v < w to ensure uniqueness
            # Since we iterate edges (u,v) where u < v:
            # Case 1: w > v. Then u < v < w. This is the unique canonical check.
            # Case 2: w < u. Then w < u < v. We would have seen it when processing (w, u).
            # Case 3: u < w < v. We would have seen it when processing (u, w).
            
            if w > v:
                eid1 = edge_to_id.get((u, v))
                eid2 = edge_to_id.get((min(v, w), max(v, w)))
                eid3 = edge_to_id.get((min(u, w), max(u, w)))
                
                if eid1 is not None and eid2 is not None and eid3 is not None:
                    triangles.append(frozenset([eid1, eid2, eid3]))
                    covered_edges.add(eid1)
                    covered_edges.add(eid2)
                    covered_edges.add(eid3)
                    
    return triangles, covered_edges

def main():
    if len(sys.argv) < 3:
        print("Usage: python solution.py input.txt output.txt")
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    t0 = time.time()
    
    # 1. Load Data
    raw_edges = DataHelper.read_edges(input_file)
    
    # Node Mapping
    nodes = set()
    for u, v in raw_edges:
        nodes.add(u)
        nodes.add(v)
    sorted_nodes = sorted(list(nodes))
    node_map = {n: i for i, n in enumerate(sorted_nodes)}
    inv_node_map = {i: n for i, n in enumerate(sorted_nodes)}
    
    edges = []
    adj = defaultdict(list)
    edge_to_id = {}
    id_to_edge = {}
    
    for idx, (u, v) in enumerate(raw_edges):
        ui, vi = node_map[u], node_map[v]
        edges.append((ui, vi))
        adj[ui].append(vi)
        adj[vi].append(ui)
        edge_to_id[(ui, vi)] = idx
        id_to_edge[idx] = (ui, vi)

    num_nodes = len(nodes)
    num_edges = len(edges)
    target_rank = num_edges - num_nodes + 1
    
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Est. Target Rank: {target_rank}")
    
    # 3. Generate Candidates
    all_candidates = []
    
    # 3a. Fast Triangle Extraction
    print("Extracting Triangles...")
    triangles, covered_triangle_edges = find_triangles(adj, edge_to_id)
    print(f"Found {len(triangles)} triangles. Covered {len(covered_triangle_edges)} edges.")
    all_candidates.extend(triangles)

    print("Extracting Squares...")
    squares, covered_square_edges = find_squares(adj, edge_to_id)
    print(f"Found {len(squares)} squares. Covered {len(covered_square_edges)} edges.")
    all_candidates.extend(squares)

    covered_edges = covered_triangle_edges.union(covered_square_edges)
    
    processed = 0
    t_scan = time.time()
    
    # Process edges
    # Staged coverage: C5 then C6 for uncovered
    
    # Stage 3: Scan for 5-cycles (Path len 4)
    print("Scanning for 5-cycles (len 5)...")
    scanned_5 = 0
    skipped_5 = 0
    
    for (u, v) in edges:
        eid_uv = edge_to_id.get((u, v))
        if eid_uv in covered_edges:
            skipped_5 += 1
            continue
            
        # BFS with limit 4 (Cycle len 5)
        # Moderate visited limit
        cycles = bfs_with_detours(u, v, adj, edge_to_id, visited_limit=150, detours=2, max_path_len=4)
        
        found_5 = False
        if cycles:
            for c in cycles:
                if len(c) <= 5:
                     all_candidates.append(c)
                     found_5 = True
                     # Greedily cover this edge and potentially others
                     covered_edges.update(c) 
                     
        if found_5:
            scanned_5 += 1
            
    print(f"  > Found {scanned_5} edges yielding 5-cycles. Skipped {skipped_5} pre-covered.")

    # Stage 4: Scan for 6-cycles (Path len 5) or longer for remaining uncovered
    print("Scanning for 6-cycles (len 6) or fallback...")
    scanned_6 = 0
    skipped_6 = 0
    
    for (u, v) in edges:
        eid_uv = edge_to_id.get((u, v))
        if eid_uv in covered_edges:
            skipped_6 += 1
            continue
            
        # BFS with limit 5 (Cycle len 6) or slightly more
        # Increased visited limit for deeper search
        cycles = bfs_with_detours(u, v, adj, edge_to_id, visited_limit=300, detours=2, max_path_len=6)
        
        if cycles:
            all_candidates.extend(cycles)
            scanned_6 += 1
            # We don't strictly need to update covered_edges here as this is the last BFS stage,
            # but for consistency if we added more stages:
            for c in cycles:
                covered_edges.update(c)

    scan_dur = time.time() - t_scan
    print(f"Scan finished in {scan_dur:.2f}s. Scanned C5: {scanned_5}, Scanned C6+: {scanned_6}. Total Candidates: {len(all_candidates)}")
    
    # 3b. Add Fundamental Cycles (Ensures Full Rank)
    print("Generating Fundamental Cycles...")
    fund_cycles = get_fundamental_cycles_eids(adj, edge_to_id, num_nodes)
    print(f"Fundamental Cycles: {len(fund_cycles)}")
    all_candidates.extend(fund_cycles)
    
    # 4. Sort Candidates
    # Primary Key: Length
    print("Sorting candidates...")
    unique_candidates = list(set(all_candidates))
    unique_candidates.sort(key=lambda x: len(x))
    
    print(f"Unique Candidates: {len(unique_candidates)}")
    
    # 5. Build Basis
    basis_obj = LinearBasis(num_edges)
    final_cycles = []
    
    total_weight = 0
    for cyc in unique_candidates:
        if basis_obj.insert(cyc):
            final_cycles.append(cyc)
            total_weight += len(cyc)
            # If we reach target rank, we can stop?
            # if len(final_cycles) >= target_rank: break 
            # Ideally yes, but checking connectedness is safer.
                
    print(f"Basis Rank: {len(final_cycles)}, Weight: {total_weight}")
    
    # Calculate stats breakdown
    len_counts = defaultdict(int)
    for c in final_cycles:
        len_counts[len(c)] += 1
        
    print("Basis Cycle Length Breakdown:")
    for length in sorted(len_counts.keys()):
        count = len_counts[length]
        print(f"  Length {length}: {count} cycles (Subtotal: {length * count})")

    # 6. Output
    with open(output_file, 'w') as f:
        f.write(f"{len(final_cycles)}\n")
        
        for cyc_eids in final_cycles:
            sub_edges = [id_to_edge[eid] for eid in cyc_eids]
            if not sub_edges: 
                f.write("0\n")
                continue
                
            local_adj = defaultdict(list)
            for u, v in sub_edges:
                local_adj[u].append(v)
                local_adj[v].append(u)
            
            # Walk cycle
            # Start at a node with degree 2 (should be all)
            start_node = next(iter(local_adj))
            path = [start_node]
            
            curr = start_node
            prev = None
            
            # We traverse len(sub_edges) times to get full sequence
            for _ in range(len(sub_edges)):
                 nbs = local_adj[curr]
                 # Pick neighbor != prev
                 next_n = None
                 for n in nbs:
                     if n != prev:
                         next_n = n
                         break
                 if next_n is None and nbs: next_n = nbs[0] # Should not happen
                 
                 path.append(next_n)
                 prev = curr
                 curr = next_n
                 
            # Output node sequence (exclude last which duplicates start)
            out_nodes = path[:-1]
            out_strs = [str(inv_node_map[n]) for n in out_nodes]
            f.write(f"{len(out_strs)} {' '.join(out_strs)}\n")

    print(f"Total Time: {time.time() - t0:.4f}s")

if __name__ == "__main__":
    main()
