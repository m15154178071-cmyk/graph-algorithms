from __future__ import annotations

import os
import re
import builtins
from datetime import datetime
import sys
import array
import math
import time
import itertools
from collections import defaultdict, Counter, deque
from typing import Dict, Set, Tuple, List, FrozenSet, Optional, Iterable, Generator, Mapping, Callable, Any, DefaultDict
import threading
import platform
import ctypes
try:
    from ctypes import wintypes
except ImportError:
    # Linux/Mac 上没有 wintypes，忽略即可
    wintypes = None
import argparse
from dataclasses import dataclass, fields


dict_str1_int2 = Dict[str, int]
dict_str2_int1 = Dict[str, int]
dict_str1_int2 = {
    "0": 10,
    "1": 11,
    "2": 20,
    "3": 21,
    "4": 30,
    "5": 31,
    "6": 40,
    "7": 41,
    "8": 50,
    "9": 51,
}

dict_str2_int1 = {
    "10": 0,
    "11": 1,
    "20": 2,
    "21": 3,
    "30": 4,
    "31": 5,
    "40": 6,
    "41": 7,
    "50": 8,
    "51": 9,
}

class DataInitialization:
    def __init__(self, input_file: str, dict_str1_int2: Dict[str, int], target_rank: int = -1, comb_max: int = -1, debug: bool = False):
        self.input_file = input_file
        self.dict_str1_int2 = dict_str1_int2
        self.lines = [(0, 0)]
        self.lines = self.input_file_lines(input_file)
        
        # Internal mappings (Initialize before use)
        self.nodes_str: Set[str] = set()
        
        self.mapping: Dict[str, int] = {}
        self.mapping = self.get_all_nodes_int_mapping()
        self.target_rank = target_rank
        self.comb_max = comb_max
        self.debug = debug

        self.time_start = time.time()

        # Internal mappings
        self.nodes_str: Set[str] = set()
        self.unordered_edge_to_eid_dict: Dict[Tuple[int, int], int] = {}
        self.eid_to_unordered_edge_dict: List[Optional[Tuple[int, int]]] = []

        # For FHBackMapperFinal
        self.dict_edge_to_cycles: Dict[Tuple[int, int], List[frozenset[int]]] = defaultdict(list)
        self.dict_edge_to_cycles_deferred: Dict[Tuple[int, int], Tuple[Set, Set, Set, Set]] = defaultdict(lambda: (set(), set(), set(), set()))
        self.dfs_dirs_by_edge_deferred: Dict[Tuple[int, int], Tuple] = {}
        
        self.duration = time.time() - self.time_start

    def input_file_lines(self, input_file: str) -> List[Tuple[int, int]]:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
        
        parts_list = [parts for line in raw_lines for parts in [re.split(r"[,\s]+", line)] if len(parts) >= 2]
        
        lines: List[Tuple[int, int]] = []
        for parts in parts_list:
            try:
                u = int(parts[0])
                v = int(parts[1])
                u = min(u, v)
                v = max(v, u)
                lines.append((u, v))
            except ValueError:
                lines.append((0, 0))
        return lines
    
    def get_all_nodes_zfillstr(self) -> Set[str]:
        if not self.nodes_str:
            max_len = 0
            for u, v in self.lines:
                max_len = max(max_len, len(str(u)), len(str(v)))

            for u, v in self.lines:
                self.nodes_str.add(str(u).zfill(max_len))
                self.nodes_str.add(str(v).zfill(max_len))
        return self.nodes_str
    
    def get_all_nodes_int_mapping(self) -> Dict[str, int]:
        nodes = self.get_all_nodes_zfillstr()
        mapping = {}
        for node in nodes:
            parts = [str(self.dict_str1_int2[c]) for c in node]
            big_int_str = "".join(parts)
            mapping[node] = int(big_int_str)
        return mapping

    def build_edge_mappings(self):
        mapping = self.get_all_nodes_int_mapping()
        
        max_len = 0
        for u, v in self.lines:
            max_len = max(max_len, len(str(u)), len(str(v)))

        for u_raw, v_raw in self.lines:
            u_str = str(u_raw).zfill(max_len)
            v_str = str(v_raw).zfill(max_len)
            
            u_int = mapping[u_str]
            v_int = mapping[v_str]
            
            u_mapped = min(u_int, v_int)
            v_mapped = max(u_int, v_int)
            
            edge_tuple = (u_mapped, v_mapped)
            if edge_tuple not in self.unordered_edge_to_eid_dict:
                eid = len(self.eid_to_unordered_edge_dict)
                self.unordered_edge_to_eid_dict[edge_tuple] = eid
                self.eid_to_unordered_edge_dict.append(edge_tuple)
        return self.unordered_edge_to_eid_dict, self.eid_to_unordered_edge_dict


class GraphCycleFinder:
    def __init__(self, lines: List[Tuple[int, int]] = [(0, 0)], target_rank: int = -1, comb_max: int = -1, debug: bool = False):
        t_start = time.time()
        self.lines = lines
        self.target_rank = target_rank
        self.comb_max = comb_max
        self.debug = debug
        self.reducer_debug = False
        
        # 1. 建立邻接表 和 无向边集合
        self.adj: Dict[int, List[int]] = defaultdict(list)
        self.edges: Set[Tuple[int, int]] = set()
        self._build_adj()
        
        # 1.5. 剪枝：移除度小于2的节点 (2-core)
        self._prune_low_degree_nodes()
        
        # 2. 统计节点度与分层
        self.node_degrees: Dict[int, int] = {}
        self.degree_to_nodes: Dict[int, List[int]] = defaultdict(list)
        self._build_degree_info()

        # 3. 动态扩展F和H summary
        # F: Maps packed_pair(u, v) -> List[base_node_id] (path length 2)
        # 记录通过度为2的节点压缩的路径信息
        self.pair_to_base_eids_F: Dict[int, List[int]] = defaultdict(list)
        
        # 缓存
        self._cached_triangles = None
        self._cached_triangle_nodes = None
        self._cached_isolated_nodes = None
        self._cached_squares = None
        
        # H: Maps packed_pair(u, v) -> List[packed_pair(a, b)] (path length 3)
        # 记录更复杂的路径压缩信息
        self.pair_to_base_eids_H: Dict[int, List[int]] = defaultdict(list)
        # 记录扩展路径
        self.pair_to_more_nodes: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        
        # -------- Enhanced Cycle Enumeration Structures --------
        self.E_SUMMARY = set()
        self.F_SUMMARY = set()
        self.G_SUMMARY = set()
        self.H_SUMMARY = set()
        self.dfs_dirs_by_edge = {}
        self.heavy_edge_threshold = 64
        self.heavy_edge_count = 0
        
        # Will be initialized in enumerate_cycles
        self.adjacency_map = None
        self.undirected_edge2_single_str = []
        self.edge_index_map = {}
        # Also used for enhanced cycle storage override
        self.dict_edge_to_cycles_enhanced = {} 
        self.dict_edge_to_cycles: Dict[Tuple[int, int], Tuple[Set, Set, Set, Set]] = defaultdict(lambda: (set(), set(), set(), set()))
        self.dict_edge_to_cycles_deferred: Dict[Tuple[int, int], Tuple[Set, Set, Set, Set]] = defaultdict(lambda: (set(), set(), set(), set()))
        self.dfs_dirs_by_edge_deferred: Dict[Tuple[int, int], Tuple] = {}
        
        self.pending_area_edges: Set[Tuple[int, int]] = set()
        self.bench_area_edges: Set[Tuple[int, int]] = set()
        
        # Linear Independence Verification
        self.independent_structural_cycles: List[FrozenSet[int]] = []
        self.structural_independence_verified = False

        self.init_duration = time.time() - t_start
        self.phase_times = {}

        # 缓存三环信息
        self._cached_triangles: Optional[List[Tuple[int, ...]]] = None
        self._cached_triangle_nodes: Optional[Set[int]] = None
        self._cached_squares: Optional[List[Tuple[int, ...]]] = None

        self._cached_isolated_nodes: Optional[Set[int]] = None

    @staticmethod
    def _generate_four_five_cycles(
        unordered_edges: Set[Tuple[int, int]],
        adj_node_node: Dict[int, Set[int]]
    ) -> Tuple[Set[Tuple[int, ...]], Set[Tuple[int, ...]]]:
        """
        生成4节点和5节点环
        """
        four_cycles_nodes_set = set()
        five_cycles_nodes_set = set()

        for (u, v) in unordered_edges:
            neighbors_u = adj_node_node[u]
            neighbors_v = adj_node_node[v]
            common_neighbors = neighbors_u & neighbors_v - {u, v}
            unique_neighbors_u = neighbors_u - {u, v} - common_neighbors
            unique_neighbors_v = neighbors_v - {u, v} - common_neighbors
            two_nodes_producs = itertools.product(unique_neighbors_u, unique_neighbors_v)
            two_nodes_set = set()
            for two_nodes in two_nodes_producs:
                two_nodes_set.add((min(two_nodes), max(two_nodes)))
            edges = two_nodes_set & unordered_edges
            nonedges = two_nodes_set - edges
            for edge in edges:
                four_cycles_nodes_set.add(tuple(sorted([u, v] + list(edge))))
            for nonedge in nonedges:
                nodes_closed = adj_node_node[nonedge[0]] & adj_node_node[nonedge[1]] - (neighbors_u | neighbors_v)
                for node_closed in nodes_closed:
                    five_cycles_nodes_set.add(tuple(sorted([u, v] + list(nonedge) + [node_closed])))
        
        return four_cycles_nodes_set, five_cycles_nodes_set

    @staticmethod
    def _generate_six_cycles(
        three_cycles_endpoints_set: Set[Tuple[int, ...]],
        adj_endpoints_midpoint: Dict[Tuple[int, ...], Set[Tuple[int, ...]]],
        endpoints_midpoints_dict: Dict[Tuple[int, ...], Set[Tuple[int, ...]]],
    ) -> Generator[Tuple[int, ...], None, None]:
        """
        生成器：寻找所有6节点环
        避免一次性计算导致笛卡尔积爆炸
        """
        # 预先获取 keys 的引用，避免在循环中重复查询
        midpoint_keys = set(adj_endpoints_midpoint.keys())
        
        for (u, v, w) in three_cycles_endpoints_set:
            # 检查边是否存在
            edge1 = (min(u, v), max(u, v))
            edge2 = (min(u, w), max(u, w))
            edge3 = (min(v, w), max(v, w))
            
            # 只有当这三条"边"（其实是路径端点对）都在 adj_endpoints_midpoint 中存在时
            # 才有可能构成由三个2-hop路径组成的6环
            if edge1 not in midpoint_keys or edge2 not in midpoint_keys or edge3 not in midpoint_keys:
                continue

            part1 = {tuple(sorted(sorted(nodes) + sorted(edge1))) for nodes in adj_endpoints_midpoint[edge1]}
            part2 = {tuple(sorted(sorted(nodes) + sorted(edge2))) for nodes in adj_endpoints_midpoint[edge2]}
            part3 = {tuple(sorted(sorted(nodes) + sorted(edge3))) for nodes in adj_endpoints_midpoint[edge3]}


            # 优化：两阶段笛卡尔积 - 修正版
            list_part2 = list(part2)
            
            combined_part12 = []
            for p1 in part1:
                set_p1 = set(p1)
                for p2 in list_part2:
                    set_p2 = set(p2)
                    
                    # 合并节点: p1={u,m1,v}, p2={u,m2,w}. 共用 u
                    candidates = set_p1 | set_p2
                    
                    # 期望 {u, v, w, m1, m2} 共 5 个节点
                    if len(candidates) != len(set_p1) + len(set_p2) - 1:
                         continue
                         
                    combined_part12.append(candidates)
            
            # 第二阶段
            for c12 in combined_part12:
                # c12 是 Set. p3 是 tuple/set {v, m3, w}
                for p3 in part3:
                    set_p3 = set(p3)
                    final_nodes = c12 | set_p3
                    
                    # 期望 {u, v, w, m1, m2, m3} 共 6 个节点
                    if len(final_nodes) != len(c12) + len(set_p3) - 2:
                        continue
                    
                    yield tuple(sorted(final_nodes))

    def _build_adj(self):
        for u, v in self.lines:
            if u == v:
                continue
            # 保证无向边存储顺序 (u < v)
            if u > v:
                u, v = v, u
            
            if (u, v) not in self.edges:
                self.edges.add((u, v))
                self.adj[u].append(v)
                self.adj[v].append(u)

    def _prune_low_degree_nodes(self):
        """递归移除度小于2的节点，只保留2-core"""
        degrees = {u: len(nbs) for u, nbs in self.adj.items()}
        # Use simple list as stack for peeling - Sequence irrelevant for K-Core
        stack = [u for u, d in degrees.items() if d < 2]
        
        pruned_count = 0
        while stack:
            u = stack.pop()
            if u not in self.adj:
                continue
            
            neighbors = self.adj[u]
            del self.adj[u]
            pruned_count += 1
            
            for v in neighbors:
                if v in self.adj:
                    try:
                        self.adj[v].remove(u)
                        degrees[v] -= 1
                        if degrees[v] == 1:
                            stack.append(v)
                    except ValueError:
                        pass
        
        if self.debug and pruned_count > 0:
            print(f"[DEBUG] Pruned {pruned_count} low-degree nodes (kept 2-core only).")

    def _build_degree_info(self):
        for node, neighbors in self.adj.items():
            deg = len(neighbors)
            self.node_degrees[node] = deg
            self.degree_to_nodes[deg].append(node)

    def find_simple_triangles(self) -> List[Tuple[int, ...]]:
        """
        以边为单位，寻找所有三环 (u, v, w)。
        Uses common neighbor intersection instead of general enumeration.
        """
        if self._cached_triangles is not None:
            return self._cached_triangles

        triangles = []
        # Use set intersection for speed
        adj_set = {u: set(nbs) for u, nbs in self.adj.items()}
        
        # To avoid duplicates (u, v, w), enforce u < v < w
        # Iterate u < v in edges
        # Common w > v
        
        sorted_nodes = sorted(self.adj.keys())
        for u in sorted_nodes:
            nbs_u = adj_set[u]
            for v in nbs_u:
                if v <= u: continue # enforce u < v
                
                nbs_v = adj_set[v]
                # Intersection
                common = nbs_u & nbs_v
                
                for w in common:
                    if w <= v: continue # enforce v < w
                    # Found u < v < w
                    # Return can be any order, let's enable (u, v, w)
                    triangles.append((u, v, w))

        if self.debug:
            print(f"[调试] 发现 {len(triangles)} 个简单三角形 (Optimized Intersect).")
        
        self._cached_triangles = triangles
        return triangles

    def _recover_cycle_path_from_nodes(self, nodes: Tuple[int, ...]) -> Optional[List[int]]:
        """
        Given a set of nodes (structure unknown), reconstruct the cycle path order.
        Simple greedy DFS/BFS within the subgraph induced by 'nodes'.
        """
        node_set = set(nodes)
        if not node_set: return None
        
        # Build induced subgraph adjacency
        local_adj = defaultdict(list)
        for u in node_set:
            for v in self.adj[u]:
                if v in node_set:
                    local_adj[u].append(v)
        
        # Check degrees (must be >= 2 for a cycle)
        # For simple 4/5/6 cycles, degrees shoud be exactly 2
        start_node = nodes[0]
        
        path = [start_node]
        visited = {start_node}
        
        def dfs(curr, target_len):
            if len(path) == target_len:
                # Check closing
                if path[0] in local_adj[curr]:
                    return True
                return False
            
            for nxt in local_adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    path.append(nxt)
                    if dfs(nxt, target_len):
                        return True
                    path.pop()
                    visited.remove(nxt)
            return False

        if dfs(start_node, len(nodes)):
            return path
        return None

    def find_simple_squares_and_pentagons(self) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
        """
        Uses _generate_four_five_cycles to find 4-cycles and 5-cycles.
        Returns lists of cycle nodes (sorted tuples).
        """
        # Convert adj to set for faster lookup in static method
        adj_node_node_set = {k: set(v) for k, v in self.adj.items()}
        
        # These return sorted TUPLES of nodes (sets), not paths
        four_set, five_set = self._generate_four_five_cycles(self.edges, adj_node_node_set)
        
        # Recover paths
        four_paths = []
        for nodes in four_set:
            path = self._recover_cycle_path_from_nodes(nodes)
            if path:
                four_paths.append(tuple(path))
                
        five_paths = []
        for nodes in five_set:
            path = self._recover_cycle_path_from_nodes(nodes)
            if path:
                five_paths.append(tuple(path))
                
        return four_paths, five_paths


    def _reconstruct_cycle_nodes(self, cyc_eids: FrozenSet[int]) -> Tuple[int, ...]:
        """Helper to reconstruct node order from edge IDs."""
        if not cyc_eids: return ()
        
        # Build local adj
        adj_local = defaultdict(set)
        for eid in cyc_eids:
             if eid < 0 or eid >= len(self.undirected_edge2_single_str): continue
             uv = self.undirected_edge2_single_str[eid]
             if uv is None: continue
             u, v = uv
             adj_local[u].add(v)
             adj_local[v].add(u)
        
        if not adj_local: return ()
        
        # Valid cycle check: all degree 2
        for d in adj_local.values():
            if len(d) != 2: return ()

        start_node = min(adj_local.keys())
        path = [start_node]
        visited = {start_node}
        curr = start_node
        
        # Traverse
        while len(path) < len(cyc_eids):
            neighbors = adj_local[curr]
            next_node = None
            for nb in neighbors:
                if nb not in visited:
                    next_node = nb
                    break
            
            # If no unvisited neighbor, check if closed with start_node (last step)
            if next_node is None:
                 # Check if start_node is neighbor of curr, and length is sufficient (already checked by loop condition vs len)
                 # Actually if len(path) == len(cyc_eids), we are done, but loop would terminate.
                 # If we are stuck with len < total, it's not a simple cycle.
                 return ()

            path.append(next_node)
            visited.add(next_node)
            curr = next_node

        # Final check: curr should be connected to start_node
        if start_node not in adj_local[curr]:
            return ()
            
        return tuple(path)

    def get_triangles_and_nodes(self) -> Tuple[List[Tuple[int, ...]], Set[int]]:
        """
        获取所有三环及其涉及的节点。
        用于后续剪枝 (skip这些节点)，但不物理修改邻接表结构，保持图的完整性。
        """
        if self._cached_triangles is not None and self._cached_triangle_nodes is not None:
            return self._cached_triangles, self._cached_triangle_nodes

        triangles = self.find_simple_triangles()
        triangle_nodes = set()
        for tri in triangles:
            triangle_nodes.update(tri)
        
        if self.debug and triangle_nodes:
            print(f"[DEBUG] Identified {len(triangles)} triangles involving {len(triangle_nodes)} nodes for pruning.")

        self._cached_triangle_nodes = triangle_nodes
        return triangles, triangle_nodes

    def get_non_triangle_nodes(self) -> Set[int]:
        """
        统计不包括在任何三环中的节点。
        """
        _, triangle_nodes = self.get_triangles_and_nodes()
        all_nodes = set(self.adj.keys())
        non_triangle_nodes = all_nodes - triangle_nodes
        
        if self.debug:
            print(f"[DEBUG] Found {len(non_triangle_nodes)} nodes NOT in any triangle.")
            
        return non_triangle_nodes

    def get_nodes_isolated_from_triangles(self) -> Set[int]:
        """
        [Shortest Path Priority]
        统计那些连邻接表（邻居）里都不包含三环节点的“绝对孤立”节点。
        这些节点远离图中的密集三角结构，是计算最短路径时的“骨架”或“树枝”部分，
        在寻路算法中具有极高的优先级（或特殊处理价值）。
        """
        if self._cached_isolated_nodes is not None:
            return self._cached_isolated_nodes

        _, triangle_nodes = self.get_triangles_and_nodes()
        all_nodes = set(self.adj.keys())
        non_triangle_nodes = all_nodes - triangle_nodes
        
        isolated_nodes = set()
        for u in non_triangle_nodes:
            # 检查 u 的邻居是否有三环节点
            connects_to_triangle = False
            if u in self.adj:
                for v in self.adj[u]:
                    if v in triangle_nodes:
                        connects_to_triangle = True
                        break
            
            if not connects_to_triangle:
                isolated_nodes.add(u)
                
        if self.debug:
            print(f"[DEBUG] [Shortest Path Priority] Found {len(isolated_nodes)} nodes completely isolated from triangles.")
        
        self._cached_isolated_nodes = isolated_nodes
        return isolated_nodes

    def categorize_edges_by_isolation(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        基于“边缘节点”（远离三环的节点），将所有边分为三类：
        1. Both Isolated: 两个端点都在边缘节点集合中（骨架/树枝内部的边）。
        2. One Isolated: 只有一个端点在边缘节点集合中（连接骨架与密集区域的桥梁）。
        3. None Isolated: 没有端点在边缘节点集合中（密集区域内部的边）。
        """
        isolated_nodes = self.get_nodes_isolated_from_triangles()
        
        edges_both_isolated = []
        edges_one_isolated = []
        edges_none_isolated = []
        
        for u, v in self.edges:
            u_iso = u in isolated_nodes
            v_iso = v in isolated_nodes
            
            if u_iso and v_iso:
                edges_both_isolated.append((u, v))
            elif u_iso or v_iso:
                edges_one_isolated.append((u, v))
            else:
                edges_none_isolated.append((u, v))
                
        if self.debug:
            print(f"[DEBUG] Edge Categorization:")
            print(f"  - Both Isolated: {len(edges_both_isolated)} (Skeleton internal)")
            print(f"  - One Isolated:  {len(edges_one_isolated)} (Bridge)")
            print(f"  - None Isolated: {len(edges_none_isolated)} (Core internal)")
            
        return edges_both_isolated, edges_one_isolated, edges_none_isolated

    def find_simple_squares(self) -> List[Tuple[int, ...]]:
        """
        枚举所有闭合的简单四环 (squares)。
        """
        if self._cached_squares is not None:
            return self._cached_squares

        # Use optimized enumeration from find_simple_squares_and_pentagons
        squares, _ = self.find_simple_squares_and_pentagons()
        
        if self.debug:
            print(f"[DEBUG] Found {len(squares)} simple squares (Optimized).")
            
        self._cached_squares = squares
        return squares

    def _dfs_find_c_cycles_upto(
        self,
        *,
        a: int,
        b: int,
        m1: int,
        m2: int,
        max_len: int,
        limit: int = 1,
    ) -> List[FrozenSet[int]]:
        forbidden = {a, b, m1, m2}
        result: List[FrozenSet[int]] = []

        max_depth = max_len - 4
        if max_depth < 2:
            return result

        def dfs(path: List[int], visited: Set[int]):
            depth = len(path)
            last = path[-1]

            if 2 <= depth <= max_depth:
                # Path is m1 -> ... -> last. 
                # Check if last connects to m2.
                if m2 in self.adj.get(last, []):
                    # Found path: a - m1 - ... - last - m2 - b - a
                    nodes = {a, b, m1, m2, *path}
                    # Simple cycle check: should have len == depth + 4
                    # Induced cycle check: edges == nodes
                    if len(nodes) != depth + 4:
                        return

                    eids: List[int] = []
                    # Check edges for induced property & collect eids
                    # We are essentially doing induced check here
                    # For cycle to be induced, number of edges between these nodes must be exactly len(nodes)
                    edge_count = 0
                    node_list = list(nodes)
                    n_nodes = len(node_list)
                    
                    # This check is O(k^2)
                    for i in range(n_nodes):
                        u = node_list[i]
                        for j in range(i + 1, n_nodes):
                            v = node_list[j]
                            eid = self.edge_index_map.get((min(u, v), max(u, v)))
                            if eid is not None:
                                edge_count += 1
                                eids.append(eid)
                    
                    if edge_count == n_nodes:
                        result.append(frozenset(eids))
                        if len(result) >= limit:
                            raise StopIteration

            if depth >= max_depth:
                return

            for nxt in self.adj.get(last, []):
                if nxt in visited or nxt in forbidden:
                    continue
                # Optimization: Prune if nxt is neighbor of a or b (unless it's m1/m2 which are forbidden)
                # To ensure chordless with a-m1 and b-m2 edges?
                # The path is strictly between m1 and m2.
                # If nxt connects to a, then a-m1-nxt-a is C3, already found.
                # So we can prune if nxt in (adj[a] | adj[b]) generally.
                if nxt in self.adj[a] or nxt in self.adj[b]:
                    continue
                    
                dfs(path + [nxt], visited | {nxt})

        try:
            for x in self.adj.get(m1, []):
                if x in forbidden:
                    continue
                # Same pruning
                if x in self.adj[a] or x in self.adj[b]:
                    continue
                dfs([x], {m1, x})
        except StopIteration:
            pass

        return result

    def find_simple_pentagons(self) -> List[Tuple[int, ...]]:
        """
        查找简单的五环 (5-cycles)
        """
        # Use optimized enumeration
        if not hasattr(self, 'cycles_by_len') or 5 not in self.cycles_by_len:
             self.cycles_by_len = self.enumerate_cycles(induced_only=True)
             
        pentagons = []
        if 5 in self.cycles_by_len:
            for cyc_eids in self.cycles_by_len[5]:
                nodes = self._reconstruct_cycle_nodes(cyc_eids)
                if nodes and len(nodes) == 5:
                     pentagons.append(nodes)
        
        if self.debug:
            print(f"[DEBUG] Found {len(pentagons)} simple pentagons (Optimized).")
            
        return pentagons

    def find_simple_hexagons(self) -> List[Tuple[int, ...]]:
        """
        查找简单的六环 (6-cycles)
        Structure: u-x-w-z-y-v-u
        Method: 
          For edge (u, v):
            Find x in N(u), y in N(v) (disjoint from v, u)
            Find w in N(x), z in N(y)
            Check if edge (w, z) exists.
        Optimization: Use DFS Guided by F/H summaries.
        """
        t0 = time.time()
        
        # Ensure dfs dirs are ready (by calling enumerate_cycles)
        if not hasattr(self, 'dfs_dirs_by_edge') or not self.dfs_dirs_by_edge:
             self.enumerate_cycles(induced_only=True)
        
        hexagons_set = set() # Avoid duplicates: same cycle from multiple edges
        hexagons = []
        
        # Iterate all edges with guidance
        for (u, v), dirs in self.dfs_dirs_by_edge.items():
            for (m1, m2) in dirs:
                # Search for path m1 -> ... -> m2 of correct length
                # Since we want C6, edge u-v is 1. path m1..m2 is length 3.
                # structure: u-m1-x-m2-v. 5 nodes. No.
                # C6: u - m1 - x - y - m2 - v - u
                # Path len between m1 and m2 should be 2. (m1-x-y-m2 is 3 edges?)
                # Wait. DFS depth.
                # dfs(path, visited). path starts with neighbor of m1.
                # dfs([x], {m1, x}). Depth 1.
                # max_depth = max_len - 4.
                # For C6: max_depth = 6 - 4 = 2.
                # dfs depth 2. path = [x, y].
                # C6: u, v, m1, m2, x, y. Total 6 nodes.
                
                # Check method signature: dfs_find_c_cycles_upto
                found_cyc_eids_list = self._dfs_find_c_cycles_upto(
                    a=u, b=v, m1=m1, m2=m2, max_len=6, limit=100
                )
                
                for cyc_eids in found_cyc_eids_list:
                     if len(cyc_eids) == 6:
                         nodes = self._reconstruct_cycle_nodes(cyc_eids)
                         if nodes:
                             # Canonical representation for dedup
                             min_node = min(nodes)
                             idx = nodes.index(min_node)
                             # Rotate and check flip to normalize
                             rotated = nodes[idx:] + nodes[:idx]
                             if rotated[1] > rotated[-1]:
                                 s = tuple(rotated[::-1]) # This puts min at end. Rotate again.
                                 # Standard canonical: min first, then smaller neighbor second.
                                 # simpler: frozenset of nodes? No, order matters.
                                 # Start with min. Second is min(neighbors).
                                 # But here simple tuple is enough if consistent.
                                 # Just use sorted tuple for set checking? No cycle is sequence.
                                 pass
                             
                             # Simple dedup strategy: transform to frozenset of edges (already have eids)
                             if cyc_eids not in hexagons_set:
                                 hexagons_set.add(cyc_eids)
                                 hexagons.append(nodes)

        if self.debug:
            elapsed = time.time() - t0
            print(f"[DEBUG] Found {len(hexagons)} simple hexagons in {elapsed:.4f}s (Guided DFS).")
        
        return hexagons

    def analyze_squares_isolation(self):
        """
        调查每一个四环中：
        1. 每一个点是否在边缘节点(isolated_nodes)中。
        2. 每一个点的邻接点中是否存在边缘节点。
        """
        squares = self.find_simple_squares()
        isolated_nodes = self.get_nodes_isolated_from_triangles()
        
        # 统计数据
        # count_node_is_iso[k]: 一个四环中有 k 个点是边缘节点
        count_node_is_iso = defaultdict(int) 
        
        # count_node_has_iso_neighbor[k]: 一个四环中有 k 个点拥有边缘节点邻居
        count_node_has_iso_neighbor = defaultdict(int)

        if self.debug:
            print(f"[DEBUG] Analyzing {len(squares)} squares for isolation properties...")

        for sq in squares:
            nodes = list(sq)
            
            # 1. Check if node itself is isolated
            is_iso_list = [node in isolated_nodes for node in nodes]
            num_iso = sum(is_iso_list)
            count_node_is_iso[num_iso] += 1
            
            # 2. Check connections to isolated nodes
            has_iso_nb_list = []
            for node in nodes:
                has_iso = False
                # 如果节点本身是 isolated，那它肯定“接触”isolated区域（它自己就是）
                # 这里既然问“邻接点中是否存在”，通常指外部邻居，也可以包含自己如果逻辑需要。
                # 但严格来说“邻接点”指 neighbors。
                # 注意：如果 u 是 isolated，v是u的邻居，那么 v 有一个 isolated neighbor (u)。
                for nb in self.adj[node]:
                    if nb in isolated_nodes:
                        has_iso = True
                        break
                has_iso_nb_list.append(has_iso)
            
            num_has_iso_nb = sum(has_iso_nb_list)
            count_node_has_iso_neighbor[num_has_iso_nb] += 1
            
        if self.debug:
            print("[DEBUG] Square Isolation Analysis:")
            for k in sorted(count_node_is_iso.keys()):
                print(f"  - Squares with {k} isolated nodes: {count_node_is_iso[k]}")
                if k == 4:
                    print(f"    -> [IMPORTANT] {count_node_is_iso[k]} squares are FULLY within the Absolute Core Skeleton (Absolute Core Squares).")
            
            print("[DEBUG] Square Neighbor Connectivity:")
            for k in sorted(count_node_has_iso_neighbor.keys()):
                print(f"  - Squares with {k} nodes having isolated neighbors: {count_node_has_iso_neighbor[k]}")
            
        return count_node_is_iso, count_node_has_iso_neighbor

    def get_structural_core_nodes(self) -> Set[int]:
        """
        获取“骨架主体”节点集合。
        定义为：
        1. 三环内的所有节点 (Triangle Nodes)。
        2. “绝对核心骨架” (Isolated Nodes) 中，构成闭合四环的节点 (Square Core Skeleton Nodes)。
           (即那些完全由 Isolated Nodes 构成的四环中的所有节点)
        
        这两部分的并集构成了图中拥有最密集环结构和骨架环结构的主体部分。
        """
        # 1. 获取三环节点
        _, triangle_nodes = self.get_triangles_and_nodes()
        
        # 2. 获取绝对核心骨架节点
        isolated_nodes = self.get_nodes_isolated_from_triangles()
        
        # 3. 获取所有四环
        squares = self.find_simple_squares()
        
        # 4. 筛选出完全位于骨架内的四环节点
        skeleton_square_nodes = set()
        count_skeleton_squares = 0
        
        for sq in squares:
            # 检查四环的所有节点是否都在孤立集合中
            if all(node in isolated_nodes for node in sq):
                skeleton_square_nodes.update(sq)
                count_skeleton_squares += 1
                
        # 5. 合并
        main_body_nodes = triangle_nodes.union(skeleton_square_nodes)
        
        if self.debug:
            print(f"[DEBUG] Structural Core Analysis:")
            print(f"  - Triangle Nodes: {len(triangle_nodes)}")
            print(f"  - Skeleton Square Nodes: {len(skeleton_square_nodes)} (from {count_skeleton_squares} Absolute Core squares)")
            print(f"  - Combined Structural Body: {len(main_body_nodes)} nodes ({len(main_body_nodes) / len(self.adj) * 100:.1f}% of total 2-core graph)")
            
        return main_body_nodes
    
    def get_core_non_edge_pairs(self) -> List[Tuple[int, int]]:
        """
        在“骨架主体”（结构核心）节点集合中，进行两两全组合，找出**不直接通过边相连**的点对。
        根据理论，这些点对是寻找 5-环 和 6-环 的核心骨架 (Anchors)。
        即：如果在核心结构中两点不直接相连，它们之间的连接路径（配合另一条路径）倾向于构成大于4的长周期环。
        """
        core_nodes = list(self.get_structural_core_nodes())
        core_nodes.sort() # Ensure deterministic order for reproducibility
        
        non_edge_pairs = []
        
        # 预先获取邻接集合，只关注核心节点即可，但查全图adj也没问题
        # 假设 adj 是完整的
        
        n = len(core_nodes)
        if self.debug:
            print(f"[DEBUG] Generating non-edge pairs for {n} core nodes...")

        # Optimization: Limit combinatorial explosion
        # If too many core nodes, generating pairs is O(N^2) and meaningless/expensive
        if n > 40:
             if self.debug:
                 print(f"[DEBUG] Core nodes count {n} > 40. Skipping pairwise generation to avoid O(N^2) explosion.")
             return non_edge_pairs
            
        # 遍历组合
        for i in range(n):
            u = core_nodes[i]
            # u 的邻居集合
            u_nbs = set(self.adj[u])
            
            for j in range(i + 1, n):
                v = core_nodes[j]
                
                # 如果 v 不是 u 的邻居，则记录
                if v not in u_nbs:
                    non_edge_pairs.append((u, v))
        
        if self.debug:
            print(f"[DEBUG] Identified {len(non_edge_pairs)} non-edge pairs within the Structural Core.")
            print(f"        These pairs serve as potential anchors for 5-cycles and 6-cycles.")
            
        return non_edge_pairs

    def get_secondary_skeleton_nodes(self) -> Set[int]:
        """
        获取“次骨架点”集合 (Optimized Node-Centric Approach)。
        
        逻辑：不再遍历 Core Pairs (O(N^2))，而是遍历所有非核心点 (O(V))。
        如果一个点 k (或其核心点本身) 能够连接到至少两个“非连边”的核心点 (距离<=2)：
        即存在 u, v in Core，dist(k, u)<=2, dist(k, v)<=2，且 (u, v) 不是边。
        
        优化算法：
        1. 找出 CoreNodes。
        2. 对每个非 Core 点 k，计算其 2 跳范围内能到达的 CoreNodes 集合 R2(k)。
           R2(k) = (N(k) & Core) U (Union_{x in N(k)} (N(x) & Core))
        3. 如果 |R2(k)| >= 2:
             检查 R2(k) 是否构成 Core 内部的 Clique (全连接)。
             如果不是 Clique -> 说明存在 non-edge pair -> k 是 Secondary。
        注意：Secondary Skeleton 包含 Core 本身。
        """
        core_nodes = self.get_structural_core_nodes()
        secondary_skeleton = set(core_nodes)  
        
        all_nodes = set(self.adj.keys())
        non_core_nodes = all_nodes - core_nodes
        
        if self.debug:
            print(f"[DEBUG] Calculating Secondary Skeleton (Optimized) from {len(non_core_nodes)} candidate nodes...")
        
        # 1. Precompute Neighbors in Core for all nodes (L1 connections)
        # map: node -> set of neighbors in Core
        neighbor_in_core = defaultdict(set)
        
        # Iterate Core nodes to populate neighbor_in_core
        # This works for both Core and Non-Core nodes
        for u in core_nodes:
            for v in self.adj[u]:
                neighbor_in_core[v].add(u)
                
        # 2. Helper to check clique
        def is_clique(nodes_subset):
            nodes_list = list(nodes_subset)
            n_sub = len(nodes_list)
            if n_sub < 2: return True
            expected_edges = n_sub * (n_sub - 1) // 2
            actual_edges = 0
            for i in range(n_sub):
                u = nodes_list[i]
                u_adj = set(self.adj[u])
                for j in range(i + 1, n_sub):
                    if nodes_list[j] in u_adj:
                        actual_edges += 1
            return actual_edges == expected_edges

        # 3. Iterate all non-core nodes
        candidates = list(non_core_nodes) # List for stability
        
        # Pre-fetch adjacency to avoid lookups if possible, but self.adj is fast dict
        
        for k in candidates:
            # reachable_core stores Core nodes within distance 2 of k
            reachable_core = set()
            
            # (a) Direct neighbors in Core (Dist 1)
            if k in neighbor_in_core:
                reachable_core.update(neighbor_in_core[k])
            
            # (b) Neighbors of Neighbors in Core (Dist 2)
            # k -> x -> u(Core). 
            k_neighbors = self.adj[k]
            for x in k_neighbors:
                if x in neighbor_in_core:
                    reachable_core.update(neighbor_in_core[x])
            
            # Check condition
            if len(reachable_core) >= 2:
                # If reachable_core form a Clique, then NO non-edge pair exists.
                # If NOT a Clique, then at least one non-edge pair exists.
                if not is_clique(reachable_core):
                    secondary_skeleton.add(k)
                    
        if self.debug:
            print(f"[DEBUG] Identified {len(secondary_skeleton)} Secondary Skeleton Nodes (Optimized).")
            
        return secondary_skeleton

    def analyze_secondary_skeleton_structure(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        对次骨架点 (Secondary Skeleton) 进行两两全组合分析：
        1. "是边的" -> 标记为 "5-cycle anchor edges" (5环骨架特征边)。
        2. "不是边的" -> 标记为 "6-cycle skeleton pairs" (6环骨架点对)。
        返回: (edges_5_cycle, non_edges_6_cycle)
        """
        nodes = list(self.get_secondary_skeleton_nodes())
        nodes.sort()
        
        edges_5_cycle = [] 
        non_edges_6_cycle = []
        
        n = len(nodes)
        if self.debug:
            print(f"[DEBUG] Analyzing structure of {n} Secondary Skeleton nodes pairwise...")
            
        for i in range(n):
            u = nodes[i]
            for j in range(i + 1, n):
                v = nodes[j]
                
                pair = (u, v)
                
                if pair in self.edges:
                    edges_5_cycle.append(pair)
                else:
                    non_edges_6_cycle.append(pair)
                    
        if self.debug:
            print(f"[DEBUG] Secondary Skeleton Pair Analysis:")
            print(f"  - Edges (5-cycle related): {len(edges_5_cycle)}")
            print(f"  - Non-Edges (6-cycle skeleton): {len(non_edges_6_cycle)}")
            
        return edges_5_cycle, non_edges_6_cycle

    def classify_all_nodes(self) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        将全图节点重新划分为三类：
        1. 核心骨架节点 (Structural Core Nodes): 三环节点 + 纯骨架四环节点
           - 对应图中的高密度区域和规则高速公路网，是最小环基的主体。
        
        2. 次骨架节点 (Secondary Skeleton Nodes - Pure): 
           - 位于 get_secondary_skeleton_nodes() 集合中，但不在核心骨架中的节点。
           - 这些是连接核心骨架的桥梁、延伸路径，是构建5环、6环的关键中间点。
           
        3. 边缘节点 (Peripheral Nodes):
           - 既不在核心骨架，也不在次骨架中的剩余节点。
           - 这些通常是图的末梢、长须或极其稀疏的连接部分。
           
        返回: (core_nodes, secondary_nodes, peripheral_nodes)
        """
        # 1. 获取核心骨架 (Set A)
        core_nodes = self.get_structural_core_nodes()
        
        # 2. 获取包含核心的宽泛次骨架 (Set B >= Set A)
        raw_secondary = self.get_secondary_skeleton_nodes()
        
        # 3. 计算纯次骨架 (B - A)
        secondary_nodes = raw_secondary - core_nodes
        
        # 4. 计算边缘节点 (Total - B)
        all_nodes = set(self.adj.keys())
        peripheral_nodes = all_nodes - raw_secondary
        
        if self.debug:
            total = len(all_nodes)
            p_core = len(core_nodes) / total * 100 if total > 0 else 0
            p_sec = len(secondary_nodes) / total * 100 if total > 0 else 0
            p_peri = len(peripheral_nodes) / total * 100 if total > 0 else 0
            
            print(f"[DEBUG] Final Node Classification:")
            print(f"  1. Structural Core:     {len(core_nodes)} ({p_core:.1f}%) [Dense & Square Skeleton]")
            print(f"  2. Secondary Skeleton:  {len(secondary_nodes)} ({p_sec:.1f}%) [Bridges & 5/6-Cycle Builders]")
            print(f"  3. Peripheral Nodes:    {len(peripheral_nodes)} ({p_peri:.1f}%) [Sparse Extremities]")
            
        return core_nodes, secondary_nodes, peripheral_nodes

    def _is_induced_cycle_nodes(self, nodes: Iterable[int]) -> bool:
        # Check if the subgraph induced by 'nodes' is a simple cycle (chordless)
        # For a cycle of length k, there should be exactly k edges in the induced subgraph.
        node_list = list(nodes)
        n = len(node_list)
        edge_count = 0
        
        # O(k^2) check, k is small (3, 4, 5)
        for i in range(n):
            u = node_list[i]
            if u not in self.adj: continue
            u_nbs = self.adj[u]
            for j in range(i + 1, n):
                v = node_list[j]
                if v in u_nbs:
                    edge_count += 1
        
        return edge_count == n

    # =====================================================
    # Enumerate C3/C4/C5 + Construct F/H + Build Guide Dict dfs_dirs_by_edge
    # =====================================================
    def enumerate_cycles(self, induced_only: bool = True) -> Dict[int, Set[FrozenSet[int]]]:
        # -------- Initialization / Sync --------
        if self.adjacency_map is None:
             valid_nodes = set(self.adj.keys())
             valid_edges = []
             for u in valid_nodes:
                 for v in self.adj[u]:
                     if u < v and v in valid_nodes:
                         valid_edges.append((u, v))
             valid_edges = sorted(list(set(valid_edges)))
             
             self.undirected_edge2_single_str = valid_edges
             self.edge_index_map = {edge: i for i, edge in enumerate(valid_edges)}
             self.adjacency_map = {u: set(self.adj[u]) for u in valid_nodes}

        cycles_by_len: Dict[int, Set[FrozenSet[int]]] = defaultdict(set)
        E2 = self.undirected_edge2_single_str
        edge_index = self.edge_index_map
        all_edges_set = set(edge_index.keys())

        # -------- Precompute Structure Cache (F/H etc.) --------
        for (a0, b0) in E2:
            a, b = (a0, b0) if a0 < b0 else (b0, a0)
            edge_set = {a, b}

            Na = self.adjacency_map.get(a, set()) - edge_set
            Nb = self.adjacency_map.get(b, set()) - edge_set

            common = Na & Nb
            only_a = Na - common
            only_b = Nb - common
            unique = only_a | only_b

            # -------- Heuristic: Skip expensive structure O(N^2) on Heavy Edges --------
            size_a, size_b = len(only_a), len(only_b)
            # Default threshold is 64. If 0, optimization is disabled.
            is_heavy = False
            if self.heavy_edge_threshold > 0:
                is_heavy = (size_a > self.heavy_edge_threshold) or (size_b > self.heavy_edge_threshold)

            if is_heavy:
                self.heavy_edge_count += 1
                self.dict_edge_to_cycles[(a, b)] = (set(), set(), set(), set()) 

                # Still check C3 as it is cheap
                # -------- C3: a-x-b-a --------
                for x in common:
                    cyc_nodes = {a, b, x}
                    if (not induced_only) or self._is_induced_cycle_nodes(cyc_nodes):
                        # Ensure edges exist in map
                        e1 = edge_index.get((min(a, b), max(a, b)))
                        e2 = edge_index.get((min(a, x), max(a, x)))
                        e3 = edge_index.get((min(b, x), max(b, x)))
                        if e1 is not None and e2 is not None and e3 is not None:
                             cycles_by_len[3].add(frozenset((e1, e2, e3)))
                continue

            B = set(itertools.combinations(sorted(only_a), 2))
            C = set(itertools.combinations(sorted(only_b), 2))
            E = (B | C) & all_edges_set

            F = (B | C) - E

            D = set(itertools.combinations(sorted(unique), 2))
            G = (D - B - C) & all_edges_set

            A = set(itertools.combinations(sorted(Na | Nb), 2))
            H = (A - B - C) - G

            self.dict_edge_to_cycles[(a, b)] = (E, F, G, H)
            self.E_SUMMARY |= E
            self.F_SUMMARY |= F
            self.G_SUMMARY |= G
            self.H_SUMMARY |= H

            # -------- C3: a-x-b-a --------
            for x in common:
                cyc_nodes = {a, b, x}
                if (not induced_only) or self._is_induced_cycle_nodes(cyc_nodes):
                    e1 = edge_index.get((min(a, b), max(a, b)))
                    e2 = edge_index.get((min(a, x), max(a, x)))
                    e3 = edge_index.get((min(b, x), max(b, x)))
                    if e1 is not None and e2 is not None and e3 is not None:
                        cycles_by_len[3].add(frozenset((e1, e2, e3)))

            # -------- C4: a-u-v-b-a (using G) --------
            for (u, v) in G:
                cyc_nodes = {a, b, u, v}
                if induced_only and (not self._is_induced_cycle_nodes(cyc_nodes)):
                    continue

                eids: List[int] = []
                ok = True
                pairs = [(min(x, y), max(x, y)) for x, y in ((a, u), (u, v), (v, b), (b, a))]
                for pair in pairs:
                    eid = edge_index.get(pair)
                    if eid is None:
                        ok = False
                        break
                    eids.append(eid)
                if ok and len(eids) == 4:
                    cycles_by_len[4].add(frozenset(eids))

            # -------- C5：a - node1 - mid - node2 - b - a --------
            for (node1, node2) in H:
                common2 = (self.adjacency_map.get(node1, set()) & self.adjacency_map.get(node2, set())) - edge_set
                if not common2:
                    continue
                for mid in common2:
                    cyc_nodes = {a, b, node1, node2, mid}
                    if induced_only and (not self._is_induced_cycle_nodes(cyc_nodes)):
                        continue

                    seq = (a, node1, mid, node2, b, a)
                    eids: List[int] = []
                    ok = True
                    for i in range(len(seq) - 1):
                        x, y = seq[i], seq[i+1]
                        pair = (min(x, y), max(x, y))
                        eid = edge_index.get(pair)
                        if eid is None:
                            ok = False
                            break
                        eids.append(eid)
                    if ok and len(eids) == 5:
                        cycles_by_len[5].add(frozenset(eids))

        # -------- 构造 DFS 指路字典 --------
        common_dirs = self.F_SUMMARY & self.H_SUMMARY
        for (a, b), (_, _, _, H) in self.dict_edge_to_cycles.items():
            dirs = tuple(H & common_dirs)
            if dirs:
                self.dfs_dirs_by_edge[(a, b)] = dirs

        return cycles_by_len

    def apply_structural_scoring_and_filtering(self, score_threshold: int = 20):
        """
        根据核心/次骨架/边缘节点分类，对 F 和 H 结构进行打分和过滤。
        得分 = score(u) + score(v)
        Score Map:
          - Core Skeleton: 100
          - Secondary Skeleton: 10
          - Peripheral: 1
        
        低于阈值 (default 20) 的结构被移入 'deferred' (冷宫) 字典，
        主字典只保留高优先级结构。
        """
        if self.debug:
            print(f"[DEBUG] Applying Structural Scoring (Threshold={score_threshold})...")
            
        # 1. 获取分类并建立得分映射
        core, secondary, peripheral = self.classify_all_nodes()
        node_scores = {}
        for u in core: node_scores[u] = 100
        for u in secondary: node_scores[u] = 10
        for u in peripheral: node_scores[u] = 1
        
        count_F_deferred = 0
        count_H_deferred = 0
        
        # 2. 遍历并过滤
        # dict_edge_to_cycles keys are mostly edge indices or tuples depending on usage.
        # Check enumerate_cycles implementation: keys are (a, b) tuples or indices?
        # In my implementation above, I used `self.dict_edge_to_cycles[(a, b)]`.
        
        # Need to handle potential modification during iteration, so list(keys)
        keys = list(self.dict_edge_to_cycles.keys())
        
        for k in keys:
            val = self.dict_edge_to_cycles[k]
            # Val is (E, F, G, H)
            if not isinstance(val, tuple) or len(val) != 4:
                continue
                
            E, F, G, H = val
            
            F_high, F_low = set(), set()
            H_high, H_low = set(), set()
            
            # Filter F
            for pair in F:
                s = node_scores.get(pair[0], 0) + node_scores.get(pair[1], 0)
                if s < score_threshold:
                    F_low.add(pair)
                else:
                    F_high.add(pair)
                    
            # Filter H
            for pair in H:
                s = node_scores.get(pair[0], 0) + node_scores.get(pair[1], 0)
                if s < score_threshold:
                    H_low.add(pair)
                else:
                    H_high.add(pair)
            
            # Update Main Dictionary
            self.dict_edge_to_cycles[k] = (E, F_high, G, H_high)
            
            # Update Deferred Dictionary (if any low priority found)
            if F_low or H_low:
                self.dict_edge_to_cycles_deferred[k] = (set(), F_low, set(), H_low)
                count_F_deferred += len(F_low)
                count_H_deferred += len(H_low)
        
        # 3. 重建 DFS 指路字典 (根据更新后的 High Priority H)
        # 需要重新计算全局 H_SUMMARY (只包含 high priority)
        # 注意：enumerate_cycles 中计算了 self.H_SUMMARY。即使我们修改了 dict，self.H_SUMMARY 还是旧的。
        # 这里我们应该重新生成一份 high priority 的 dfs_dirs
        
        # Rebuild summaries from High Priority dict
        new_F_summary = set()
        new_H_summary = set()
        for _, (_, F, _, H) in self.dict_edge_to_cycles.items():
            new_F_summary.update(F)
            new_H_summary.update(H)
            
        self.F_SUMMARY = new_F_summary
        self.H_SUMMARY = new_H_summary
        
        # Rebuild main dfs_dirs
        self.dfs_dirs_by_edge.clear()
        common_dirs = self.F_SUMMARY & self.H_SUMMARY
        for k, (_, _, _, H) in self.dict_edge_to_cycles.items():
            dirs = tuple(H & common_dirs)
            if dirs:
                self.dfs_dirs_by_edge[k] = dirs
                
        # Optional: Build deferred dfs_dirs? 
        # 用户说 "最后才考虑"，意味着现在先别动。
        
        if self.debug:
            print(f"[DEBUG] Structural Scoring Complete.")
            print(f"  - Deferred F-structures: {count_F_deferred}")
            print(f"  - Deferred H-structures: {count_H_deferred}")

    def rescue_cold_structures(self):
        """
        [Hot/Cold Palace Splicing]
        尝试将冷宫 (Deferred) 中的结构与热宫 (High Priority) 中的结构进行拼接。
        拼接逻辑：如果冷宫中的点对 (u, v) 中有任意一点已经存在于该边对应的热宫活跃节点中，
        则认为可以拼接（延长路径），将其“救回”热宫。
        
        分类存储：
        1. 待定区 (Pending Area / Medium-Long Edges): 
           - 第一轮被热宫直接救回的结构。
           - 存储在 self.pending_area_edges 中。
        2. 板凳区 (Bench Area / Longer Edges):
           - 后续轮次被“待定区”或“板凳区”救回的结构（即通过多跳拼接）。
           - 存储在 self.bench_area_edges 中。
        """
        self.pending_area_edges = set()
        self.bench_area_edges = set()
        self.long_bench_area_edges = set()
        self.pending_area_cycles = [] # New: Store reconstructed cycles from Pending Area
        self.bench_area_cycles = []   # New: Store reconstructed cycles from Bench Area
        self.long_bench_area_cycles = [] # New: Longer Bench, Close 2 times.
        
        if self.debug:
            print("[DEBUG] Attempting to rescue cold structures by splicing (Pending & Bench)...")
            
        count_rescued = 0
        
        # Pre-convert adjacency lists to sets for fast intersection
        adj_sets = {u: set(self.adj[u]) for u in self.adj}

        # Iterate over all edges that have deferred content
        keys = list(self.dict_edge_to_cycles_deferred.keys())
        
        for k in keys:
            # 1. Get Hot structures (Target for splicing)
            if k not in self.dict_edge_to_cycles:
                continue
            E_high, F_high, G_high, H_high = self.dict_edge_to_cycles[k]
            
            # 2. Get Cold structures (Source)
            val_def = self.dict_edge_to_cycles_deferred[k]
            _, F_low, _, H_low = val_def
            
            if not F_low and not H_low:
                continue
                
            # Base edge nodes for path reconstruction
            base_u, base_v = k
            
            # 3. Build Initial Active Nodes Set from original Hot Palace
            active_nodes = set()
            for u, v in F_high:
                active_nodes.add(u)
                active_nodes.add(v)
            for u, v in H_high:
                active_nodes.add(u)
                active_nodes.add(v)
                
            # 4. Iterative Splicing
            iteration = 0
            while True:
                iteration += 1
                rescued_this_round = set()
                
                # Check F_low
                to_rescue_F = set()
                for pair in F_low:
                    u, v = pair
                    if (u in active_nodes) or (v in active_nodes):
                        to_rescue_F.add(pair)
                
                # Check H_low
                to_rescue_H = set()
                for pair in H_low:
                    u, v = pair
                    if (u in active_nodes) or (v in active_nodes):
                        to_rescue_H.add(pair)
                        
                if not to_rescue_F and not to_rescue_H:
                    break
                    
                # Execute Rescue & Categorize
                total_rescue = to_rescue_F | to_rescue_H
                
                # Assign to correct area based on iteration
                if iteration == 1:
                    # Round 1: Directly connected to Hot -> Pending Area (Medium-Long)
                    self.pending_area_edges.update(total_rescue)
                    
                    # === ON-THE-FLY CYCLE GENERATION FOR PENDING AREA ===
                    # User Instruction: "Closed only twice: one with a common point, and one with an existing edge. If it can't be closed, give up directly."
                    
                    # -- Process F-rescues (Implicit Path: u - Pivot - v) --
                    for (u, v) in to_rescue_F:
                        pivot = None
                        if u in self.adj[base_u] and v in self.adj[base_u]: pivot = base_u
                        elif u in self.adj[base_v] and v in self.adj[base_v]: pivot = base_v
                        
                        if pivot is not None:
                            # 1. Existing Edge Closure -> Triangle (u-pivot-v-u)
                            if (min(u, v), max(u, v)) in self.edges:
                                self.pending_area_cycles.append((u, pivot, v))

                            # 2. Common Point Closure -> Square (u-pivot-v-w-u)
                            if u in adj_sets and v in adj_sets:
                                common_w = adj_sets[u] & adj_sets[v]
                                for w in common_w:
                                    if w != pivot:
                                         self.pending_area_cycles.append((u, pivot, v, w))

                    # -- Process H-rescues (Implicit Path: u - U_ANC - V_ANC - v) --
                    for (u, v) in to_rescue_H:
                        path_core = None
                        if u in self.adj[base_u] and v in self.adj[base_v]:
                            path_core = [u, base_u, base_v, v]
                        elif u in self.adj[base_v] and v in self.adj[base_u]:
                            path_core = [u, base_v, base_u, v]
                        elif v in self.adj[base_u] and u in self.adj[base_v]:
                             path_core = [v, base_u, base_v, u]
                        elif v in self.adj[base_v] and u in self.adj[base_u]:
                             path_core = [v, base_v, base_u, u]
                             
                        if path_core:
                             u_prim, _, _, v_prim = path_core

                             # 1. Existing Edge Closure -> Square (u-b1-b2-v-u)
                             if (min(u_prim, v_prim), max(u_prim, v_prim)) in self.edges:
                                 self.pending_area_cycles.append(tuple(path_core))

                             # 2. Common Point Closure -> Pentagon (u-b1-b2-v-w-u)
                             if u_prim in adj_sets and v_prim in adj_sets:
                                 common_w = adj_sets[u_prim] & adj_sets[v_prim]
                                 for w in common_w:
                                      if w not in path_core:
                                          self.pending_area_cycles.append(tuple(path_core + [w]))

                else:
                    # Round 2+: Connected via Pending -> Bench Area (Longer)
                    if iteration == 2:
                        self.bench_area_edges.update(total_rescue)

                        # === CYCLE GENERATION FOR BENCH AREA (One-Time Closure) ===
                        # Optimization: Volume is large, so only try to close ONCE using Common Point.
                        # We limit to 1 common neighbor (break) to avoid explosion.

                        # -- Process F-rescues --
                        for (u, v) in to_rescue_F:
                            pivot = None
                            if u in self.adj[base_u] and v in self.adj[base_u]: pivot = base_u
                            elif u in self.adj[base_v] and v in self.adj[base_v]: pivot = base_v
                            
                            if pivot is not None:
                                if u in adj_sets and v in adj_sets:
                                    common_w = adj_sets[u] & adj_sets[v]
                                    for w in common_w:
                                        if w != pivot:
                                            self.bench_area_cycles.append((u, pivot, v, w))
                                            break

                        # -- Process H-rescues --
                        for (u, v) in to_rescue_H:
                            path_core = None
                            if u in self.adj[base_u] and v in self.adj[base_v]:
                                path_core = [u, base_u, base_v, v]
                            elif u in self.adj[base_v] and v in self.adj[base_u]:
                                path_core = [u, base_v, base_u, v]
                            elif v in self.adj[base_u] and u in self.adj[base_v]:
                                path_core = [v, base_u, base_v, u]
                            elif v in self.adj[base_v] and u in self.adj[base_u]:
                                path_core = [v, base_v, base_u, u]
                                
                            if path_core:
                                u_prim, _, _, v_prim = path_core
                                if u_prim in adj_sets and v_prim in adj_sets:
                                    common_w = adj_sets[u_prim] & adj_sets[v_prim]
                                    for w in common_w:
                                        if w not in path_core:
                                            self.bench_area_cycles.append(tuple(path_core + [w]))
                                            break

                    else:
                        # Round 3+: Long Bench (Deferred, rare).
                        # User instructed: "Entering Long Bench can be closed twice... close twice"
                        self.long_bench_area_edges.update(total_rescue)

                        # -- Process F-rescues (Implicit Path: u - Pivot - v) --
                        for (u, v) in to_rescue_F:
                            pivot = None
                            if u in self.adj[base_u] and v in self.adj[base_u]: pivot = base_u
                            elif u in self.adj[base_v] and v in self.adj[base_v]: pivot = base_v
                            
                            if pivot is not None:
                                # 1. Existing Edge Closure -> Triangle (u-pivot-v-u)
                                if (min(u, v), max(u, v)) in self.edges:
                                    self.long_bench_area_cycles.append((u, pivot, v))

                                # 2. Common Point Closure -> Square (u-pivot-v-w-u) [ALL common neighbors]
                                if u in adj_sets and v in adj_sets:
                                    common_w = adj_sets[u] & adj_sets[v]
                                    for w in common_w:
                                        if w != pivot:
                                            self.long_bench_area_cycles.append((u, pivot, v, w))

                        # -- Process H-rescues --
                        for (u, v) in to_rescue_H:
                            path_core = None
                            if u in self.adj[base_u] and v in self.adj[base_v]:
                                path_core = [u, base_u, base_v, v]
                            elif u in self.adj[base_v] and v in self.adj[base_u]:
                                path_core = [u, base_v, base_u, v]
                            elif v in self.adj[base_u] and u in self.adj[base_v]:
                                path_core = [v, base_u, base_v, u]
                            elif v in self.adj[base_v] and u in self.adj[base_u]:
                                path_core = [v, base_v, base_u, u]
                                
                            if path_core:
                                u_prim, _, _, v_prim = path_core

                                # 1. Existing Edge Closure -> Square (u-b1-b2-v-u)
                                if (min(u_prim, v_prim), max(u_prim, v_prim)) in self.edges:
                                    self.long_bench_area_cycles.append(tuple(path_core))

                                # 2. Common Point Closure -> Pentagon (u-b1-b2-v-w-u)
                                if u_prim in adj_sets and v_prim in adj_sets:
                                    common_w = adj_sets[u_prim] & adj_sets[v_prim]
                                    for w in common_w:
                                        if w not in path_core:
                                            self.long_bench_area_cycles.append(tuple(path_core + [w]))

                
                # Update logic
                if to_rescue_F:
                    F_high.update(to_rescue_F)
                    F_low.difference_update(to_rescue_F)
                if to_rescue_H:
                    H_high.update(to_rescue_H)
                    H_low.difference_update(to_rescue_H)
                    
                count_rescued += len(total_rescue)
                rescued_this_round.update(total_rescue)
                
                # Update active nodes for next round
                # Nodes in rescued pairs become new active hooks
                for u, v in rescued_this_round:
                    active_nodes.add(u)
                    active_nodes.add(v)
                    
        # 5. Rebuild Global Summaries
        new_F_summary = set()
        new_H_summary = set()
        for _, (_, F, _, H) in self.dict_edge_to_cycles.items():
            new_F_summary.update(F)
            new_H_summary.update(H)
            
        self.F_SUMMARY = new_F_summary
        self.H_SUMMARY = new_H_summary
        
        self.dfs_dirs_by_edge.clear()
        common_dirs = self.F_SUMMARY & self.H_SUMMARY
        for k, (_, _, _, H) in self.dict_edge_to_cycles.items():
            dirs = tuple(H & common_dirs)
            if dirs:
                self.dfs_dirs_by_edge[k] = dirs
                
        if self.debug:
            print(f"[DEBUG] Rescue Complete. Rescued {count_rescued} pairs.")
            print(f"        [Pending Area] Medium-Long Edges: {len(self.pending_area_edges)}")
            print(f"        [Pending Area] Generated Cycles:  {len(self.pending_area_cycles)} (Squares/Pentagons)")
            print(f"        [Bench Area]   Longer Edges:      {len(self.bench_area_edges)}")



    def verify_structural_independence(self):
        """
        [Pipeline Execution]
        1. Phase 1: Artillery Barrage (Short Cycles < 6)
           - Generate & Bombard: Triangles, Squares, Pentagons, Hexagons.
        2. Phase 2: Chain Bombardment (Medium Structures)
           - Extract chains from uncovered regions -> Form Cycles -> Bombard.
        3. Phase 3: Spanning Tree Completion (Trace & Complete)
           - Reverse Source Tracing (Identifying persistent uncovered nodes).
           - Priority Tree Construction.
           - Fundamental Cycles.
        """
        t_total_start = time.time()
        if self.structural_independence_verified:
            return

        print("\n[PIPELINE] Starting Structural Resolution Pipeline...")
        
        # --- Preparation ---
        # Ensure edge index map exists
        if not self.edge_index_map:
             if self.debug:
                 print("[调试] 正在重新构建边索引映射以启动流水线...")
             valid_edges = []
             valid_nodes = set(self.adj.keys())
             for u in valid_nodes:
                 for v in self.adj[u]:
                     if u < v and v in valid_nodes:
                         valid_edges.append((u, v))
             valid_edges = sorted(list(set(valid_edges)))
             self.undirected_edge2_single_str = valid_edges
             self.edge_index_map = {edge: i for i, edge in enumerate(valid_edges)}

        eid_count = len(self.edge_index_map)
        reducer_debug_flag = getattr(self, 'reducer_debug', False)
        reducer = BlockwiseCycleReducer(eid_count=eid_count, target_rank=-1, debug=reducer_debug_flag)
        self.reducer = reducer
        
        independent_count = 0
        total_structural = 0
        
        # Helper: Convert node-tuple cycle to edge-index set
        def cycle_nodes_to_eids(nodes: Tuple[int, ...]) -> FrozenSet[int]:
            eids = []
            n = len(nodes)
            debug_miss = False
            for i in range(n):
                u, v = nodes[i], nodes[(i+1)%n]
                if u > v: u, v = v, u
                if (u, v) in self.edge_index_map:
                    eids.append(self.edge_index_map[(u, v)])
                else:
                    debug_miss = True
            if len(eids) == 0 and debug_miss and self.debug:
                 print(f"[DEBUG] Cycle edges MISSING from map: nodes={nodes}")
            return frozenset(eids)

        # =========================================================================
        # PHASE 1: ARTILLERY BARRAGE (Short Cycles < 6)
        # "小于6环，加入炮兵阵地"
        # =========================================================================
        t_p1 = time.time()
        print("[流水线] 第一阶段：重炮轰炸 (短环结构 < 6)")
        
        # 1.1 Triangles (3-cycles)
        triangles = self.find_simple_triangles()
        tri_indep = 0
        for tri in triangles:
            cyc_eids = cycle_nodes_to_eids(tri)
            indep_cyc, _ = reducer.add_candidate(cyc_eids)
            if indep_cyc is not None:
                independent_count += 1
                self.independent_structural_cycles.append(cyc_eids)
            total_structural += 1
        tri_indep = independent_count
        print(f"    - [重炮] 三角形 (3环):   {len(triangles)} 发现, {tri_indep} 线性无关。")

        # 1.2 Squares (4-cycles)
        squares = self.find_simple_squares()
        for sq in squares:
            cyc_eids = cycle_nodes_to_eids(sq)
            indep_cyc, _ = reducer.add_candidate(cyc_eids)
            if indep_cyc is not None:
                independent_count += 1
                self.independent_structural_cycles.append(cyc_eids)
            total_structural += 1
        sq_indep = independent_count - tri_indep
        print(f"    - [重炮] 正方形 (4环):   {len(squares)} 发现, {sq_indep} 线性无关。")

        # 1.3 Pentagons (5-cycles)
        pentagons = self.find_simple_pentagons()
        for pent in pentagons:
            cyc_eids = cycle_nodes_to_eids(pent)
            indep_cyc, _ = reducer.add_candidate(cyc_eids)
            if indep_cyc is not None:
                independent_count += 1
                self.independent_structural_cycles.append(cyc_eids)
            total_structural += 1
        pent_indep = independent_count - tri_indep - sq_indep
        print(f"    - [重炮] 五边形 (5环):   {len(pentagons)} 发现, {pent_indep} 线性无关。")
        
        # 1.4 Hexagons (6-cycles)
        hexagons = self.find_simple_hexagons()
        for hexa in hexagons:
            cyc_eids = cycle_nodes_to_eids(hexa)
            indep_cyc, _ = reducer.add_candidate(cyc_eids)
            if indep_cyc is not None:
                independent_count += 1
                self.independent_structural_cycles.append(cyc_eids)
            total_structural += 1
        hex_indep = independent_count - tri_indep - sq_indep - pent_indep
        print(f"    - [重炮] 六边形 (6环):   {len(hexagons)} 发现, {hex_indep} 线性无关。")
        
        self.phase_times['Phase 1'] = time.time() - t_p1

        # =========================================================================
        # PHASE 2: CHAIN BOMBARDMENT (Medium Structures)
        # "链式节点和中长区对接...生成...轰炸"
        # =========================================================================
        t_p2 = time.time()
        print("[流水线] 第二阶段：链式轰炸 (中型结构)")
        
        # 2.1 Identify Uncovered Edges
        covered_now = set()
        for c in self.independent_structural_cycles:
            covered_now.update(c)
        uncovered_for_chains = set(self.edge_index_map.values()) - covered_now
        
        # 2.2 Build Virtual Graph (Uncovered Only)
        v_adj = defaultdict(list)
        for eid in uncovered_for_chains:
            u, v = self.undirected_edge2_single_str[eid]
            v_adj[u].append((v, eid))
            v_adj[v].append((u, eid))
        
        # 2.3 Extract Chains
        bombarded_count = 0
        visited_chain_edges = set()
        
        sorted_uncovered_nodes = sorted(list(v_adj.keys()))
        
        for start_node in sorted_uncovered_nodes:
            # We want to start chains from degree!=2 nodes if possible, or any node
            if len(v_adj[start_node]) != 2 or start_node not in visited_chain_edges: # simplified logic check
                 pass
            
            # Simple traversal for each neighbor
            for (neighbor, start_eid) in v_adj[start_node]:
                if start_eid in visited_chain_edges: continue
                
                # Start tracing a chain
                chain_eids = [start_eid]
                curr = neighbor
                prev = start_node
                chain_start_node = start_node
                
                # Walk while degree == 2
                while len(v_adj[curr]) == 2:
                    # Get neighbors
                    nbs = v_adj[curr]
                    # Find the one that is not prev
                    next_hop = None
                    next_eid = None
                    
                    for (n, e) in nbs:
                        if n != prev:
                            next_hop = n
                            next_eid = e
                            break
                    
                    if next_hop is None: break # Loop back or dead end?
                    if next_eid in visited_chain_edges: break
                    if next_eid in chain_eids: break # Cycle detected within chain walk
                    
                    chain_eids.append(next_eid)
                    prev = curr
                    curr = next_hop
                    
                    if curr == chain_start_node: break # Closed loop found
                    if len(chain_eids) > 2000: break # Safety limit
                
                chain_end_node = curr
                visited_chain_edges.update(chain_eids)
                
                # 2.4 Form Candidates & Bombard
                final_candidate = None
                
                # Case A: Self-Closed Loop
                if chain_start_node == chain_end_node:
                    final_candidate = frozenset(chain_eids)
                else:
                    # Case B: Open Chain - Try 1-hop or 2-hop closure
                    pair = tuple(sorted((chain_start_node, chain_end_node)))
                    if pair in self.edge_index_map:
                        cand = list(chain_eids)
                        cand.append(self.edge_index_map[pair])
                        final_candidate = frozenset(cand)
                    else:
                        # Try 2-hop (Triangle)
                        n1_set = set(self.adj[chain_start_node])
                        n2_set = set(self.adj[chain_end_node])
                        common = n1_set & n2_set
                        if common:
                            w = next(iter(common))
                            p1 = tuple(sorted((chain_start_node, w)))
                            p2 = tuple(sorted((w, chain_end_node)))
                            if p1 in self.edge_index_map and p2 in self.edge_index_map:
                                cand = list(chain_eids)
                                cand.append(self.edge_index_map[p1])
                                cand.append(self.edge_index_map[p2])
                                final_candidate = frozenset(cand)

                if final_candidate:
                    # --- VALIDATION START ---
                    is_valid_structure = True
                    check_counts = defaultdict(int)
                    for e_chk in final_candidate:
                        tu_chk, tv_chk = self.undirected_edge2_single_str[e_chk]
                        check_counts[tu_chk] += 1
                        check_counts[tv_chk] += 1
                    for _, c_chk in check_counts.items():
                        if c_chk % 2 != 0:
                            is_valid_structure = False
                            break
                    # --- VALIDATION END ---

                    if is_valid_structure:
                        indep, _ = reducer.add_candidate(final_candidate)
                        if indep:
                            self.independent_structural_cycles.append(final_candidate)
                            bombarded_count += 1
                        
        print(f"    - [链式] 轰炸结果: 新增 {bombarded_count} 个线性无关基。")
        self.phase_times['Phase 2'] = time.time() - t_p2

        # =========================================================================
        # PHASE 3: SPANNING TREE COMPLETION (Trace & Complete)
        # "反溯源，补全"
        # =========================================================================
        t_p3 = time.time()
        print("[流水线] 第三阶段：生成树补全 (溯源 & 闭合)")

        # 3.1 Reverse Source Tracing (Updated Coverage)
        covered_now = set()
        for c in self.independent_structural_cycles:
            covered_now.update(c)
        uncovered_eids_final = set(self.edge_index_map.values()) - covered_now
        
        # "反溯源": Identify sources (nodes) of remaining incompleteness
        priority_nodes = set()
        for eid in uncovered_eids_final:
            u, v = self.undirected_edge2_single_str[eid]
            priority_nodes.add(u)
            priority_nodes.add(v)
            
        print(f"    - [溯源] 发现 {len(priority_nodes)} 个顽固未覆盖节点。")
        
        # 3.2 Priority Tree Construction
        # Build BFS tree starting from these priority nodes
        parent_map = {}
        depth_map = {}
        tree_edges = set()
        visited = set()
        
        bfs_queue = deque()
        
        # Priority Queueing: [Priority Nodes] ... [Others]
        all_nodes = list(self.adj.keys())
        p_list = sorted(list(priority_nodes))
        other_list = sorted(list(set(all_nodes) - priority_nodes))
        start_order = p_list + other_list
        
        for root in start_order:
            if root in visited: continue
            
            visited.add(root)
            parent_map[root] = None
            depth_map[root] = 0
            bfs_queue.append(root)
            
            while bfs_queue:
                u = bfs_queue.popleft()
                for v in self.adj[u]:
                    if v not in visited:
                        visited.add(v)
                        parent_map[v] = u
                        depth_map[v] = depth_map[u] + 1
                        bfs_queue.append(v)
                        
                        # Mark Tree Edge
                        pair = tuple(sorted((u, v)))
                        if pair in self.edge_index_map:
                            tree_edges.add(self.edge_index_map[pair])
        
        # 3.3 Fundamental Cycle Completion
        tree_added_count = 0
        all_edges = sorted(list(self.edge_index_map.keys()))
        
        for (u, v) in all_edges:
            eid = self.edge_index_map[(u, v)]
            if eid in tree_edges: continue # Tree Edge
            
            # Co-Tree Edge -> Form Cycle
            du = depth_map.get(u, -1)
            dv = depth_map.get(v, -1)
            if du == -1 or dv == -1: continue 
            
            curr_u, curr_v = u, v
            path_u = [u]
            path_v = [v]
            
            # Trace Up
            while du > dv:
                curr_u = parent_map[curr_u]
                path_u.append(curr_u)
                du -= 1
            while dv > du:
                curr_v = parent_map[curr_v]
                path_v.append(curr_v)
                dv -= 1
            while curr_u != curr_v:
                curr_u = parent_map[curr_u]
                curr_v = parent_map[curr_v]
                path_u.append(curr_u)
                path_v.append(curr_v)
                
            # Merge Paths: u..LCA..v
            cycle_nodes = path_u + path_v[:-1][::-1]
            
            # Convert to EIDs
            cyc_eids = []
            valid = True
            for i in range(len(cycle_nodes)):
                n1 = cycle_nodes[i]
                n2 = cycle_nodes[(i+1) % len(cycle_nodes)]
                pair = tuple(sorted((n1, n2)))
                if pair in self.edge_index_map:
                    cyc_eids.append(self.edge_index_map[pair])
                else:
                    valid = False; break
            
            if valid:
                fcyc = frozenset(cyc_eids)
                indep, _ = reducer.add_candidate(fcyc)
                if indep:
                    self.independent_structural_cycles.append(fcyc)
                    tree_added_count += 1
                    
        print(f"    - [补全] 生成树补全结束。新增基环: {tree_added_count}")
        self.phase_times['Phase 3'] = time.time() - t_p3
        self.pipeline_total_duration = time.time() - t_total_start
        
        self.structural_independence_verified = True
        
        # Final Stats Logic
        self.verify_final_statistics()

    def verify_final_statistics(self):
        """
        汇报最终简要统计数据：
        1. 线性无关基数量
        2. 总权重 (Total Length)
        3. 平均环长
        """
        # 1. Basis Count
        basis_count = len(self.independent_structural_cycles)
        
        # 2. Length Stats
        total_len = sum(len(c) for c in self.independent_structural_cycles)
        avg_len = total_len / basis_count if basis_count > 0 else 0
        
        print("\n[最终简报]")
        print(f"1. 线性无关基数量: {basis_count}")
        print(f"2. 基环总长度:     {total_len}")
        print(f"3. 平均环长:       {avg_len:.2f}")

    def rescue_cold_structures_legacy(self):
        """
        “热宫救冷宫” (Rescue Cold with Hot):
        Deprecated in favor of split specific rescue.
        """
        pass

# =====================================================
# BlockwiseCycleReducer —— 基于块的环消元器 (CSR 优化)
class BlockwiseCycleReducer:
    """
    GF(2) Array-based Block Elimination (CSR Optimized):
    1. All basis vectors stored in ONE contiguous array('Q') -> self.basis_data.
    2. self.basis_map maps pivot -> (start_index, length).
    3. Uses 64-bit blocks.
    """

    def __init__(self, *, eid_count: int, target_rank: int, block_size: int = 64, debug: bool = False):
        self.eid_count = eid_count
        self.target_rank = target_rank
        self.block_size = 64
        self.block_count = (eid_count + self.block_size - 1) // self.block_size
        self.debug = debug

        # Globally pooled storage [idx, val, idx, val, ...]
        self.basis_data = array.array('Q')
        # Map: pivot_bit -> (start_index, length_in_entries)
        self.basis_map: Dict[int, Tuple[int, int]] = {}

        # perf counters
        self.perf_candidates_in = 0
        self.perf_added = 0
        self.perf_xor_ops = 0
        
        self.time_start = time.time()

    def get_stats(self) -> str:
        """CSR 统计"""
        total_vectors = len(self.basis_map)
        if total_vectors == 0:
            return "Basis empty"
        # basis_data 长度即总 entries (每 block 2 entries)
        total_blocks = len(self.basis_data) // 2
        avg_blocks = total_blocks / total_vectors
        # array buffer size in MB
        size_mb = self.basis_data.buffer_info()[1] * self.basis_data.itemsize / (1024 * 1024)
        return (f"Vectors: {total_vectors}, "
                f"Total Blocks: {total_blocks}, "
                f"Avg Blocks/Vec: {avg_blocks:.1f}, "
                f"Pool Size: {size_mb:.2f} MB")

    def eids_to_vector(self, eids: FrozenSet[int]) -> array.array:
        """Converts eids to sorted flat array vector [idx, val, idx, val...]"""
        # ... logic unchanged, returns a temporary array ...
        if not eids:
            return array.array('Q')
            
        sorted_eids = sorted(eids)
        vector_data = []
        current_blk = -1
        current_val = 0
        
        for eid in sorted_eids:
            # blk = eid // 64, rem = eid % 64
            blk = eid >> 6
            rem = eid & 63
            if blk != current_blk:
                if current_blk != -1:
                    vector_data.append(current_blk)
                    vector_data.append(current_val)
                current_blk = blk
                current_val = 0
            current_val |= (1 << rem)
            
        if current_blk != -1:
            vector_data.append(current_blk)
            vector_data.append(current_val)
            
        return array.array('Q', vector_data)

    def vector_to_eids(self, vector: array.array) -> FrozenSet[int]:
        eids = []
        for k in range(0, len(vector), 2):
            blk = vector[k]
            val = vector[k+1]
            base = blk << 6
            temp = val
            offset = 0
            while temp:
                if temp & 1:
                    eids.append(base + offset)
                temp >>= 1
                offset += 1
        return frozenset(eids)

    def find_pivot_and_reduce(self, vector: array.array) -> Tuple[array.array, Optional[int]]:
        """
        Reduces the vector against the basis (stored in CSR).
        vector is modified in-place or replaced.
        """
        while len(vector) > 0:
            last_blk_idx = vector[-2]
            last_blk_val = vector[-1]
            if last_blk_val == 0:
                vector.pop()
                vector.pop()
                continue
                
            high_bit = last_blk_val.bit_length() - 1
            pivot = (last_blk_idx << 6) + high_bit
            
            if pivot in self.basis_map:
                start, length = self.basis_map[pivot]
                # Pass data ref and slice info
                vector = self.xor_with_basis(vector, self.basis_data, start, length)
            else:
                return vector, pivot
                
        return vector, None

    def xor_with_basis(self, v1: array.array, pool: array.array, start: int, length: int) -> array.array:
        """
        v1 ^ pool[start : start+length]
        """
        self.perf_xor_ops += 1
        res_list = []
        
        # v1 pointer
        i = 0
        len1 = len(v1)
        
        # pool pointer (virtual v2)
        j = 0
        len2 = length
        
        while i < len1 and j < len2:
            b1 = v1[i]
            # pool stores [blk, val, blk, val...]
            # j is relative offset in pairs of entries? No, length is total entries.
            # pointers need to jump by 2
            b2 = pool[start + j]
            
            if b1 < b2:
                res_list.append(b1)
                res_list.append(v1[i+1])
                i += 2
            elif b1 > b2:
                res_list.append(b2)
                res_list.append(pool[start + j + 1])
                j += 2
            else:
                val1 = v1[i+1]
                val2 = pool[start + j + 1]
                new_val = val1 ^ val2
                if new_val:
                    res_list.append(b1)
                    res_list.append(new_val)
                i += 2
                j += 2
        
        if i < len1:
            res_list.extend(v1[i:])
            
        if j < len2:
            # slice from pool
            res_list.extend(pool[start+j : start+len2])
            
        return array.array('Q', res_list)

    def add_candidate(self, cyc: FrozenSet[int]) -> Tuple[Optional[FrozenSet[int]], Optional[FrozenSet[int]]]:
        """
        尝试加入候选环。
        Returns:
            (independent_cycle, dependent_cycle)
            - 成功加入(线性无关): (cyc, None)
            - 失败(线性相关): (None, cyc)
        """
        if not cyc:
            return (None, None)

        self.perf_candidates_in += 1
        vector = self.eids_to_vector(cyc)
        reduced_vector, pivot = self.find_pivot_and_reduce(vector)
        
        if pivot is not None:
            # 线性无关，加入基
            start_idx = len(self.basis_data)
            length = len(reduced_vector)
            self.basis_data.extend(reduced_vector)
            self.basis_map[pivot] = (start_idx, length)
            
            self.perf_added += 1
            return (cyc, None)
        else:
            # 线性相关，丢弃
            return (None, cyc)

    def check_independent(self, cyc: FrozenSet[int]) -> bool:
        """纯检查，不修改内部状态"""
        if not cyc: 
            return False
        vector = self.eids_to_vector(cyc)
        vector, pivot = self.find_pivot_and_reduce(vector)
        return pivot is not None

    def get_reduced_candidate(self, cyc: FrozenSet[int]) -> Optional[FrozenSet[int]]:
        """只返回消元后的剩余环（不加入基）"""
        if not cyc:
            return None
        vector = self.eids_to_vector(cyc)
        vector, pivot = self.find_pivot_and_reduce(vector)
        if not vector:
            return None
        return self.vector_to_eids(vector)

    def get_dependent_cycle_if_short(self, cyc: FrozenSet[int], max_len: int = 8) -> Optional[FrozenSet[int]]:
        """
        [New] 如果环线性相关，且长度 <= max_len，则返回原环。
        这用于捕获那些被认为“冗余”但实际上很短、可能比基中现有组合更优的环。
        """
        if not cyc: return None
        if len(cyc) > max_len: return None
        
        # 1. 检查是否线性相关 (check_independent 返回 False 表示相关)
        if not self.check_independent(cyc):
            return cyc
            
        return None

    def clone(self) -> "BlockwiseCycleReducer":
        new_reducer = BlockwiseCycleReducer(
            eid_count=self.eid_count,
            target_rank=self.target_rank,
            block_size=self.block_size,
            debug=self.debug
        )
        new_reducer.basis_data = array.array('Q', self.basis_data)
        new_reducer.basis_map = self.basis_map.copy()
        
        new_reducer.perf_candidates_in = self.perf_candidates_in
        new_reducer.perf_added = self.perf_added
        new_reducer.perf_xor_ops = self.perf_xor_ops
        return new_reducer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative MCB Solver (2-Core Only)")
    parser.add_argument("input_file", type=str, help="Edge list file")
    parser.add_argument("output_file", type=str, help="Output file for cycles")
    parser.add_argument("--debug-di", action="store_true", help="Enable debug for DataInitialization")
    parser.add_argument("--debug-finder", action="store_true", help="Enable debug for GraphCycleFinder")
    parser.add_argument("--debug-reducer", action="store_true", help="Enable debug for BlockwiseCycleReducer")
    
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}。")
        sys.exit(1)
    
    dict_map = {
        "0": 10, "1": 11, "2": 20, "3": 21, "4": 30, "5": 31, "6": 40, "7": 41, "8": 50, "9": 51,
        "10": 0, "11": 1, "20": 2, "21": 3, "30": 4, "31": 5, "40": 6, "41": 7, "50": 8, "51": 9,
    }

    t_di_start = time.time()
    print(f"正在加载 {input_file}...")
    di = DataInitialization(input_file, dict_map, debug=args.debug_di)
    t_di_end = time.time()
    di_duration = t_di_end - t_di_start
    
    print(f"图加载完成，共 {len(di.lines)} 条原始边。")
    
    t_finder_init_start = time.time()
    finder = GraphCycleFinder(di.lines, debug=args.debug_finder)
    # Store reducer debug option to be used in verify_structural_independence
    finder.reducer_debug = args.debug_reducer
    t_finder_init_end = time.time()
    finder_init_duration = t_finder_init_end - t_finder_init_start
    
    print("正在运行求解器...")
    start_time = time.time()
    finder.verify_structural_independence()
    solver_duration = time.time() - start_time
    print(f"求解器在 {solver_duration:.2f} 秒内完成。")
    
    cycles = finder.independent_structural_cycles
    print(f"找到 {len(cycles)} 个独立环。")
    
    # helper for reconstruction (local)
    def reconstruct_cycle_path(cyc_eids, edge_list):
        if not cyc_eids: return []
        # Build adjacency for this cycle
        local_adj = defaultdict(list)
        for eid in cyc_eids:
            if eid < 0 or eid >= len(edge_list): continue
            u, v = edge_list[eid]
            local_adj[u].append(v)
            local_adj[v].append(u)
        
        # Validate degree 2
        for n, nbs in local_adj.items():
            if len(nbs) != 2: return None # Not a simple cycle
            
        # Walk
        start_node = next(iter(local_adj))
        path = [start_node]
        curr = start_node
        prev = None
        
        # Safety limit
        for _ in range(len(local_adj) + 2):
            nbs = local_adj[curr]
            next_node = nbs[0]
            if next_node == prev:
                if len(nbs) > 1:
                    next_node = nbs[1]
                else:
                    return None # Dead end
            
            if next_node == start_node:
                # Closed
                path.append(start_node)
                return path
            
            path.append(next_node)
            prev = curr
            curr = next_node
        return None

    print("Writing output to " + output_file)
    write_count = 0
    with open(output_file, 'w') as f:
        # Buffer output
        buffer = []
        for cyc in cycles:
            path = reconstruct_cycle_path(cyc, finder.undirected_edge2_single_str)
            if path:
                # Output format: length node1 node2 ...
                # To match other solutions, output the closed path (with repeated start node)
                nodes = path # Use full path [v1, v2, ..., v1]
                buffer.append(f"{len(nodes)} {' '.join(str(x) for x in nodes)}")
                write_count += 1
        
        f.write(f"{write_count}\n")
        f.write("\n".join(buffer))
        f.write("\n")
            
    print("Done.")

    # --- Comprehensive Performance & Stats Report ---
    print("\n==================================================")
    print("综合性能与统计报告")
    print("==================================================")
    print("[时间统计]")
    print(f"1. 数据加载:       {di_duration:.4f}秒")
    print(f"2. 求解器初始化:   {finder_init_duration:.4f}秒")
    
    pipeline_total = getattr(finder, 'pipeline_total_duration', 0)
    print(f"3. 核心流水线:     {pipeline_total:.4f}秒")
    
    phase_times = getattr(finder, 'phase_times', {})
    p1 = phase_times.get('Phase 1', 0)
    p2 = phase_times.get('Phase 2', 0)
    p3 = phase_times.get('Phase 3', 0)
    print(f"   - 第一阶段 (重炮): {p1:.4f}秒")
    print(f"   - 第二阶段 (链式): {p2:.4f}秒")
    print(f"   - 第三阶段 (补全): {p3:.4f}秒")
    
    total_app_time = di_duration + finder_init_duration + solver_duration
    print("-" * 50)
    print(f"总耗时:            {total_app_time:.4f}秒")
    
    print("\n[数据统计]")
    rank = len(cycles)
    print(f"1. 线性无关基数量: {rank}")
    
    # Coverage logic duplicated slightly from verify_stats but useful here
    covered_eids = set()
    total_len = 0
    min_len = float('inf')
    max_len = 0
    
    for c in cycles:
        length = len(c)
        total_len += length
        min_len = min(min_len, length)
        max_len = max(max_len, length)
        covered_eids.update(c)
        
    avg_len = total_len / rank if rank > 0 else 0
    if min_len == float('inf'): min_len = 0
    
    num_edges_map = len(finder.edge_index_map)
    coverage_pct = (len(covered_eids) / num_edges_map * 100) if num_edges_map > 0 else 0.0
    
    print(f"2. 边覆盖率:       {coverage_pct:.2f}% ({len(covered_eids)}/{num_edges_map})")
    print(f"3. 总权重:         {total_len}")
    print(f"4. 平均环长:       {avg_len:.2f}")
    print(f"5. 最小环长:       {min_len}")
    print(f"6. 最大环长:       {max_len}")
    print("==================================================\n")
