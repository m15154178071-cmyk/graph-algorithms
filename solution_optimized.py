from __future__ import annotations

import os
# ===== Remove locale settings; ensure compatibility via standard libraries =====

import sys
import re
import time
import itertools
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, Set, Tuple, List, Optional, FrozenSet, Callable, Any


# =====================================================
# Utils
# =====================================================
class CycleUtils:
    @staticmethod
    def reconstruct_cycle_order_from_eids(
        cycle_eids: List[int],
        eid_to_edge: List[Optional[Tuple[str, str]]],
    ) -> Optional[List[str]]:
        """
        Reconstruct a valid simple cycle node order from a set of edge IDs (eids).
        Returns: [v1, v2, ..., v1]
        """
        adj: Dict[str, List[str]] = defaultdict(list)
        for eid in cycle_eids:
            uv = eid_to_edge[eid]
            if uv is None:
                return None
            u, v = uv
            adj[u].append(v)
            adj[v].append(u)

        if not adj:
            return None

        # simple cycle: each node must have a degree of 2
        for x, ns in adj.items():
            if len(ns) != 2:
                return None

        start = next(iter(adj))
        path = [start]
        prev: Optional[str] = None
        curr: str = start

        while True:
            n0, n1 = adj[curr]
            if prev is None:
                nxt = n0
            else:
                nxt = n0 if n1 == prev else n1

            if nxt == start:
                path.append(start)
                break

            # infinite loop protection (data anomaly fallback)
            if nxt in path:
                return None

            path.append(nxt)
            prev, curr = curr, nxt

        return path

    @staticmethod
    def assert_simple_cycle_by_eids(
        cyc: FrozenSet[int],
        eid_to_edge: List[Optional[Tuple[str, str]]],
        idx: int = -1,
    ) -> None:
        """
        Strict simple cycle check: each node must have a degree of 2.
        """
        deg: Dict[str, int] = defaultdict(int)
        for eid in cyc:
            uv = eid_to_edge[eid]
            if uv is None:
                raise RuntimeError(f"Invalid eid={eid} (index={idx})")
            u, v = uv
            deg[u] += 1
            deg[v] += 1
            if deg[u] > 2 or deg[v] > 2:
                raise RuntimeError(f"Non-simple cycle (index={idx})")
        if any(d != 2 for d in deg.values()):
            raise RuntimeError(f"Non-simple cycle (index={idx})")


# =====================================================
# Input processor
# =====================================================
class InputDataProcessor:
    """
    - Enumerates C3 / C4 / C5 (optional induced_only = chordless)
    - Extracts structures F / H
    - Constructs dfs_dirs_by_edge: Guides C6+ DFS (provides directions, does not directly form cycles)
    """

    def __init__(self, lines: List[List[str]], heavy_edge_threshold: int = 64):
        self.lines = lines
        self.pad_width: Optional[int] = None
        self.heavy_edge_threshold = heavy_edge_threshold
        self.heavy_edge_count = 0

        # -------- Graph Structure --------
        self.undirected_edge2_single_str: Set[Tuple[str, str]] = set()
        self.adjacency_map: Dict[str, Set[str]] = {}

        # -------- Edge Indexing --------
        self.edge_index_map: Dict[Tuple[str, str], int] = {}
        self.eid_to_edge: List[Optional[Tuple[str, str]]] = [None]  # 1-based
        self.node_index_map: Dict[str, int] = {}

        # -------- Structure Cache --------
        # dict_edge_to_cycles[(a,b)] = (E, F, G, H) each is set[(u,v)]
        self.dict_edge_to_cycles: Dict[
            Tuple[str, str],
            Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Set[Tuple[str, str]], Set[Tuple[str, str]]]
        ] = defaultdict(lambda: (set(), set(), set(), set()))

        self.E_SUMMARY: Set[Tuple[str, str]] = set()
        self.F_SUMMARY: Set[Tuple[str, str]] = set()
        self.G_SUMMARY: Set[Tuple[str, str]] = set()
        self.H_SUMMARY: Set[Tuple[str, str]] = set()

        # -------- DFS Direction Index (Core) --------
        # dfs_dirs_by_edge[(a,b)] = tuple[(m1,m2), ...]
        self.dfs_dirs_by_edge: Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]] = {}

        self._build_graph()
        self._build_edge_index()
        self._build_node_index()

    # =====================================================
    # Build undirected graph (compatible with dirty rows: row may not have exactly two numbers)
    # =====================================================
    def _build_graph(self) -> None:
        edges: List[Tuple[int, int]] = []

        for row in self.lines:
            nums: List[int] = []
            for x in row:
                if x is None:
                    continue
                s = str(x).strip()
                if not s:
                    continue
                nums.append(int(s))

            if len(nums) < 2:
                continue

            # Compatible with "maybe more than 2 numbers in a row": take min/max as undirected edge
            u = min(nums)
            v = max(nums)
            if u != v:
                edges.append((u, v))

        if not edges:
            self.pad_width = 1
            self.adjacency_map = {}
            self.undirected_edge2_single_str = set()
            return

        max_node = max(max(u, v) for u, v in edges)
        self.pad_width = len(str(max_node))

        adj: Dict[str, Set[str]] = defaultdict(set)
        for u, v in edges:
            su = str(u).zfill(self.pad_width)
            sv = str(v).zfill(self.pad_width)
            if su > sv:
                su, sv = sv, su

            self.undirected_edge2_single_str.add((su, sv))
            adj[su].add(sv)
            adj[sv].add(su)

        self.adjacency_map = dict(adj)

    # =====================================================
    # Build Edge IDs
    # =====================================================
    def _build_edge_index(self) -> None:
        for u, v in sorted(self.undirected_edge2_single_str):
            eid = len(self.eid_to_edge)
            self.edge_index_map[(u, v)] = eid
            self.edge_index_map[(v, u)] = eid
            self.eid_to_edge.append((u, v))

    # =====================================================
    # Build Node Index
    # =====================================================
    def _build_node_index(self) -> None:
        nid = 1
        for u, v in self.undirected_edge2_single_str:
            if u not in self.node_index_map:
                self.node_index_map[u] = nid
                nid += 1
            if v not in self.node_index_map:
                self.node_index_map[v] = nid
                nid += 1

    # =====================================================
    # Cyclomatic Number (Rank) beta = |E| - |V| + cc
    # =====================================================
    def connected_components_count(self) -> int:
        """Count connected components in the current graph (nodes implied by edges)."""
        if not self.adjacency_map:
            return 0
        visited: Set[str] = set()
        comp = 0
        for root in self.adjacency_map.keys():
            if root in visited:
                continue
            comp += 1
            q = deque([root])
            visited.add(root)
            while q:
                x = q.popleft()
                for y in self.adjacency_map.get(x, ()):
                    if y not in visited:
                        visited.add(y)
                        q.append(y)
        return comp

    def minimal_cycle_count(self) -> int:
        cc = self.connected_components_count()
        if cc == 0:
            return 0
        return len(self.undirected_edge2_single_str) - len(self.node_index_map) + cc

    # =====================================================
    # Chordless (Induced) Cycle Check: edge count == node count in subgraph
    # =====================================================
    def _is_induced_cycle_nodes(self, nodes: Set[str]) -> bool:
        for x in nodes:
            deg = 0
            for y in self.adjacency_map.get(x, set()):
                if y in nodes:
                    deg += 1
                    if deg > 2:
                        return False
            if deg != 2:
                return False
        return True

    # =====================================================
    # Enumerate C3/C4/C5 + Construct F/H + Build Guide Dict dfs_dirs_by_edge
    # =====================================================
    def enumerate_cycles(self, induced_only: bool = True) -> Dict[int, Set[FrozenSet[int]]]:
        cycles_by_len: Dict[int, Set[FrozenSet[int]]] = defaultdict(set)
        E2 = self.undirected_edge2_single_str
        edge_index = self.edge_index_map

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
                        cycles_by_len[3].add(frozenset((
                            edge_index[(a, b)],
                            edge_index[(a, x)],
                            edge_index[(b, x)]
                        )))
                continue

            B = set(itertools.combinations(sorted(only_a), 2))
            C = set(itertools.combinations(sorted(only_b), 2))
            E = (B | C) & E2
            F = ((B | C) - E) - E2

            D = set(itertools.combinations(sorted(unique), 2))
            G = (D - B - C) & E2

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
                    cycles_by_len[3].add(frozenset((
                        edge_index[(a, b)],
                        edge_index[(a, x)],
                        edge_index[(b, x)]
                    )))

            # -------- C4: a-u-v-b-a (using G) --------
            for (u, v) in G:
                cyc_nodes = {a, b, u, v}
                if induced_only and (not self._is_induced_cycle_nodes(cyc_nodes)):
                    continue

                eids: List[int] = []
                ok = True
                for x, y in ((a, u), (u, v), (v, b), (b, a)):
                    eid = edge_index.get((x, y))
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
                    for x, y in zip(seq, seq[1:]):
                        eid = edge_index.get((x, y))
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


# =====================================================
# Guided DFS finder (all logic moved here)
# =====================================================
class GuidedDFSFinder:
    """
    负责：
    - 统一 DFS 找 C6+ induced simple cycles
    - 用 dfs_dirs_by_edge 指路，对未覆盖边补环
    """

    def __init__(
        self,
        *,
        adjacency_map: Dict[str, Set[str]],
        edge_index_map: Dict[Tuple[str, str], int],
        eid_to_edge: List[Optional[Tuple[str, str]]],
        undirected_edges: Set[Tuple[str, str]],
        dfs_dirs_by_edge: Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]],
    ):
        self.adj = adjacency_map
        self.edge_index_map = edge_index_map
        self.eid_to_edge = eid_to_edge
        self.undirected_edges = undirected_edges
        self.dfs_dirs_by_edge = dfs_dirs_by_edge

    # -----------------------------
    # DFS: a - m1 - (path) - m2 - b - a
    # -----------------------------
    def dfs_find_c_cycles_upto(
        self,
        *,
        a: str,
        b: str,
        m1: str,
        m2: str,
        max_len: int,
        limit: int = 1,
    ) -> List[FrozenSet[int]]:
        forbidden = {a, b, m1, m2}
        result: List[FrozenSet[int]] = []

        max_depth = max_len - 4
        if max_depth < 2:
            return result

        def dfs(path: List[str], visited: Set[str]):
            depth = len(path)
            last = path[-1]

            if 2 <= depth <= max_depth:
                if m2 in self.adj.get(last, ()):
                    nodes = {a, b, m1, m2, *path}
                    if len(nodes) != depth + 4:
                        return

                    eids: List[int] = []
                    for u, v in itertools.combinations(nodes, 2):
                        eid = self.edge_index_map.get((u, v))
                        if eid is not None:
                            eids.append(eid)

                    # induced cycle: edges == nodes
                    if len(eids) == len(nodes):
                        result.append(frozenset(eids))
                        if len(result) >= limit:
                            raise StopIteration

            if depth >= max_depth:
                return

            for nxt in self.adj.get(last, ()):
                if nxt in visited or nxt in forbidden:
                    continue
                dfs(path + [nxt], visited | {nxt})

        try:
            for x in self.adj.get(m1, ()):
                if x in forbidden:
                    continue
                dfs([x], {m1, x})
        except StopIteration:
            pass

        return result

    # -----------------------------
    # guided supplement
    # -----------------------------
    def guided_supplement(
        self,
        *,
        base_cycles_by_len: Dict[int, Set[FrozenSet[int]]],
        min_len: int,
        max_len: int,
        per_edge_limit: int,
    ) -> Dict[int, Set[FrozenSet[int]]]:
        if min_len > max_len or per_edge_limit <= 0:
            return {}

        # 1) Collect covered edges
        used_edges: Set[Tuple[str, str]] = set()
        for cycles in base_cycles_by_len.values():
            for cyc in cycles:
                for eid in cyc:
                    uv = self.eid_to_edge[eid]
                    if uv is not None:
                        used_edges.add(uv)

        # 2) Uncovered edges
        rest_edges = [e for e in self.undirected_edges if e not in used_edges]
        rest_edges.sort()

        added: Dict[int, Set[FrozenSet[int]]] = defaultdict(set)

        # 3) Supplement cycles for each uncovered edge using guided dirs
        for (a0, b0) in rest_edges:
            a, b = (a0, b0) if a0 < b0 else (b0, a0)

            dirs = self.dfs_dirs_by_edge.get((a, b))
            if not dirs:
                continue

            got = 0
            for (m1, m2) in dirs:
                found_cycles = self.dfs_find_c_cycles_upto(
                    a=a, b=b, m1=m1, m2=m2,
                    max_len=max_len,
                    limit=per_edge_limit,
                )
                for cyc in found_cycles:
                    L = len(cyc)
                    if L < min_len or L > max_len:
                        continue
                    added[L].add(cyc)
                    got += 1
                    if got >= per_edge_limit:
                        break
                if got >= per_edge_limit:
                    break

        return added


# =====================================================
# Simple + GF(2) independent selector
# =====================================================
class SimpleIndependentCycleSelector:
    def __init__(self, cycles_by_len, eid_to_edge, beta, max_len, undirected_edge2_single_str, debug_print=None):
        self.cycles_by_len = cycles_by_len
        self.eid_to_edge = eid_to_edge
        self.beta = beta
        self.max_len = max_len
        self.undirected_edge2_single_str = undirected_edge2_single_str
        self.debug_print = debug_print or (lambda *args, **kwargs: None)

        self.independent_cycles: List[FrozenSet[int]] = []
        self._basis: Dict[int, int] = {}  # pivot_bit -> reduced_mask

        # --- perf counters ---
        self.perf_simple_scanned = 0
        self.perf_indep_added = 0

    def _is_simple_cycle(self, cyc: FrozenSet[int]) -> bool:
        undirected_edge2_single_str = self.undirected_edge2_single_str
        nodes: Set[str] = set()

        for eid in cyc:
            if eid <= 0 or eid >= len(self.eid_to_edge):
                return False
            uv = self.eid_to_edge[eid]
            if uv is None:
                return False
            u, v = uv
            nodes.add(u)
            nodes.add(v)
            if len(nodes) > self.max_len:
                return False

        nodes_sorted = sorted(nodes)
        cycle_lines = tuple(itertools.combinations(nodes_sorted, 2))
        edge_set = set(cycle_lines)

        # induced simple ring: edge count == node count in subgraph
        if len(edge_set & undirected_edge2_single_str) != len(nodes_sorted):
            return False
        return True

    def _cycle_to_mask(self, cyc: FrozenSet[int]) -> int:
        m = 0
        for eid in cyc:
            m ^= (1 << eid)
        return m

    def _reduce_mask(self, m: int) -> int:
        while m:
            pivot = m & -m
            base = self._basis.get(pivot)
            if base is None:
                break
            m ^= base
        return m

    def _try_add_independent(self, cyc: FrozenSet[int]) -> bool:
        m = self._cycle_to_mask(cyc)
        r = self._reduce_mask(m)
        if r == 0:
            return False
        pivot = r & -r
        self._basis[pivot] = r
        self.independent_cycles.append(cyc)
        return True

    def run(self):
        # 1) Collect simple cycles (sorted by length) + deduplicate
        simple_set: Set[FrozenSet[int]] = set()
        simple_list: List[FrozenSet[int]] = []
        for L in sorted(self.cycles_by_len):
            for cyc in self.cycles_by_len[L]:
                if cyc in simple_set:
                    continue
                if self._is_simple_cycle(cyc):
                    simple_set.add(cyc)
                    simple_list.append(cyc)

        self.debug_print("==================================")
        self.debug_print("Phase 2 Step1: Simple cycle filtering + deduplication complete")
        self.debug_print("  Total simple cycles:", len(simple_list))
        self.debug_print("  Target beta:", self.beta)
        self.debug_print("==================================")

        # 2) GF(2) linear independence check: scan all simple cycles
        for cyc in simple_list:
            self.perf_simple_scanned += 1
            added = self._try_add_independent(cyc)
            if added:
                self.perf_indep_added += 1
            if added and len(self.independent_cycles) > self.beta:
                raise RuntimeError(
                    f"GF(2) independent cycles exceed beta: "
                    f"{len(self.independent_cycles)} > {self.beta}"
                )

        is_complete = (len(self.independent_cycles) == self.beta)

        self.debug_print("==================================")
        self.debug_print("Phase 2 Step2: GF(2) independence selection complete")
        self.debug_print("  independent(simple) count:", len(self.independent_cycles))
        self.debug_print("  beta:", self.beta)
        self.debug_print("  Beta satisfied:", is_complete)
        self.debug_print("==================================")

        return self.independent_cycles, is_complete


# =====================================================
# Cycle basis completion builder
# =====================================================
class CycleBasisBuilder:
    def __init__(
        self,
        *,
        adjacency_map: Dict[str, Set[str]],
        edge_index_map: Dict[Tuple[str, str], int],
        eid_to_edge: List[Optional[Tuple[str, str]]],
        dfs_dirs_by_edge: Optional[Dict[Tuple[str, str], Tuple[Tuple[str, str], ...]]] = None,
        dfs_finder: Optional[GuidedDFSFinder] = None,
        max_len: int = 9,
        use_fundamental_fallback: bool = True,
    ):
        self.adj = adjacency_map
        self.edge_index_map = edge_index_map
        self.eid_to_edge = eid_to_edge

        self.dfs_dirs_by_edge = dfs_dirs_by_edge or {}
        self.dfs_finder = dfs_finder
        self.max_len = max_len
        self.use_fundamental_fallback = use_fundamental_fallback

        # spanning tree (for fundamental)
        self.parent: Dict[str, Optional[str]] = {}
        self.parent_eid: Dict[str, Optional[int]] = {}
        self.depth: Dict[str, int] = {}
        self.tree_edge_eids: Set[int] = set()

        # --- O(1) tree-path mask support ---
        # component root id + XOR mask from root to node (GF(2) edge-incidence)
        self.comp_root: Dict[str, str] = {}
        self.root_xor_mask: Dict[str, int] = {}  # node -> int mask

        # perf counters (init for safety)
        self.perf_fallback_tried = 0
        self.perf_fallback_added = 0

    # -------- spanning tree (BFS) --------
    def build_spanning_tree(self) -> None:
        visited: Set[str] = set()

        def get_eid(u: str, v: str) -> Optional[int]:
            return self.edge_index_map.get((u, v)) or self.edge_index_map.get((v, u))

        for root in self.adj:
            if root in visited:
                continue

            visited.add(root)
            self.parent[root] = None
            self.parent_eid[root] = None
            self.depth[root] = 0
            self.comp_root[root] = root
            self.root_xor_mask[root] = 0

            q = deque([root])
            while q:
                x = q.popleft()
                for y in self.adj.get(x, ()):
                    if y in visited:
                        continue
                    visited.add(y)

                    self.parent[y] = x
                    eid = get_eid(x, y)
                    self.parent_eid[y] = eid
                    self.depth[y] = self.depth[x] + 1
                    self.comp_root[y] = root

                    # xor-mask from root to y
                    if eid is None:
                        self.root_xor_mask[y] = self.root_xor_mask.get(x, 0)
                    else:
                        self.tree_edge_eids.add(eid)
                        self.root_xor_mask[y] = self.root_xor_mask.get(x, 0) ^ (1 << eid)

                    q.append(y)


    # -------- tree path eids (u -> v) --------
    def _tree_path_eids(self, u: str, v: str) -> List[int]:
        """Return the list of tree-edge eids along the unique tree path between u and v."""
        eids: List[int] = []
        uu: Optional[str] = u
        vv: Optional[str] = v

        # lift deeper side
        while uu is not None and vv is not None and self.depth[uu] > self.depth[vv]:
            pe = self.parent_eid.get(uu)
            if pe is not None:
                eids.append(pe)
            uu = self.parent.get(uu)

        while uu is not None and vv is not None and self.depth[vv] > self.depth[uu]:
            pe = self.parent_eid.get(vv)
            if pe is not None:
                eids.append(pe)
            vv = self.parent.get(vv)

        while uu is not None and vv is not None and uu != vv:
            e1 = self.parent_eid.get(uu)
            e2 = self.parent_eid.get(vv)
            if e1 is not None:
                eids.append(e1)
            if e2 is not None:
                eids.append(e2)
            uu = self.parent.get(uu)
            vv = self.parent.get(vv)

        return eids

    # -------- fundamental basis (simple cycles) --------
    def build_fundamental_basis(self) -> List[FrozenSet[int]]:
        """
        Build a (simple) fundamental cycle basis using the current spanning forest.
        For each non-tree edge (chord), add chord + unique tree path between its endpoints.

        This yields an independent cycle basis of size |E|-|V|+cc (assuming the tree was built).
        """
        if not self.parent:
            self.build_spanning_tree()

        basis: List[FrozenSet[int]] = []

        for eid in range(1, len(self.eid_to_edge)):
            uv = self.eid_to_edge[eid]
            if uv is None:
                continue
            if eid in self.tree_edge_eids:
                continue  # tree edge -> no fundamental cycle
            u, v = uv
            path_eids = self._tree_path_eids(u, v)
            cyc = frozenset([eid, *path_eids])
            basis.append(cyc)

        return basis

    # -------- tree path mask (u -> v) --------
    def _tree_path_mask(self, u: str, v: str) -> int:
        """GF(2) tree path mask using root-prefix XOR (O(1) big-int XOR).

        In a rooted tree, XOR(root->u) ^ XOR(root->v) equals XOR(u<->v) path.
        Works component-wise; if u,v are not in the same spanning-tree component, returns 0.
        """
        ru = self.comp_root.get(u)
        rv = self.comp_root.get(v)
        if ru is None or rv is None or ru != rv:
            return 0
        return self.root_xor_mask.get(u, 0) ^ self.root_xor_mask.get(v, 0)

    @staticmethod
    def _cycle_to_mask(cyc: FrozenSet[int]) -> int:
        m = 0
        for eid in cyc:
            m ^= (1 << eid)
        return m

    @staticmethod
    def _mask_to_cycle(mask: int) -> FrozenSet[int]:
        eids = set()
        while mask:
            lsb = mask & -mask
            eids.add(lsb.bit_length() - 1)
            mask ^= lsb
        return frozenset(eids)

    def complete_cycle_basis(
        self,
        *,
        basis_cycles: List[FrozenSet[int]],
        target_rank: int,
    ) -> Tuple[List[FrozenSet[int]], int]:

        basis: Dict[int, int] = {}
        basis_out: List[FrozenSet[int]] = []
        basis_set: Set[FrozenSet[int]] = set()

        # --- perf counters (reset per call) ---
        self.perf_fallback_tried = 0
        self.perf_fallback_added = 0

        def reduce_mask(m: int) -> int:
            while m:
                p = m & -m
                b = basis.get(p)
                if b is None:
                    break
                m ^= b
            return m

        def try_add_cycle_with_mask(cyc: FrozenSet[int], m: int) -> bool:
            """
            Unified entry point:
            - cyc: The concrete simple cycle to store (for output)
            - m: The GF(2) mask for this cycle (for independence check + pivot)
            """
            if cyc in basis_set:
                return False
            r = reduce_mask(m)
            if r == 0:
                return False
            basis[r & -r] = r
            basis_set.add(cyc)
            basis_out.append(cyc)
            return True

        def try_add_cycle(cyc: FrozenSet[int]) -> bool:
            return try_add_cycle_with_mask(cyc, self._cycle_to_mask(cyc))

        # 1) existing cycles
        for cyc in basis_cycles:
            try_add_cycle(cyc)
            if len(basis_out) >= target_rank:
                return basis_out[:target_rank], 0

        added = 0

        # 2) guided DFS (guide-dict priority)
        covered_edges: Set[Tuple[str, str]] = set()
        for cyc in basis_out:
            for eid in cyc:
                uv = self.eid_to_edge[eid]
                if uv is not None:
                    covered_edges.add(uv)

        if self.dfs_finder and self.dfs_dirs_by_edge:
            for (a, b), dirs in self.dfs_dirs_by_edge.items():
                if (a, b) in covered_edges or (b, a) in covered_edges:
                    continue
                for (m1, m2) in dirs:
                    found = self.dfs_finder.dfs_find_c_cycles_upto(
                        a=a, b=b, m1=m1, m2=m2,
                        max_len=self.max_len, limit=1
                    )
                    if not found:
                        continue
                    if try_add_cycle(found[0]):
                        added += 1
                        for eid in found[0]:
                            uv = self.eid_to_edge[eid]
                            if uv is not None:
                                covered_edges.add(uv)
                        if len(basis_out) >= target_rank:
                            return basis_out[:target_rank], added
                        break

        # 3) fundamental fallback
        if self.use_fundamental_fallback:
            if not self.parent:
                self.build_spanning_tree()

            # Avoid duplicates (u,v) vs (v,u): scan only unique undirected edges
            undirected_unique: Set[Tuple[str, str]] = set()
            for eid in range(1, len(self.eid_to_edge)):
                uv = self.eid_to_edge[eid]
                if uv is None:
                    continue
                u, v = uv
                a, b = (u, v) if u < v else (v, u)
                undirected_unique.add((a, b))

            for (u, v) in undirected_unique:
                eid = self.edge_index_map.get((u, v))
                if eid is None:
                    continue
                if eid in self.tree_edge_eids:
                    continue

                mask = self._tree_path_mask(u, v) ^ (1 << eid)

                # Fundamental cycle is naturally simple (tree path + chord)
                cyc = self._mask_to_cycle(mask)

                self.perf_fallback_tried += 1
                if try_add_cycle_with_mask(cyc, mask):
                    added += 1
                    self.perf_fallback_added += 1

                    if len(basis_out) >= target_rank:
                        break

        return basis_out[:target_rank], added


# =====================================================
# App-level orchestration
# =====================================================
@dataclass(frozen=True)
class AppConfig:
    """Config container - all parameters set in main"""
    # Find induced chordless cycles only. True provides "cleaner" cycles but might miss some. Default: True.
    induced_only: bool = True
    # Min cycle length for DFS. Default: 6.
    dfs_min_len: int = 6
    # Max cycle length for DFS. Increase to find longer cycles but slower. Suggested [7, 10]. Default: 9.
    dfs_max_len: int = 9
    # Limit number of cycles added per uncovered edge via DFS. Default: 1.
    dfs_per_edge_limit: int = 1
    # Max strict edge count for memory safety.


    max_edges: int = 800000
    # Output node IDs as integers (remove leading zeros). Default: True.
    output_pad_to_int: bool = True
    # Fast fundamental threshold: Large graphs might trigger fast heuristics.
    fast_fundamental_threshold: int = 12000
    # Force use of fundamental basis only. Fast but produces longer cycles. Better for huge/complex graphs. Default: False.
    force_fundamental_only: bool = False
    # Use DFS guidance in fundamental basis algo to improve cycle quality. Default: True.
    fundamental_use_dfs_guidance: bool = True
    # Use fundamental basis fallback after partial basis construction. Default: True.
    fundamental_use_fallback: bool = True
    # Build spanning tree for fundamental basis support. Default: True.
    fundamental_build_spanning_tree: bool = True
    # Print detailed debug/progress info. Default: False.
    verbose: bool = True
    # Log filename for debug output.
    debug_log_file: Optional[str] = None
    # Generate separate log file for each run. Default: False.
    generate_individual_log: bool = False
    # Delete individual logs after consolidation (used with --task=consolidate_logs). Default: True.
    delete_after_consolidate: bool = True
    
    # Heavy Edge optimization: skip exhaustive structure search for nodes with too many neighbors
    # to avoid O(N^2) explosion. 0 means disabled.
    heavy_edge_threshold: int = 64

    # New: Callback interfaces
    on_progress: Optional[Callable[[str, float], None]] = None
    on_complete: Optional[Callable[[Dict], None]] = None


class CycleBasisApp:
    def __init__(self, config: Optional[AppConfig] = None):
        self.cfg = config or AppConfig()
        self.debug_output = []
        self.progress_data = {}
        
    def _progress(self, stage: str, progress: float = 0.0) -> None:
        """Progress callback"""
        self.progress_data[stage] = progress
        if self.cfg.on_progress:
            self.cfg.on_progress(stage, progress)
        
    def _debug_print(self, *args, **kwargs) -> None:
        """Unified debug printer"""
        msg = " ".join(str(arg) for arg in args)
        self.debug_output.append(msg)
        
        if self.cfg.verbose:
            print(*args, **kwargs)
            
    def _save_debug_log(self, log_file: Optional[str] = None) -> None:
        """Save debug log to file"""
        if log_file is None:
            log_file = self.cfg.debug_log_file
            
        if log_file and self.debug_output:
            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(self.debug_output))
                if self.cfg.verbose:
                    print(f"[DEBUG] Log saved to: {log_file}")
            except Exception as e:
                print(f"❌ Failed to save log: {e}", file=sys.stderr)
                
    def consolidate_logs(self, directory: str, consolidated_log_file: str) -> None:
        """Scan directory, consolidate all .delog files, and optionally delete them."""
        self._debug_print(f"Start consolidating logs in directory: {directory}")
        log_files = [f for f in os.listdir(directory) if f.endswith(".delog")]

        if not log_files:
            self._debug_print("No .delog files found.")
            return

        self._debug_print(f"Found {len(log_files)} .delog files: {log_files}")

        # Open consolidated file in append mode
        with open(consolidated_log_file, "a", encoding="utf-8") as outfile:
            for filename in sorted(log_files):
                filepath = os.path.join(directory, filename)
                outfile.write(f"\n{'='*20} Source: {filename} {'='*20}\n\n")
                try:
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                    outfile.write("\n\n")
                except Exception as e:
                    outfile.write(f"*** Failed to read file: {filepath}, Error: {e} ***\n\n")

        self._debug_print(f"All logs consolidated to: {consolidated_log_file}")

        if self.cfg.delete_after_consolidate:
            self._debug_print("Deleting individual .delog files...")
            deleted_count = 0
            for filename in log_files:
                filepath = os.path.join(directory, filename)
                try:
                    os.remove(filepath)
                    deleted_count += 1
                except Exception as e:
                    self._debug_print(f"Failed to delete file: {filepath}, Error: {e}")
            self._debug_print(f"Successfully deleted {deleted_count} .delog files.")
        else:
            self._debug_print("Individual .delog files preserved.")

    # -------- I/O --------
    def _read_input(self, input_file: str) -> List[List[str]]:
        with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [line.strip() for line in f if line.strip()]
        lines = [re.split(r"[,\s]+", line) for line in raw_lines]
        lines = [list(filter(None, row)) for row in lines]
        return lines

    def run_from_file(self, input_file: str, output_file: str) -> None:
        """Run full pipeline from file input"""
        self._progress("start", 0.0)
        self._debug_print(f"Processing: {input_file}")
        
        start_time = time.time()
        
        # Determine log filename for this run
        log_file_to_use = self.cfg.debug_log_file
        if self.cfg.generate_individual_log:
            base_name = os.path.basename(input_file)
            file_name_without_ext = os.path.splitext(base_name)[0]
            log_file_to_use = f"{file_name_without_ext}.delog"

        try:
            lines = self._read_input(input_file)
            self._progress("read_input", 0.1)

            result_eids, beta, processor = self._execute_pipeline(lines)
            
            # Reconstruct node order
            result_cycles: List[List[str]] = []
            for cyc_eids in result_eids:
                path = CycleUtils.reconstruct_cycle_order_from_eids(
                    list(cyc_eids), processor.eid_to_edge
                )
                if path:
                    result_cycles.append(path)
            
            self._progress("reconstruct_paths", 0.9)

            self._write_output(output_file, result_cycles)
            self._progress("write_output", 1.0)

        except Exception as e:
            self._debug_print(f"Process failed: {input_file}, Error: {e}")
            import traceback
            self._debug_print(traceback.format_exc())
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self._debug_print(f"Completed: {input_file}, Duration: {duration:.2f}s")
            
            final_stats = {
                "input_file": input_file,
                "output_file": output_file,
                "duration": duration,
                "progress": self.progress_data,
            }
            
            # Use the determined log filename
            if log_file_to_use:
                self._save_debug_log(log_file_to_use)
            
            self._complete(final_stats)

    def _complete(self, stats: Dict) -> None:
        """Completion callback"""
        if self.cfg.on_complete:
            self.cfg.on_complete(stats)
        
    def _read_input_from_text(self, text: str) -> List[List[str]]:
        """Read input from text string"""
        raw_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        lines = [re.split(r"[,\s]+", line) for line in raw_lines]
        lines = [list(filter(None, row)) for row in lines]
        return lines

    def _write_output(self, output_file: str, result_cycles: List[List[str]]) -> None:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"{len(result_cycles)}\n")
            for cycle_nodes in result_cycles:
                if self.cfg.output_pad_to_int:
                    nodes_clean = [str(int(x)) for x in cycle_nodes]
                else:
                    nodes_clean = list(cycle_nodes)
                f.write(f"{len(nodes_clean)} {' '.join(nodes_clean)}\n")
                
    def _format_result(self, result_cycles: List[List[str]]) -> str:
        """Format result as string"""
        lines = [f"{len(result_cycles)}"]
        for cycle_nodes in result_cycles:
            if self.cfg.output_pad_to_int:
                nodes_clean = [str(int(x)) for x in cycle_nodes]
            else:
                nodes_clean = list(cycle_nodes)
            lines.append(f"{len(nodes_clean)} {' '.join(nodes_clean)}")
        return "\n".join(lines)

    # -------- debug printer --------
    def _print_debug(self, processor: InputDataProcessor, cycles_by_len: Dict[int, Set[FrozenSet[int]]], beta: int) -> None:
        self._debug_print(" ")
        self._debug_print("=========== ENUM DEBUG INFO ===========")
        self._debug_print("DEBUG F_SUMMARY:", len(processor.F_SUMMARY))
        self._debug_print("DEBUG H_SUMMARY:", len(processor.H_SUMMARY))
        self._debug_print("DEBUG F ∩ H:", len(processor.F_SUMMARY & processor.H_SUMMARY))
        self._debug_print("DEBUG dfs_dirs_by_edge:", len(processor.dfs_dirs_by_edge))
        self._debug_print("==================================")
        for k in sorted(cycles_by_len):
            self._debug_print(f"Cycle count C{k}: {len(cycles_by_len[k])} cycles")
        self._debug_print("==================================")
        self._debug_print("Target Rank (beta):", beta)
        self._debug_print("==================================")

    def _execute_pipeline(self, lines: List[List[str]]) -> Tuple[List[FrozenSet[int]], int, InputDataProcessor]:
        """Execute core pipeline, returning (basis, beta, processor)"""
        processor = InputDataProcessor(lines, heavy_edge_threshold=self.cfg.heavy_edge_threshold)
        self._progress("build_processor", 0.2)

        edge_cnt = len(processor.undirected_edge2_single_str)
        
        if edge_cnt > self.cfg.max_edges:
            raise MemoryError(f"Edge count {edge_cnt} exceeds limit {self.cfg.max_edges}")

        beta = processor.minimal_cycle_count()
        self._progress("calc_beta", 0.3)

        if beta == 0:
            return [], 0, processor

        # -------- Force fundamental-only mode --------
        if self.cfg.force_fundamental_only:
            self._debug_print("Forcing fundamental-only mode")
            builder = CycleBasisBuilder(
                adjacency_map=processor.adjacency_map,
                edge_index_map=processor.edge_index_map,
                eid_to_edge=processor.eid_to_edge,
            )
            builder.build_spanning_tree()
            fundamental_basis = builder.build_fundamental_basis()
            return fundamental_basis[:beta], beta, processor

        # -------- Phase 1: Enumerate C3-C5 + Guided DFS for C6-C9 --------
        cycles_by_len = processor.enumerate_cycles(induced_only=self.cfg.induced_only)
        self._progress("enum_c3_c5", 0.5)
        
        if self.cfg.verbose:
            self._print_debug(processor, cycles_by_len, beta)

        finder = GuidedDFSFinder(
            adjacency_map=processor.adjacency_map,
            edge_index_map=processor.edge_index_map,
            eid_to_edge=processor.eid_to_edge,
            undirected_edges=processor.undirected_edge2_single_str,
            dfs_dirs_by_edge=processor.dfs_dirs_by_edge,
        )
        
        added_cycles = finder.guided_supplement(
            base_cycles_by_len=cycles_by_len,
            min_len=self.cfg.dfs_min_len,
            max_len=self.cfg.dfs_max_len,
            per_edge_limit=self.cfg.dfs_per_edge_limit,
        )
        self._progress("guided_dfs", 0.7)

        for k, v in added_cycles.items():
            cycles_by_len[k].update(v)

        # -------- Phase 2: Independent Cycle Selection --------
        selector = SimpleIndependentCycleSelector(
            cycles_by_len=cycles_by_len,
            eid_to_edge=processor.eid_to_edge,
            beta=beta,
            max_len=self.cfg.dfs_max_len,
            undirected_edge2_single_str=processor.undirected_edge2_single_str,
            debug_print=self._debug_print,
        )
        independent_cycles, is_complete = selector.run()
        self._progress("select_independent", 0.8)

        # -------- Phase 3: Complete the basis --------
        if not is_complete:
            self._debug_print("Basis incomplete, supplementing...")
            builder = CycleBasisBuilder(
                adjacency_map=processor.adjacency_map,
                edge_index_map=processor.edge_index_map,
                eid_to_edge=processor.eid_to_edge,
                dfs_finder=finder,
                dfs_dirs_by_edge=processor.dfs_dirs_by_edge,
                max_len=self.cfg.dfs_max_len,
            )
            completed_basis, added_count = builder.complete_cycle_basis(
                basis_cycles=independent_cycles,
                target_rank=beta,
            )
            self._debug_print(f"Supplement complete, added {added_count} cycles.")
            independent_cycles = completed_basis
        
        self._progress("complete_basis", 0.85)

        return independent_cycles, beta, processor


    def run_on_networkx_graph(self, G) -> List[List[Any]]:
        """
        Execute pipeline directly on a NetworkX graph.
        
        Parameters
        ----------
        G : networkx.Graph
            Input graph
            
        Returns
        -------
        list[list[Any]]
            List of cycles, where each cycle is a list of node IDs (original type).
        """
        start_time = time.time()
        self._progress("start", 0.0)
        
        # 1. Adapt NetworkX graph to internal lines format (List[List[str]])
        # Node mapping: original obj -> internal str
        # We need to map back at the end.
        
        # We'll use str() for internal processing, but keep a map if nodes aren't strings
        # However, InputDataProcessor expects "lines" of connectivity. 
        # A simpler way is to construct InputDataProcessor directly or adapt input lines.
        # Let's produce the "lines" format: each line is "u v".
        
        # Map original nodes to strings safely
        node_map_to_str = {}
        node_map_from_str = {}
        
        # To ensure consistent string representation (e.g. for integers), handle them carefully.
        # The easiest way: Assign a unique integer ID to each node first, then stringify.
        
        for i, n in enumerate(G.nodes()):
            s = str(i) # Use index as string ID
            node_map_to_str[n] = s
            node_map_from_str[s] = n
            
        lines: List[List[str]] = []
        for u, v in G.edges():
            lines.append([node_map_to_str[u], node_map_to_str[v]])
            
        try:
            result_eids, beta, processor = self._execute_pipeline(lines)
            
            # Reconstruct cycles
            result_cycles: List[List[Any]] = []
            for cyc_eids in result_eids:
                path_strs = CycleUtils.reconstruct_cycle_order_from_eids(
                    list(cyc_eids), processor.eid_to_edge
                )
                if path_strs:
                    # Map back to original nodes
                    # path_strs are stringified indices like "0", "1"... (with some padding from processor)
                    # InputDataProcessor pads with zeros! We need to handle that.
                    
                    # processor.node_index_map maps padded_str -> int_id (1-based)
                    # wait, processor rebuilds its own logic.
                    # The `path_strs` are keys in `processor.adjacency_map`.
                    # Let's look at `InputDataProcessor._build_graph`.
                    # It calls `zfill(pad_width)`. 
                    
                    # We need to unpad carefully.
                    # Since we fed "0", "1", "10", the processor might pad them to "00", "01", "10".
                    
                    # Robust mapping back:
                    # processor used input strings to create internal padded strings.
                    # But we don't easily know the mapping unless we track it or reverse `int(s)`.
                    # Our input to processor was `str(index)`. Processor converts to int, finds max, then zfills.
                    
                    cycle_nodes_original = []
                    for s in path_strs:
                        # s is padded string. int(s) gives the original index we passed.
                        idx = int(s)
                        # map back index -> original node
                        # We need to access the node by index from `enumerate(G.nodes())` order
                        # node_map_from_str uses "0", "1" (unpadded).
                        original_node = node_map_from_str[str(idx)]
                        cycle_nodes_original.append(original_node)
                    
                    result_cycles.append(cycle_nodes_original)
                    
            self._progress("complete", 1.0)
            return result_cycles
            
        except Exception as e:
            self._debug_print(f"Graph processing failed: {e}")
            import traceback
            self._debug_print(traceback.format_exc())
            return []

# =====================================================
# NetworkX Adapter (Drop-in replacement)
# =====================================================
def minimum_cycle_basis_heuristic(G, weight=None) -> List[List[Any]]:
    """
    Compute an approximate Minimum Cycle Basis (MCB) for a graph G.
    
    This function is designed to be a high-performance, drop-in replacement
    algorithm for `networkx.minimum_cycle_basis`, especially optimized for
    sparse, planar-like, or large graphs.
    
    Parameters
    ----------
    G : NetworkX Graph
        The input graph (undirected).
    weight : str, optional
        (Ignored) Edge attribute to use for edge weights. 
        Currently this heuristic assumes unweighted MCB (shortest length).

    Returns
    -------
    list[list]
         A list of cycles. Each cycle is a list of nodes.
    
    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.grid_2d_graph(10, 10)
    >>> basis = minimum_cycle_basis_heuristic(G)
    >>> len(basis)
    81
    """
    config = AppConfig(
        verbose=False, 
        induced_only=True,
        dfs_max_len=9
    )
    app = CycleBasisApp(config)
    return app.run_on_networkx_graph(G)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Find Minimum Cycle Basis (MCB) in a graph.")
    
    # Core Task Arguments
    parser.add_argument("--task", type=str, default="run", choices=["run", "consolidate_logs"],
                        help="Task to execute: 'run' (process input) or 'consolidate_logs'.")
    
    # 'run' task arguments
    parser.add_argument("--input", "-i", type=str, help="Input file path (required for task='run').")
    parser.add_argument("--output", "-o", type=str, help="Output file path (required for task='run').")
    
    # Positional args compatibility: python solution.py input_file output_file
    parser.add_argument("input_pos", nargs="?", help="Input file path (positional)")
    parser.add_argument("output_pos", nargs="?", help="Output file path (positional)")

    # 'consolidate_logs' task arguments
    parser.add_argument("--log_dir", type=str, default=".", 
                        help="Directory containing .delog files (for 'consolidate_logs').")
    parser.add_argument("--consolidated_log", type=str, default="consolidated_debug.log",
                        help="Consolidated log filename (for 'consolidate_logs').")

    # AppConfig dynamic arguments
    for field in fields(AppConfig):
        if field.name in ["on_progress", "on_complete"]:
            continue
        
        arg_name = f"--{field.name.replace('_', '-')}"
        
        # Handle type annotation being string (from __future__ import annotations)
        is_bool = field.type is bool or (isinstance(field.type, str) and field.type == "bool")
        
        if is_bool:
            # For boolean flags, use store_true and store_false pair (or similar)
            if field.default:
                parser.add_argument(f"--no-{field.name.replace('_', '-')}", dest=field.name, action="store_false", help=f"Disable {field.name}")
                # Keep -- arg, but ensure it doesn't conflict
                parser.set_defaults(**{field.name: True})
            else:
                parser.add_argument(arg_name, dest=field.name, action="store_true", help=f"Enable {field.name}")
        else:
            # Handle int, float, str
            arg_type = str
            raw_type = field.type
            
            if raw_type is int or (isinstance(raw_type, str) and raw_type == "int"):
                arg_type = int
            elif raw_type is float or (isinstance(raw_type, str) and raw_type == "float"):
                arg_type = float

            parser.add_argument(arg_name, type=arg_type, default=field.default, help=f"Set {field.name} (default: {field.default})")

    args = parser.parse_args()

    # Pass parsed args to AppConfig
    config_dict = {f.name: getattr(args, f.name) for f in fields(AppConfig) if f.name in args}
    config = AppConfig(**config_dict)
    
    app = CycleBasisApp(config)

    if args.task == "run":
        input_file = args.input or args.input_pos
        output_file = args.output or args.output_pos
        
        if not input_file or not output_file:
            parser.error("Must specify input and output files (via positional args or --input/--output).")
        app.run_from_file(input_file, output_file)
    elif args.task == "consolidate_logs":
        # Clear consolidated log file once before consolidation
        if os.path.exists(args.consolidated_log):
            open(args.consolidated_log, 'w').close()
        app.consolidate_logs(args.log_dir, args.consolidated_log)
    else:
        parser.error(f"Unknown task: {args.task}")


if __name__ == "__main__":
    import argparse
    from dataclasses import fields
    main()