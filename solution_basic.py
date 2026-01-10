import os
# ===== locale 兜底（华为评测必加）=====
os.environ['LC_ALL'] = 'C'
os.environ['LANG'] = 'C'
import sys
import itertools
from collections import Counter
from collections import defaultdict
from typing import Any
from collections import deque
# import re
# import time


def reconstruct_cycle_order_from_eids(cycle_eids, eid_to_edge):
    """
    从一组 eids 还原一个合法 simple cycle 的节点顺序
    返回: [v1, v2, ..., v1]
    """
    adj = defaultdict(list)
    for eid in cycle_eids:
        u, v = eid_to_edge[eid]
        adj[u].append(v)
        adj[v].append(u)

    if not adj:
        return None

    # simple cycle：每个点度应为 2
    for x, ns in adj.items():
        if len(ns) != 2:
            return None

    start = next(iter(adj))
    path = [start]
    prev = None
    curr = start

    while True:
        n0, n1 = adj[curr]
        if prev is None:
            nxt = n0
        else:
            nxt = n0 if n1 == prev else n1

        if nxt == start:
            path.append(start)
            break

        # 防死循环（异常数据兜底）
        if nxt in path:
            return None

        path.append(nxt)
        prev, curr = curr, nxt

    return path

def extract_independent_cycle_basis_bitmask(cycles_by_len, target_rank=None):
    """
    输入: cycles_by_len[k] = set[frozenset[eid]]
    输出: list[frozenset[eid]] 线性无关环（按短到长取）
    """
    basis = {}        # pivot_bit -> reduced_mask
    out = []
    out_set = set()

    def reduce_mask(m: int) -> int:
        while m:
            pivot = m & -m
            b = basis.get(pivot)
            if b is None:
                break
            m ^= b
        return m

    for k in sorted(cycles_by_len.keys()):
        for cyc in cycles_by_len[k]:
            if cyc in out_set:
                continue
            m = 0
            for eid in cyc:
                m ^= (1 << eid)

            r = reduce_mask(m)
            if not r:
                continue

            pivot = r & -r
            basis[pivot] = r
            out.append(cyc)
            out_set.add(cyc)

            if target_rank is not None and len(out) >= target_rank:
                return out[:target_rank]

    return out

# ===== cycle type constants =====

class InputDataProcessor:
    """
    统一管线（最终可用版）：

    - 无向图 + adjacency(set)
    - edge_index_map：无向边 -> eid（int, 1-based）
    - 路径生长法统一枚举 C3 / C4 / C5 / C6 / ...
    - 环用 frozenset(eids) 表示，天然去重
    - 可选 induced_only：只保留无弦环（诱导环）
    """

    # =====================================================
    # 初始化
    # =====================================================
    def __init__(self, lines):
        self.lines = lines
        self.debug: bool = False
        self.pad_width: int | None = None

        # 图结构
        self.undirected_edge4_set_str: set[tuple[str, str, str, str]] = set()
        self.undirected_edge2_set_str: set[tuple[str, str]] = set()
        self.adjacency_map: dict[str, set[str]] = {}

        # 边编号
        self.edge_index_map: dict[tuple[str, str], int] = {}
        self.eid_to_edge: list[Any] = [None]  # 1-based，占位用 None
        self.node_index_map: dict[str, int] = {}
        self.count_to_node: dict[str, tuple[int, int]] = {}
        self.dict_edge_to_cycles: dict[tuple[str, str], tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]]]] = defaultdict(lambda: (set(), set(), set(), set()))
        self.dict_edgebreak_center : dict[tuple[str, str], tuple[set[str], set[tuple[str, str]]]] = defaultdict(lambda: (set(), set()))

        self._build_graph()
        self._build_edge_index()
        self._build_node_index()
        self.partial_C3_structure = set()
        self.partial_C4_structure = set()
        self.C3_structure = set()
        self.C4_structure = set()
        self.open_cycle_nodes = set()
        self.E_SUMMARY = set()
        self.F_SUMMARY = set()
        self.G_SUMMARY = set()
        self.H_SUMMARY = set()
    # =====================================================
    # 构建无向图 + 邻接表
    # =====================================================
    def _build_graph(self):
        edges_int: list[tuple[int, int]] = []

        for row in self.lines:
            nums = list(map(int, row))
            u = min(nums)
            v = max(nums)
            if u != v:
                edges_int.append((u, v))

        max_node = max(max(u, v) for u, v in edges_int)
        self.pad_width = len(str(max_node))

        E4: set[tuple[str, str, str, str]] = set()
        E2: set[tuple[str, str]] = set()
        E2_single: set[tuple[str, str]] = set()
        for u, v in edges_int:
            su = str(u).zfill(self.pad_width)
            sv = str(v).zfill(self.pad_width)
            if su > sv:
                su, sv = sv, su
            E4.add((su, sv, sv, su))  # 4-tuple for easier cycle node extraction
            E2.add((su, sv))
            E2.add((sv, su))
            E2_single.add((su, sv))

        self.undirected_edge4_set_str = E4
        self.undirected_edge2_set_str = E2
        self.undirected_edge2_single_str = E2_single

        adj: dict[str, set[str]] = defaultdict(set)
        for u, v in E2_single:
            adj[u].add(v)
            adj[v].add(u)

        self.adjacency_map = dict(adj)

    # =====================================================
    # 构建 edge_index_map（无向边唯一 eid）
    # =====================================================

    def _build_edge_index(self):
        self.edge_index_map = {}
        self.eid_to_edge = [None]  # 1-based，占位

        # undirected_edge2_single_str 中的边已经是：
        # - 字符串
        # - zfill 后
        # - 且保证 u < v
        for u, v in sorted(self.undirected_edge2_single_str):
            eid = len(self.eid_to_edge)

            # 正反两个方向都映射到同一个 eid
            self.edge_index_map[(u, v)] = eid
            self.edge_index_map[(v, u)] = eid

            # eid_to_edge 只存一次无向边
            self.eid_to_edge.append((u, v))
        return self.edge_index_map, self.eid_to_edge

    
    # =====================================================
    # 构建 node_index_map（附加信息）
    # =====================================================
    def _build_node_index(self):
        self.node_index_map = {}
        self.count_to_node = {}

        nodes = set()
        for u, v in self.undirected_edge2_single_str:
            nodes.add(u)
            nodes.add(v)

        for node in sorted(nodes):
            nid = len(self.node_index_map) + 1
            self.node_index_map[node] = nid
            deg = len(self.adjacency_map.get(node, set()))
            self.count_to_node[node] = (nid, deg)

        return nodes, self.node_index_map, self.count_to_node
    # =====================================================

    def minimal_cycle_count(self):
        minimal = len(self.undirected_edge2_single_str) - len(self.node_index_map) + 1
        return minimal

    # induced cycle 判定（可选）
    # =====================================================
    def _is_induced_cycle_nodes(self, node_seq: tuple[str, ...]) -> bool:
        nodes = set(node_seq)
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

    def _is_induced_cycle_partical(self, node_seq: tuple[str, ...], num: int) -> bool:
        nodes = sorted(set(node_seq))
        node_set_target = set(nodes)

        cycle_tuple = tuple(itertools.combinations(nodes, num))
        cycle_tuple_commons = set(cycle_tuple) & self.undirected_edge2_single_str

        if len(cycle_tuple_commons) != len(nodes) - (num - 1):
            return False

        # ===== 第一层：节点集合校验（你说的“set 去重用途”）=====
        node_tuple = tuple(x for rel in cycle_tuple_commons for x in rel)
        if set(node_tuple) != node_set_target:
            return False

        # ===== 第二层：度数形状校验 =====
        freq = Counter(node_tuple)

        if num == 2:
            degs = sorted(freq.values())
            return degs.count(1) == 2 and degs.count(2) == len(nodes) - 2

        return False

    def open_path_close_state_L3(self, edges_tuple):
        """
        通用 OPEN-k 收口判定（最多 2 层有效）

        返回值：
            1  -> 1 层可收口
            2  -> 2 层可收口
            0  -> ≥3 层或不可收口
        """

        # ---------- 1. 从边反推节点 ----------
        nodes = set()
        for u, v in edges_tuple:
            nodes.add(u)
            nodes.add(v)

        if len(nodes) != len(edges_tuple) + 1:
            return 0

        # ---------- 2. 统计度数 ----------
        deg = defaultdict(int)
        for u, v in edges_tuple:
            deg[u] += 1
            deg[v] += 1

        endpoints = [n for n in nodes if deg[n] == 1]
        if len(endpoints) != 2:
            return 0

        a, b = endpoints

        # ---------- 3. 禁止走原路径边 ----------
        path_edge_set = set()
        for u, v in edges_tuple:
            path_edge_set.add((u, v))
            path_edge_set.add((v, u))

        # ---------- 4. BFS 分层（只关心 1、2 层） ----------
        visited = {a}
        frontier = {a}

        for depth in (1, 2):   # ⚠️ 只跑 1、2 层
            next_frontier = set()

            for x in frontier:
                for y in self.adjacency_map.get(x, []):
                    if (x, y) in path_edge_set:
                        continue

                    if y == b:
                        return depth   # 1 或 2，语义正好

                    if y not in visited:
                        visited.add(y)
                        next_frontier.add(y)

            if not next_frontier:
                break

            frontier = next_frontier

        # ≥3 层 or 不可达
        return 0

    def extract_path_endpoints(self, edges_tuple):
        """
        从一条 simple path（OPEN-k）的边集合中提取两端接口 (a, b)

        参数：
            edges_tuple : tuple[(u, v), ...]

        返回：
            (a, b)  -> 两端端点（顺序不重要）
            None    -> 不是合法 simple path
        """

        # ---------- 1. 收集节点 ----------
        nodes = set()
        for u, v in edges_tuple:
            nodes.add(u)
            nodes.add(v)

        # simple path 必须满足 |V| = |E| + 1
        if len(nodes) != len(edges_tuple) + 1:
            return None

        # ---------- 2. 统计度数 ----------
        deg = defaultdict(int)
        for u, v in edges_tuple:
            deg[u] += 1
            deg[v] += 1

        # ---------- 3. 找端点 ----------
        endpoints = [n for n in nodes if deg[n] == 1]

        if len(endpoints) != 2:
            return None

        a, b = endpoints
        return a, b

    # =====================================================
    def dfs_find_c_cycles_upto_9(self, a, b, m1, m2, edge_index, max_len=9, limit=1):
        forbidden = {a, b, m1, m2}
        result = []

        # 最大中间路径长度 t = max_len - 4
        max_depth = max_len - 4
        if max_depth < 2:
            return result

        def dfs(path, visited):
            """
            path: 当前路径（从 m1 开始）
            visited: 已访问节点集合
            """
            # 当前路径长度（不含 m1）
            depth = len(path)

            last = path[-1]

            # ===== 尝试收口到 m2 =====
            if depth >= 2 and depth <= max_depth:
                if m2 in self.adjacency_map.get(last, ()):
                    nodes = {a, b, m1, m2, *path}
                    if len(nodes) != depth + 4:
                        return

                    # 无弦判定：边数 == 点数
                    eid_list = []
                    for u, v in itertools.combinations(nodes, 2):
                        eid = edge_index.get((min(u, v), max(u, v)))
                        if eid is not None:
                            eid_list.append(eid)

                    if len(eid_list) == len(nodes):
                        result.append(frozenset(eid_list))
                        if len(result) >= limit:
                            raise StopIteration

            # ===== 深度剪枝 =====
            if depth >= max_depth:
                return

            # ===== 继续 DFS =====
            for nxt in self.adjacency_map.get(last, ()):
                if nxt in visited or nxt in forbidden:
                    continue
                dfs(path + [nxt], visited | {nxt})

        try:
            for x in self.adjacency_map.get(m1, ()):
                if x in forbidden:
                    continue
                dfs([x], {m1, x})
        except StopIteration:
            pass

        return result

    # =====================================================
    # 核心：统一路径生长枚举
    def enumerate_cycles(self):
        E2_single = self.undirected_edge2_single_str
        E2_single_list = list(E2_single)
        cycles_by_len: dict[int, set[frozenset[int]]] = defaultdict(set)
        dict_edge_to_cycles = self.dict_edge_to_cycles
        edge_index = self.edge_index_map
        E2_single = self.undirected_edge2_single_str

        edge_set = set()
        E_SUMMARY = self.E_SUMMARY
        F_SUMMARY = self.F_SUMMARY
        G_SUMMARY = self.G_SUMMARY
        H_SUMMARY = self.H_SUMMARY
        dict_edge_to_cycles = self.dict_edge_to_cycles
        for i in range(0, len(E2_single_list)):
            edge = E2_single_list[i]
            edge_set = set(edge)
            edge_start, edge_end = edge[0], edge[-1]
            edge = (min(edge_start, edge_end), max(edge_start, edge_end))

            # set()不能动
            nodes_start_connections = self.adjacency_map.get(edge_start, set()) - edge_set
            nodes_end_connections = self.adjacency_map.get(edge_end, set()) - edge_set
            nodes_connected_all = (nodes_start_connections | nodes_end_connections) - edge_set

            # 用于C3，边的另一端直接相连某一点
            nodes_connected_common = nodes_start_connections & nodes_end_connections
            # set()要转成tuple并排序，才能做后续组合
            nodes_start_only = nodes_start_connections - nodes_connected_common
            nodes_end_only = nodes_end_connections - nodes_connected_common
            nodes_connected_unique = nodes_start_only | nodes_end_only

            # 全集合
            A = set(itertools.combinations(sorted(nodes_connected_all), 2)) - E2_single
            # start only 集合
            B = set(itertools.combinations(sorted(nodes_start_only), 2))
            # end only 集合
            C = set(itertools.combinations(sorted(nodes_end_only), 2))
            # unique 集合，至少和一点相连
            D = set(itertools.combinations(sorted(nodes_connected_unique), 2))
            # 与点两侧都相连的点对集合
            E = (B | C) & E2_single
            # 存储 C3中间结构
            F = ((B | C) - E) - E2_single                
            # 与边两端都相连的点对集合，去除直连其中一点的集合
            G = ((D - B - C) & E2_single)
            # 存储 C4中间结构
            H = ((A - B - C) - G)
            for node_pair in H:
                node1, node2 = node_pair
                node1_connections = self.adjacency_map.get(node1, set())
                node2_connections = self.adjacency_map.get(node2, set())
                node_common_connections = node1_connections & node2_connections - edge_set
                if len(node_common_connections) > 0:
                    for mid_node in node_common_connections:
                        nodes_in_cycle = sorted((edge_start, node1, mid_node, node2, edge_end))
                        eid_list = []
                        eids = itertools.combinations(nodes_in_cycle, 2)
                        for u, v in eids:
                            eid = edge_index.get((min(u, v), max(u, v)), None)
                            if eid is not None:
                                eid_list.append(eid)
                        if len(eid_list) != len(nodes_in_cycle):
                            continue
                        eids_in_cycle = tuple(sorted(eid_list))
                        cycles_by_len[len(eids_in_cycle)].add(frozenset(eids_in_cycle))
            dict_edge_to_cycles[edge] = (E, F, G, H)
            E_SUMMARY |= E
            F_SUMMARY |= F
            G_SUMMARY |= G
            H_SUMMARY |= H
            E = set()
            F = set()
            G = set()
            H = set()

        for node_pair in E2_single:
            u, v = node_pair
            node_pair_set = set(node_pair)
            edge_start, edge_end = node_pair[0], node_pair[-1]
            edge_point = (min(edge_start, edge_end), max(edge_start, edge_end))
            edge_start = edge_point[0]
            edge_end = edge_point[-1]
            E, F, G, H = dict_edge_to_cycles[edge_point]
            # set()不能动
            nodes_start_connections = self.adjacency_map.get(edge_start, set()) - node_pair_set
            nodes_end_connections = self.adjacency_map.get(edge_end, set()) - node_pair_set
            nodes_connected_all = (nodes_start_connections | nodes_end_connections) - node_pair_set

            # 用于C3，边的另一端直接相连某一点
            nodes_connected_common = nodes_start_connections & nodes_end_connections
            # set()要转成tuple并排序，才能做后续组合
            nodes_start_only = nodes_start_connections - nodes_connected_common
            nodes_end_only = nodes_end_connections - nodes_connected_common
            nodes_connected_unique = nodes_start_only | nodes_end_only

            for mid_node in nodes_connected_common:
                nodes_in_cycle = sorted((edge_start, mid_node, edge_end))
                eid_list = []
                eids = itertools.combinations(nodes_in_cycle, 2)
                for u, v in eids:
                    eid = edge_index.get((min(u, v), max(u, v)), None)
                    if eid is not None:
                        eid_list.append(eid)
                if len(eid_list) != len(nodes_in_cycle):
                    continue
                eids_in_cycle = tuple(sorted(eid_list))
                cycles_by_len[len(eids_in_cycle)].add(frozenset(eids_in_cycle))

            # C4
            if len(G) >= 3:
                continue
            # set()不能动
            for node_pair in G:
                u, v = node_pair
                nodes_in_cycle = sorted((edge_start, u, v, edge_end))
                eid_list = []
                eids = itertools.combinations(nodes_in_cycle, 2)
                for x, y in eids:
                    eid = edge_index.get((min(x, y), max(x, y)), None)
                    if eid is not None:
                        eid_list.append(eid)
                if len(eid_list) != len(nodes_in_cycle):
                    continue
                eids_in_cycle = tuple(sorted(eid_list))
                cycles_by_len[len(eids_in_cycle)].add(frozenset(eids_in_cycle))
        self.E_SUMMARY = E_SUMMARY
        self.F_SUMMARY = F_SUMMARY
        self.G_SUMMARY = G_SUMMARY
        self.H_SUMMARY = H_SUMMARY

        cycles_count = 0
        for k in cycles_by_len.keys():
            cycles_count += len(cycles_by_len[k])

        for node_pair in F_SUMMARY & H_SUMMARY:
            u, v = node_pair
            node_pair_set = set(node_pair)
            edge_start, edge_end = node_pair[0], node_pair[-1]
            edge_point = (min(edge_start, edge_end), max(edge_start, edge_end))
            edge_start = edge_point[0]
            edge_end = edge_point[-1]
            nodes_start_connections = self.adjacency_map.get(edge_start, set()) - node_pair_set
            nodes_end_connections = self.adjacency_map.get(edge_end, set()) - node_pair_set
            nodes_connected_all = (nodes_start_connections | nodes_end_connections) - node_pair_set
            nodes_connected_common = nodes_start_connections & nodes_end_connections
            # set()要转成tuple并排序，才能做后续组合
            if len(nodes_connected_common) == 0:
                continue
            nodes_start_only = nodes_start_connections - nodes_connected_common
            nodes_end_only = nodes_end_connections - nodes_connected_common
            nodes_connected_unique = nodes_start_only | nodes_end_only
            D = set(itertools.combinations(sorted(nodes_connected_unique), 2))
            # 与点两侧都相连的点对集合
            i_set = D & E2_single
            if len(i_set) == 1:
                for mid_node in nodes_connected_common:
                    for node_pair_2 in i_set:
                        node1, node2 = node_pair_2
                        nodes_in_cycle = sorted((edge_start, node1, mid_node, node2, edge_end))
                        eid_list = []
                        eids = itertools.combinations(nodes_in_cycle, 2)
                        for x, y in eids:
                            eid = edge_index.get((min(x, y), max(x, y)), None)
                            if eid is not None:
                                eid_list.append(eid)
                        if len(eid_list) != len(nodes_in_cycle):
                            continue
                        eids_in_cycle = tuple(sorted(eid_list))
                        cycles_by_len[len(eids_in_cycle)].add(frozenset(eids_in_cycle))
        # 1) 去重（其实 set 本来就去重了，这段可省略）
        for k, v in list(cycles_by_len.items()):
            cycles_by_len[k] = set(v)
        common_set = F_SUMMARY & H_SUMMARY

        # 2) 统计“已被环覆盖的边对”
        used_edge_pairs = set()
        for k, v in cycles_by_len.items():
            for cycle_eids in v:
                for eid in cycle_eids:
                    u, w = self.eid_to_edge[eid]  # 注意：这里存的是 (u,v) 且 u<v
                    used_edge_pairs.add((u, w))   # 用无向规范化边对

        # 3) 剩余“未覆盖”的边对
        E2_single_rest_set = E2_single - used_edge_pairs

        # ---------- C6 DFS 补环 ----------
        for (a, b) in E2_single_rest_set:
            a, b = min(a, b), max(a, b)

            # dict_edge_to_cycles 里已经缓存过
            E, F, G, H = dict_edge_to_cycles[(a, b)]

            # 只从 C4 中间结构 (m1, m2) 出发
            for (m1, m2) in H & common_set:

                # 调用 DFS（最多找 1 个）
                c6_list = self.dfs_find_c_cycles_upto_9(
                    a, b,
                    m1, m2,
                    edge_index,
                    limit=1
                )

                if not c6_list:
                    continue
                cycle = c6_list[0]
                cycles_by_len[len(cycle)].add(cycle)

                break


        return cycles_by_len

    
    # =====================================================
    # 由结构对（原 F/H）生成候选环（按长度分桶）
    # =====================================================
    def build_struct_cycles_by_len(self, max_len: int = 5):
        """
        从结构信息生成候选环字典（只生成短环）
        返回: dict[int, set[frozenset[int]]]
        """
        struct_cycles_by_len = defaultdict(set)
        edge_index = self.edge_index_map

        # ===== 结构 C3（原 F_SUMMARY）=====
        if max_len >= 3:
            for (u, v) in self.F_SUMMARY:
                # 找公共邻点
                common = self.adjacency_map[u] & self.adjacency_map[v]
                for x in common:
                    e1 = edge_index.get((u, x))
                    e2 = edge_index.get((v, x))
                    e3 = edge_index.get((u, v))
                    if e1 and e2 and e3:
                        struct_cycles_by_len[3].add(
                            frozenset((e1, e2, e3))
                        )

        # ===== 结构 C4 / C5（暂不展开，留接口位）=====
        # if max_len >= 4:
        #     ...

        return struct_cycles_by_len

    # =====================================================
    # 管线入口
    # =====================================================
    def run_pipeline(self, max_len: int = 12, induced_only: bool = False):
        cycles_by_len = self.enumerate_cycles()

        if induced_only:
            # 仅保留无弦环
            induced_cycles_by_len: dict[int, set[frozenset[int]]] = defaultdict(set)

            for k in cycles_by_len.keys():
                for cycle_eids in cycles_by_len[k]:
                    # 必须用 set，避免节点重复
                    nodes_in_cycle = set()
                    for eid in cycle_eids:
                        u, v = self.eid_to_edge[eid]
                        nodes_in_cycle.add(u)
                        nodes_in_cycle.add(v)

                    if self._is_induced_cycle_nodes(tuple(nodes_in_cycle)):
                        induced_cycles_by_len[k].add(cycle_eids)

            cycles_by_len = induced_cycles_by_len

        # 截断到 max_len
        cycles_by_len = {k: v for k, v in cycles_by_len.items() if k <= max_len}

        return cycles_by_len


class CycleBasisBuilder:
    """
    Build a cycle basis of an undirected graph using:
    - BFS spanning tree
    - Fundamental cycles
    - GF(2) Gaussian elimination (bitmask version)

    Key invariants:
    - basis_reduced : int bitmask (for independence check only)
    - basis_out     : frozenset[int] (original simple cycles)
    """

    # =====================================================
    # Init
    # =====================================================
    def __init__(self, adjacency_map, edge_index_map, eid_to_edge):
        self.adj = adjacency_map
        self.edge_index_map = edge_index_map
        self.eid_to_edge = eid_to_edge

        self.parent = {}
        self.parent_eid = {}
        self.depth = {}
        self.tree_edge_eids = set()

    # =====================================================
    # 1. Build spanning tree / forest (BFS)
    # =====================================================
    def build_spanning_tree(self):
        visited = set()

        def get_eid(u, v):
            return self.edge_index_map.get((u, v)) or \
                   self.edge_index_map.get((v, u))

        for root in self.adj:
            if root in visited:
                continue

            visited.add(root)
            self.parent[root] = None
            self.parent_eid[root] = None
            self.depth[root] = 0

            q = deque([root])
            while q:
                x = q.popleft()
                for y in self.adj.get(x, ()):
                    if y not in visited:
                        visited.add(y)
                        self.parent[y] = x
                        eid = get_eid(x, y)
                        self.parent_eid[y] = eid
                        self.depth[y] = self.depth[x] + 1
                        if eid is not None:
                            self.tree_edge_eids.add(eid)
                        q.append(y)

    # =====================================================
    # 2. Tree path (u -> v): bitmask
    # =====================================================
    def tree_path_mask(self, u, v):
        mask = 0
        uu, vv = u, v

        while self.depth[uu] > self.depth[vv]:
            eid = self.parent_eid[uu]
            if eid is not None:
                mask ^= (1 << eid)
            uu = self.parent[uu]

        while self.depth[vv] > self.depth[uu]:
            eid = self.parent_eid[vv]
            if eid is not None:
                mask ^= (1 << eid)
            vv = self.parent[vv]

        while uu != vv:
            eid1 = self.parent_eid[uu]
            eid2 = self.parent_eid[vv]
            if eid1 is not None:
                mask ^= (1 << eid1)
            if eid2 is not None:
                mask ^= (1 << eid2)
            uu = self.parent[uu]
            vv = self.parent[vv]

        return mask

    # =====================================================
    # 2.5 Tree path length only (heuristic)
    # =====================================================
    def tree_path_len(self, u, v):
        uu, vv = u, v
        du, dv = self.depth[uu], self.depth[vv]
        length = 0

        while du > dv:
            uu = self.parent[uu]
            du -= 1
            length += 1

        while dv > du:
            vv = self.parent[vv]
            dv -= 1
            length += 1

        while uu != vv:
            uu = self.parent[uu]
            vv = self.parent[vv]
            length += 2

        return length

    # =====================================================
    # 3. Complete cycle basis (bitmask GF(2))
    # =====================================================
    def complete_cycle_basis(self, basis_cycles, target_rank):
        """
        Parameters
        ----------
        basis_cycles : list[frozenset[int]]
        target_rank  : int

        Returns
        -------
        basis_out : list[frozenset[int]]
        added_cnt : int
        """

        # pivot_bit -> mask
        basis = {}
        basis_out = []
        basis_set = set()

        def reduce_mask(m):
            while m:
                pivot = m & -m
                b = basis.get(pivot)
                if b is None:
                    break
                m ^= b
            return m

        # ---------- Step 1: existing cycles ----------
        for cyc in basis_cycles:
            if cyc in basis_set:
                continue

            m = 0
            for eid in cyc:
                m ^= (1 << eid)

            r = reduce_mask(m)
            if r:
                pivot = r & -r
                basis[pivot] = r
                basis_out.append(cyc)
                basis_set.add(cyc)

            if len(basis_out) >= target_rank:
                return basis_out[:target_rank], 0

        # ---------- Step 2: fundamental cycles ----------
        all_eids = set(self.edge_index_map.values())
        non_tree_edges = list(all_eids - self.tree_edge_eids)

        non_tree_edges.sort(
            key=lambda eid: self.tree_path_len(*self.eid_to_edge[eid])
        )

        added = 0
        for eid in non_tree_edges:
            if len(basis_out) >= target_rank:
                break

            u, v = self.eid_to_edge[eid]
            path_mask = self.tree_path_mask(u, v)
            fund_mask = path_mask ^ (1 << eid)

            r = reduce_mask(fund_mask)
            if not r:
                continue

            pivot = r & -r
            basis[pivot] = r

            # 还原为 frozenset[int]
            eids = set()
            tmp = fund_mask
            while tmp:
                lsb = tmp & -tmp
                eids.add(lsb.bit_length() - 1)
                tmp ^= lsb

            cyc = frozenset(eids)
            if cyc in basis_set:
                continue

            basis_out.append(cyc)
            basis_set.add(cyc)
            added += 1

        return basis_out[:target_rank], added


def main(input_file, output_file):
    import time
    import re
    debug_mode = False
    start_time = time.time()
    print("\n ")
    print("========================================= ")
    print(f"Processing input file: {input_file}")
    # =====================================================
    # 1. 读取与预处理输入
    # =====================================================
    with open(input_file, 'r') as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    lines = [re.split(r'[,\s]+', line) for line in raw_lines]
    lines = [list(filter(None, line)) for line in lines]

    # =====================================================
    # 2. 构建图处理器
    # =====================================================
    processor = InputDataProcessor(lines)

    MAX_EDGES = 1000000
    edge_cnt = len(processor.undirected_edge2_single_str)
    if edge_cnt > MAX_EDGES:
        raise RuntimeError(
            f"Abort: edge count {edge_cnt} exceeds limit {MAX_EDGES}"
        )

    beta = processor.minimal_cycle_count()


    # =====================================================
    # 3. 图信息统计（保留打印）
    # =====================================================
    # print(" ")
    # print("\n\n========= 图信息统计 =========")
    # print("输入文件 =", input_file)
    # print("输入边数 =", len(processor.undirected_edge2_single_str))
    # print("输入节点数 =", len(processor.node_index_map))
    # print("题目要求的最小环数 =", beta)

    # =====================================================
    # 4. 枚举候选环（一次可闭合）
    # =====================================================
    cycles_by_len = processor.run_pipeline(
        max_len=8,
        induced_only=True
    )

    # =====================================================
    # 结构环（来自结构信息）
    # =====================================================
    struct_cycles_by_len = processor.build_struct_cycles_by_len(max_len=5)
    if debug_mode:
        print("struct cycles:", {k: len(v) for k, v in struct_cycles_by_len.items()})

    # 合并到候选环集合
    for k, v in struct_cycles_by_len.items():
        cycles_by_len[k] |= v

    if debug_mode:
        print("DEBUG F_SUMMARY:", len(processor.F_SUMMARY))
        print("DEBUG H_SUMMARY:", len(processor.H_SUMMARY))
        print("DEBUG F ∩ H:", len(processor.F_SUMMARY & processor.H_SUMMARY))


    # =====================================================
    # 5. 打印各长度环数量（保留）
    # =====================================================
    for k in sorted(cycles_by_len.keys()):
        if debug_mode:
            print(f"最小环基类型数量  C{k}: {len(cycles_by_len[k])} cycles")
        pass

    # =====================================================
    # 6. 提取线性无关环基（GF(2)）
    # =====================================================
    independent_basis = extract_independent_cycle_basis_bitmask(cycles_by_len, target_rank=beta)

    # print(f"线性无关环基秩（补前）: {len(independent_basis)}")

    # =====================================================
    # >>> PATCH：补环逻辑
    # =====================================================
    if len(independent_basis) < beta:
        builder = CycleBasisBuilder(
            processor.adjacency_map,
            processor.edge_index_map,
            processor.eid_to_edge
        )
        builder.build_spanning_tree()

        independent_basis, added_cnt = builder.complete_cycle_basis(
            independent_basis,
            beta
        )

        print(f"补充的基础环数量: {added_cnt}")
        print(f"线性无关环基秩（补后）: {len(independent_basis)}")

    # =====================================================
    # >>> ASSERT：环基秩必须等于 beta（就在这里）
    # =====================================================
    assert len(independent_basis) == beta, \
        f"cycle basis rank mismatch: {len(independent_basis)} != {beta}"


    # =====================================================
    # 8. 环的节点顺序还原
    # =====================================================
    result_cycles = []
    for cycle_eids in independent_basis:
        node_seq = reconstruct_cycle_order_from_eids(list(cycle_eids), processor.eid_to_edge)
        if node_seq is not None:
            result_cycles.append(node_seq)
    if debug_mode:
        print("打印的环数(秩/目标):", len(independent_basis), "输出环数:", len(result_cycles))
    result_cycles.sort(key=lambda x: (len(x), x))

    # =====================================================
    # 9. 输出结果
    # =====================================================
    cycle_count = len(result_cycles)
    node_count = 0
    with open(output_file, 'w') as f:
        f.write(f"{len(result_cycles)}\n")
        for cycle_nodes in result_cycles:
            nodes_clean = [str(int(x)) for x in cycle_nodes]
            node_count += len(nodes_clean) - 1
            f.write(f"{len(nodes_clean)} {' '.join(nodes_clean)}\n")

    end_time = time.time()
    spend_time = end_time - start_time
    print(f"环数: {cycle_count}，节点数: {node_count}，边数: {edge_cnt}，最小环基数: {beta}，平均环长: {node_count / cycle_count:.2f}")
    print(f"总用时: {spend_time:.2f} 秒，输出文件: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python solution03.py <input_file> <output_file>")
        sys.exit(1)

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    main(sys.argv[1], sys.argv[2])

# python solution03.py edge1.txt output_file1.txt
# python solution03.py edge2.txt output_file2.txt
# python solution03.py edge3.txt output_file3.txt
# python solution03.py edge4.txt output_file4.txt
# python solution03.py edge5.txt output_file5.txt
# python solution03.py edge6.txt output_file6.txt
# python solution03.py edge7.txt output_file7.txt
# python solution03.py edge8.txt output_file8.txt
# python solution03.py edge9.txt output_file9.txt
# python solution03.py edge10.txt output_file10.txt
# python solution03.py edge11.txt output_file11.txt
# python solution03.py edge12.txt output_file12.txt
# python solution03.py edge13.txt output_file13.txt
# python solution03.py edge14.txt output_file14.txt
# python solution03.py edge15.txt output_file15.txt

