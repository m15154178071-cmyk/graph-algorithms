
# =============================
# 最小环基（MCB）性能基准脚本
# 对比 NetworkX、rustworkx、你的算法
# =============================

import time  # 计时用
import random
import os
import sys


# 导入 NetworkX（纯Python图论库）
try:
    import networkx as nx
except ImportError:
    print("错误: 未找到 networkx 库，请使用以下命令安装：")
    print("  pip install networkx")
    sys.exit(1)


# 尝试导入 igraph（C实现，MCB，需VC++运行库，常见DLL问题）
try:
    import igraph as ig
except (ImportError, OSError) as e:
    print(f"警告: 无法加载 igraph 库 ({e})，无法对比 igraph 性能。")
    ig = None


# 尝试导入 rustworkx（Rust实现，极快，cycle_basis为FCB）
try:
    import rustworkx as rx
except (ImportError, OSError):
    rx = None


# 引入你的高性能 MCB 算法（solution_enhanced.py）
# 需确保同目录下有 solution_enhanced.py 文件
try:
    from solution_enhanced import InputDataProcessor, SimpleIndependentCycleSelector, CycleBasisBuilder, GuidedDFSFinder, CycleUtils
except ImportError:
    print("错误: 未找到 solution_enhanced.py，请确保该文件在当前目录下。")
    sys.exit(1)


# =============================
# 你的 MCB 算法主流程封装
# 输入：边列表、节点数
# 输出：每个环的长度（用于统计总权重）
# =============================
def run_my_mcb_algorithm(edges, num_nodes):
    """
    封装你的算法调用过程。
    由于你的代码是基于 InputDataProcessor 读取 lines 的，我们需要构造类似于文件输入的 lines 结构。
    """
    # 1. 构造模拟输入：lines = [[u, v], [u, v], ...]，全部转成字符串
    lines = [[str(u), str(v)] for u, v in edges]

    # 2. 初始化数据处理器，计算最小环基的理论数量 beta
    processor = InputDataProcessor(lines)
    beta = processor.minimal_cycle_count()
    if beta == 0:
        return []

    # 3. 阶段1：枚举短环 + 引导式DFS补充长环
    # 配置参数（可根据实际调整）
    cfg_induced_only = True  # 只枚举诱导环
    cfg_dfs_min_len = 6      # DFS枚举的最小环长
    cfg_dfs_max_len = 15     # DFS枚举的最大环长
    cfg_dfs_per_edge_limit = 3  # 每条边DFS补充环的数量上限

    # 3.1 枚举所有短环（C3~C5）
    cycles_by_len = processor.enumerate_cycles(induced_only=cfg_induced_only)

    # 3.2 用引导式DFS补充长环
    finder = GuidedDFSFinder(
        adjacency_map=processor.adjacency_map,
        edge_index_map=processor.edge_index_map,
        eid_to_edge=processor.eid_to_edge,
        undirected_edges=processor.undirected_edge2_single_str,
        dfs_dirs_by_edge=processor.dfs_dirs_by_edge,
    )
    added_cycles = finder.guided_supplement(
        base_cycles_by_len=cycles_by_len,
        min_len=cfg_dfs_min_len,
        max_len=cfg_dfs_max_len,
        per_edge_limit=cfg_dfs_per_edge_limit,
    )
    for k, v in added_cycles.items():
        cycles_by_len[k].update(v)

    # 4. 阶段2：独立环筛选（GF(2)消元，选出线性无关的环）
    selector = SimpleIndependentCycleSelector(
        cycles_by_len=cycles_by_len,
        eid_to_edge=processor.eid_to_edge,
        beta=beta,
        max_len=cfg_dfs_max_len,
        undirected_edge2_single_str=processor.undirected_edge2_single_str,
    )
    independent_cycles, is_complete = selector.run()

    # 5. 阶段3：如未满秩则用DFS补全环基
    if not is_complete:
        builder = CycleBasisBuilder(
            adjacency_map=processor.adjacency_map,
            edge_index_map=processor.edge_index_map,
            eid_to_edge=processor.eid_to_edge,
            dfs_finder=finder,
            dfs_dirs_by_edge=processor.dfs_dirs_by_edge,
            max_len=cfg_dfs_max_len,
        )
        completed_basis, _ = builder.complete_cycle_basis(
            basis_cycles=independent_cycles,
            target_rank=beta,
        )
        independent_cycles = completed_basis

    # 6. 返回每个环的长度（用于统计总权重）
    return [len(c) for c in independent_cycles]


# =============================
# 单个测试用例的基准测试流程
# 输入：用例名、networkx图对象
# =============================
def benchmark_one_case(case_name, G):
    """
    运行单个测试用例，分别用 NetworkX、igraph、rustworkx、你的算法求最小环基并计时。
    输出每种方法的用时、环基权重、环数，并对比加速比。
    """
    print(f"\n--- {case_name} ---")
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    # 1. 准备边数据
    edges = list(G.edges())
    # 你的算法需要节点映射到字符串或整数，InputProcessor 兼容 str(int)

    # 2. NetworkX 求最小环基
    start_nx = time.time()
    nx_basis = nx.minimum_cycle_basis(G)
    end_nx = time.time()
    nx_time = end_nx - start_nx
    nx_total_len = sum(len(c) for c in nx_basis)
    print(f"[NetworkX]   Time: {nx_time:.4f}s | Basis Weight: {nx_total_len} | Count: {len(nx_basis)}")

    # 3. igraph（如可用）
    ig_time = -1.0
    if ig is not None:
        try:
            # igraph 节点必须是 0..N-1，需重映射
            mapping = {n: i for i, n in enumerate(G.nodes())}
            ig_edges = [(mapping[u], mapping[v]) for u, v in edges]
            g_ig = ig.Graph(n=len(mapping), edges=ig_edges, directed=False)

            start_ig = time.time()
            # minimum_cycle_basis 通常返回边的索引列表，不需特殊参数
            ig_basis_node_lists = g_ig.minimum_cycle_basis()
            end_ig = time.time()

            ig_time = end_ig - start_ig
            ig_total_len = sum(len(c) for c in ig_basis_node_lists)
            print(f"[igraph (C)] Time: {ig_time:.4f}s | Basis Weight: {ig_total_len} | Count: {len(ig_basis_node_lists)}")

        except Exception as e:
            print(f"[igraph] Failed: {e}")

    # 4. rustworkx（如可用，FCB，仅作性能参考）
    rx_time = -1.0
    if rx is not None:
        try:
            # rustworkx 节点同样需 0..N-1
            mapping_rx = {n: i for i, n in enumerate(G.nodes())}
            py_graph = rx.PyGraph()
            py_graph.add_nodes_from(range(len(G.nodes())))
            edge_list_rx = [(mapping_rx[u], mapping_rx[v]) for u, v in edges]
            py_graph.add_edges_from_no_data(edge_list_rx)

            start_rx = time.time()
            # cycle_basis 返回 FCB，速度极快但权重不最优
            rx_basis = rx.cycle_basis(py_graph)
            end_rx = time.time()

            rx_time = end_rx - start_rx
            rx_total_len = sum(len(c) for c in rx_basis)
            print(f"[rustworkx (Rust)] Time: {rx_time:.4f}s | Basis Weight: {rx_total_len} | Count: {len(rx_basis)} (Note: Likely FCB, not MCB)")

        except Exception as e:
            print(f"[rustworkx] Failed: {e}")

    # 5. 你的算法
    start_my = time.time()
    try:
        my_basis_lengths = run_my_mcb_algorithm(edges, G.number_of_nodes())
        end_my = time.time()

        my_time = end_my - start_my
        my_total_len = sum(my_basis_lengths)
        print(f"[My Algorithm] Time: {my_time:.4f}s | Basis Weight: {my_total_len} | Count: {len(my_basis_lengths)}")

        # 6. 性能对比
        speedup_nx = nx_time / my_time if my_time > 0 else 0.0
        print(f"Speedup vs NetworkX: {speedup_nx:.2f}x")

        if ig_time > 0:
            speedup_ig = ig_time / my_time if my_time > 0 else 0.0
            print(f"Speedup vs igraph:   {speedup_ig:.2f}x (My Algo is {'FASTER' if speedup_ig > 1 else 'slower'})")

        if rx_time > 0:
            speedup_rx = rx_time / my_time if my_time > 0 else 0.0
            print(f"Speedup vs rustworkx:{speedup_rx:.2f}x (My Algo is {'FASTER' if speedup_rx > 1 else 'slower'})")

        # 7. 正确性校验
        if nx_total_len == my_total_len:
            print("✅ 正确性验证通过 (vs NX)")
        else:
            print(f"⚠️ 权重不一致 (vs NX)! Diff: {my_total_len - nx_total_len}")

    except Exception as e:
        print(f"[My Algorithm] Failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=========================================")
    print("Benchmark: My MCB vs NetworkX.minimum_cycle_basis")
    print("=========================================")

    # Case 1: 随机稀疏图 (Erdos-Renyi)
    # 节点少一点方便调试，边概率低一点模拟稀疏
    G1 = nx.erdos_renyi_graph(n=50, p=0.15, seed=42)
    # MCB 算法通常处理无向图，且不带自环和重边
    G1 = nx.Graph(G1) 
    G1.remove_edges_from(nx.selfloop_edges(G1))
    benchmark_one_case("Random Sparse Graph (n=50, p=0.15)", G1)

    # Case 2: 网格图 (Grid Graph) - 典型的 planar graph
    G2 = nx.grid_2d_graph(10, 10)
    # grid_2d_graph 节点是 (0,0) tuple，转成 int 或 str 方便统一处理
    G2 = nx.convert_node_labels_to_integers(G2)
    benchmark_one_case("Grid Graph (10x10)", G2)

    # Case 3: 稍大的随机图 (测试性能优势)
    # 你的算法在 dense graph 上可能有优势，或者在结构明显的图上有优势
    G3 = nx.erdos_renyi_graph(n=100, p=0.08, seed=2024)
    G3 = nx.Graph(G3)
    G3.remove_edges_from(nx.selfloop_edges(G3))
    benchmark_one_case("Random Graph (n=100, p=0.08)", G3)
    
    # Case 4: 轮图 (Wheel Graph) - 包含很多小环
    G4 = nx.wheel_graph(30)
    benchmark_one_case("Wheel Graph (n=30)", G4)

if __name__ == "__main__":
    main()
