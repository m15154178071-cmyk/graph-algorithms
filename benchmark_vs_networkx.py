import time
import random
import os
import sys

try:
    import networkx as nx
except ImportError:
    print("错误: 未找到 networkx 库，请使用以下命令安装：")
    print("  pip install networkx")
    sys.exit(1)

try:
    import igraph as ig
except (ImportError, OSError) as e:
    print(f"警告: 无法加载 igraph 库 ({e})，无法对比 igraph 性能。")
    ig = None

try:
    import rustworkx as rx
except (ImportError, OSError):
    rx = None

# 引入你的 enhanced 版本逻辑
# 假设 solution_enhanced.py 在同一目录下，且我们要调用其中的 InputDataProcessor, SimpleIndependentCycleSelector 等
# 为了方便调用，我们稍微 wrap 一下你的代码逻辑
try:
    from solution_enhanced import InputDataProcessor, SimpleIndependentCycleSelector, CycleBasisBuilder, GuidedDFSFinder, CycleUtils
except ImportError:
    print("错误: 未找到 solution_enhanced.py，请确保该文件在当前目录下。")
    sys.exit(1)

def run_my_mcb_algorithm(edges, num_nodes):
    """
    封装你的算法调用过程。
    由于你的代码是基于 InputDataProcessor读取 lines 的，我们需要构造类似于文件输入的 lines 结构。
    """
    # 构造模拟输入：lines = [[u, v], [u, v], ...]
    # 注意：你的 InputDataProcessor 期望的是字符串或者是数字，我们统一转成 list of lists
    lines = [[str(u), str(v)] for u, v in edges]
    
    # 1. Pipeline 初始化
    processor = InputDataProcessor(lines)
    beta = processor.minimal_cycle_count()
    if beta == 0:
        return []

    # 2. 阶段1: 枚举 + 引导式 DFS
    # 这里模拟 solution_enhanced.py _execute_pipeline 的主要逻辑
    # 配置参数（硬编码模拟 AppConfig）
    cfg_induced_only = True
    cfg_dfs_min_len = 6
    cfg_dfs_max_len = 15 # 这里的 max_len 可以适当调大以应对随机图
    cfg_dfs_per_edge_limit = 3
    
    cycles_by_len = processor.enumerate_cycles(induced_only=cfg_induced_only)
    
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

    # 3. 阶段2: 独立环筛选
    selector = SimpleIndependentCycleSelector(
        cycles_by_len=cycles_by_len,
        eid_to_edge=processor.eid_to_edge,
        beta=beta,
        max_len=cfg_dfs_max_len,
        undirected_edge2_single_str=processor.undirected_edge2_single_str,
    )
    independent_cycles, is_complete = selector.run()

    # 4. 阶段3: 补全环基
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
    
    # 5. 为了对比，我们返回环的长度列表（总权重）
    # cycle 是 eids 的 frozenset，转成长度很简单
    return [len(c) for c in independent_cycles]

def benchmark_one_case(case_name, G):
    """
    运行单个测试用例
    case_name: 描述
    G: networkx Graph 对象
    """
    print(f"\n--- {case_name} ---")
    print(f"节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")

    # 准备数据
    edges = list(G.edges())
    # 你的算法需要节点映射到 1..N 这种或者字符串，这里直接用 G.edges() 的 int/str 都可以，
    # 只要 run_my_mcb 里的 InputProcessor 能处理。InputProcessor 处理 str(int) 没问题。

    # --- Run NetworkX ---
    start_nx = time.time()
    # networkx返回的是 list of list of nodes
    nx_basis = nx.minimum_cycle_basis(G)
    end_nx = time.time()
    
    nx_time = end_nx - start_nx
    nx_total_len = sum(len(c) for c in nx_basis)
    print(f"[NetworkX]   Time: {nx_time:.4f}s | Basis Weight: {nx_total_len} | Count: {len(nx_basis)}")

    # --- Run igraph (if available) ---
    ig_time = -1.0
    if ig is not None:
        try:
            # 转换 NetworkX 图到 igraph
            # 注意：igraph 的节点 ID 必须是连续整数 0..N-1，这里简单起见我们只传边列表
            # NetworkX 的 convert_node_labels_to_integers 通常已经保证了 0..N-1
            # 但为了通用，我们重新建立映射
            mapping = {n: i for i, n in enumerate(G.nodes())}
            ig_edges = [(mapping[u], mapping[v]) for u, v in edges]
            g_ig = ig.Graph(n=len(mapping), edges=ig_edges, directed=False)
            
            start_ig = time.time()
            # igraph 的 result 是 edge IDs 的 list of lists，或者 vertex lists
            # minimum_cycle_basis 方法返回 edge IDs
            # use_cycle_basis=True 可能会启用优化
            ig_basis_node_lists = g_ig.minimum_cycle_basis(use_cycle_basis=True)
            end_ig = time.time()
            
            ig_time = end_ig - start_ig
            # igraph 返回的是 node list，直接 len 即可
            ig_total_len = sum(len(c) for c in ig_basis_node_lists)
            print(f"[igraph (C)] Time: {ig_time:.4f}s | Basis Weight: {ig_total_len} | Count: {len(ig_basis_node_lists)}")

        except Exception as e:
            print(f"[igraph] Failed: {e}")

    # --- Run rustworkx (if available) ---
    rx_time = -1.0
    if rx is not None:
        try:
            # Construct rustworkx graph
            # mapping matches 0..N-1
            mapping_rx = {n: i for i, n in enumerate(G.nodes())}
            py_graph = rx.PyGraph()
            py_graph.add_nodes_from(range(len(G.nodes())))
            edge_list_rx = [(mapping_rx[u], mapping_rx[v]) for u, v in edges]
            py_graph.add_edges_from_no_data(edge_list_rx)
            
            start_rx = time.time()
            # Note: rx.cycle_basis is usually Fundamental Cycle Basis (FCB), NOT Minimum Cycle Basis (MCB).
            # It serves as a performance baseline for compiled graph traversal.
            rx_basis = rx.cycle_basis(py_graph)
            end_rx = time.time()
            
            rx_time = end_rx - start_rx
            rx_total_len = sum(len(c) for c in rx_basis)
            print(f"[rustworkx (Rust)] Time: {rx_time:.4f}s | Basis Weight: {rx_total_len} | Count: {len(rx_basis)} (Note: Likely FCB, not MCB)")

        except Exception as e:
            print(f"[rustworkx] Failed: {e}")

    # --- Run My Algo ---
    start_my = time.time()
    try:
        my_basis_lengths = run_my_mcb_algorithm(edges, G.number_of_nodes())
        end_my = time.time()
        
        my_time = end_my - start_my
        my_total_len = sum(my_basis_lengths)
        print(f"[My Algorithm] Time: {my_time:.4f}s | Basis Weight: {my_total_len} | Count: {len(my_basis_lengths)}")
        
        # --- Comparison ---
        speedup_nx = nx_time / my_time if my_time > 0 else 0.0
        print(f"Speedup vs NetworkX: {speedup_nx:.2f}x")
        
        if ig_time > 0:
            speedup_ig = ig_time / my_time if my_time > 0 else 0.0
            print(f"Speedup vs igraph:   {speedup_ig:.2f}x (My Algo is {'FASTER' if speedup_ig > 1 else 'slower'})")
        
        if rx_time > 0:
            speedup_rx = rx_time / my_time if my_time > 0 else 0.0
            print(f"Speedup vs rustworkx:{speedup_rx:.2f}x (My Algo is {'FASTER' if speedup_rx > 1 else 'slower'})")
        
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
