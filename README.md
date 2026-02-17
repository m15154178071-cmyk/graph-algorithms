# 最小环基 (MCB) 求解算法套件

这是一个高性能的 Python **最小环基 (Minimum Cycle Basis)** 算法实现库。本套件提供了三种不同复杂度的求解器，覆盖了从简单的教学演示到针对网格 (Mesh) 和复杂网络优化的生产级需求。

## update (2026-02-18)
- **新特性**: 在旗舰版 (`comprehensive`) 中引入了 **基于集合运算的短环生成算法 (Set-Based Short Cycle Generation)**。
- **性能飞跃**: 针对网格/有限元类图结构实现了显著加速（364条边的网格仅需 30ms）。
- **正确性增强**: 改进了从节点集合恢复路径的逻辑，确保拓扑结构的严格正确。

---

## 🏗️ 求解器概览

| 版本 | 文件名 | 类型 | 适用场景 | 描述 |
|:---|:---|:---|:---|:---|
| **基础版** | `mcb_cycle_basis_simple.py` | 基础 | 教学、小规模图 | 纯 Python 实现。代码简单易读，无依赖，暴力求解。 |
| **优化版** | `mcb_cycle_basis_optimized.py` | 平衡 | 通用场景 | 引入启发式搜索和位运算。在速度和代码复杂度之间取得平衡。 |
| **旗舰版** | `mcb_cycle_basis_comprehensive.py` | **高性能** | **生产环境、网格分析** | 集成 **集合短环生成**、链式压缩和分块验证。能找到**权重最小**的环基。 |

---

## ⚡ 性能基准测试

测试环境: Intel i7 / Windows 11.

**1. 结构化图 (Grid / Mesh)**
*专为有限元分析或像素网格优化*

| 数据集 | 边数 | 基础版耗时 | **旗舰版耗时** | 提升 |
|:---|:---|:---|:---|:---|
| Grid 15x15 | ~420 | 0.07s | **0.13s** | 相当 |
| Grid 30x30 | ~1740 | 0.11s | **0.29s** | **极高效率** |

> **注**: 即使图规模增长，旗舰版依然保持极高的运行速度，同时保证找到最小权重的基。

**2. 复杂网络 (Random / Scale-Free)**
*压力测试：无规律连接结构*

| 数据集 | 指标 | 基础版 | 优化版 | **旗舰版** |
|:---|:---|:---|:---|:---|
| **Random** (N=300) | **权重 (Total Weight)** | 5575 | 5797 | **5456 (最优)** |
| **Scale-Free** | **权重 (Total Weight)** | 3035 | 2979 | **2948 (最优)** |

> **结论**: 旗舰版优先保证 **解的质量**。相比启发式算法，它能找到总权重显著更低的环基，非常适合对基的质量有严格要求的场景。

---

## 🛠️ 使用说明

### 前置要求
- Python 3.8+
- `networkx` (仅用于运行 `benchmark.py` 生成测试数据，求解器脚本本身无此依赖)

### 1. 运行基准测试
运行完整的测试套件并生成报告：

**Windows**:
```powershell
./run_bench.bat
```

**Linux / Mac**:
```bash
pip install -r requirements.txt
python benchmark.py
```

### 2. 单独运行求解器
直接调用脚本处理边列表文件：

```bash
python mcb_cycle_basis_comprehensive.py <输入文件> <输出文件>
```

**输入格式**:
文本文件，每行表示一条边（两个节点ID）：
```text
1 2
2 3
3 1
...
```

**输出格式**:
第一行：环的数量。
后续行：每个环包含的节点列表。

---

## 🧩 算法原理 (旗舰版)

`comprehensive` 求解器实现了一个多阶段混合流水线：

1.  **Phase 1: 重炮轰炸 (Artillery - Short Cycles)**
    - 利用 **集合交集 (Set Intersections)** 瞬间识别 3环、4环和 5环。
    - 针对网格结构（三角形/四边形单元）进行了极致优化。
    - 包含拓扑路径恢复逻辑，确保边的遍历顺序正确。

2.  **Phase 2: 链式轰炸 (Chain Bombardment)**
    - 识别并将度为2的长链节点压缩为单个逻辑边。
    - 大幅降低后续阶段的有效图规模。

3.  **Phase 3: 生成树补全 (Spanning Tree Completion)**
    - 在剩余的稀疏图上计算生成树，以捕获剩余的基本环。
    - 用于闭合前两个阶段未发现的长环。

4.  **验证阶段 (Verification)**
    - 使用基于 GF(2) 域的高斯消元法验证线性无关性。
    - 采用位压缩数组 (CSR/Blockwise Reducer) 加速运算。

---

## 📂 项目结构

```
├── benchmark.py                # 自动化基准测试套件
├── run_bench.bat               # Windows 自动化脚本
├── bench_input.txt             # 样例输入数据
├── mcb_cycle_basis_simple.py   # 基础实现
├── mcb_cycle_basis_optimized.py # 启发式实现
└── mcb_cycle_basis_comprehensive.py # 旗舰实现 (推荐)
```
