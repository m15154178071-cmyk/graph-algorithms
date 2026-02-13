# MCB Project (Minimum Cycle Basis and Graph Analysis)

[中文说明](#项目介绍) | [English Description](#english-description)

---

## <a id="项目介绍"></a>项目介绍

这是一个基于 Python 的图论分析工具，专门用于无向图的结构分析。主要功能包括构建图的邻接表、识别特定长度的环（如 C3、C4 结构），以及提取图的最小环基（Minimum Cycle Basis）。该工具适用于需要对复杂网络结构进行拓扑分析的场景。

### 能处理什么任务？ (Supported Tasks)

本项目可以处理以下图分析任务：

#### 1. **最小环基（Minimum Cycle Basis, MCB）计算**
   - 提取图的最小环基（一组线性无关的环，总长度最小）
   - 支持大规模稀疏图的高效计算
   - 可作为 NetworkX 的 `minimum_cycle_basis()` 的高性能替代方案

#### 2. **短环结构检测**
   - **三角形（C3）检测**：识别图中所有的三角形结构
   - **四边形（C4）检测**：识别图中所有的四边形结构  
   - **五边形（C5）检测**：识别图中所有的五边形结构
   - 支持诱导环（无弦环，chordless cycles）的筛选

#### 3. **长环发现**
   - 使用引导式 DFS 高效发现 C6-C9 长度的环
   - 针对中大规模图进行性能优化
   - 可配置搜索深度和每条边的搜索限制

#### 4. **图的拓扑分析**
   - 计算图的连通分量数量
   - 计算图的环基维度（beta = |E| - |V| + connected_components）
   - 构建图的生成树（Spanning Tree）

#### 5. **环的重建与可视化准备**
   - 从边 ID 集合重建完整的节点顺序
   - 验证环的简单性（每个节点度数为 2）
   - 输出标准格式便于后续可视化或分析

#### 6. **数据输入/输出处理**
   - 从文件读取边列表（支持多种分隔符：逗号、空格等）
   - 自动节点编号和格式化（zero-padding）
   - 输出环基结果到文件

### 主要功能

1.  **图构建与预处理**：
    -   支持从输入数据读取边列表，自动处理节点编号并进行格式化（zero-padding）。
    -   构建高效的邻接表和边索引映射。
2.  **环基提取 (Cycle Basis Extraction)**：
    -   实现了提取线性无关环的算法。
    -   支持基于位掩码（Bitmask）的线性无关性检测，确保提取的环构成图的基。
3.  **短环结构分析**：
    -   专门针对 C3（三角形）和 C4（四边形）等短环结构进行识别和分类。
4.  **路径与环还原**：
    -   提供了从边 ID 集合还原节点顺序的功能，便于可视化或进一步分析。

### 应用场景示例 (Use Cases)

以下是一些典型的应用场景：

1. **网络拓扑分析**：分析计算机网络、社交网络中的环路结构
2. **化学分子分析**：识别有机化合物中的环状结构（如苯环、环己烷等）
3. **电路分析**：检测电路图中的基本回路
4. **道路网络规划**：分析城市道路网络中的环形路线
5. **数据结构优化**：为图数据库或导航系统优化存储结构
6. **算法研究**：作为其他图算法的基础工具（如平面性测试、着色问题等）

### 如何使用

该项目主要是作为一个算法脚本运行。

#### 依赖环境
- Python 3.8+ (建议)
- 仅依赖 Python 标准库（`collections`, `itertools`, `typing` 等），无需安装额外的第三方包。

#### 运行方式
通常该脚本通过标准输入或文件读取图数据。

**基础版 (Basic Version)**
适用于一般规模的图结构分析。
```bash
python solution_basic.py
```

**增强版 (Enhanced Version)**
`solution_enhanced.py` 是高级版本，包含以下改进：
- **引导式搜索 (Guided DFS)**：大大提高了查找长环（C6+）的效率。
- **诱导环支持**：支持筛选无弦环（Induced Cycles）。
- **性能优化**：针对中大规模稀疏图进行了深度优化。
- **NetworkX 对比**：内置了与 NetworkX 库的对比验证（如果安装了 NetworkX）。

```bash
# 运行增强版算法（如果不修改代码，默认运行内置的 Demo 和验证流程）
python solution_enhanced.py
```

---

## <a id="english-description"></a>English Description

This project is a Python-based graph theory analysis tool designed for analyzing the structure of undirected graphs. Its core functionalities include building graph adjacency lists, identifying specific cycle structures (like C3, C4), and extracting the Minimum Cycle Basis (MCB) of a graph. It is suitable for scenarios requiring topological analysis of complex networks.

### What Tasks Can It Handle? (Supported Tasks)

This project can handle the following graph analysis tasks:

#### 1. **Minimum Cycle Basis (MCB) Computation**
   - Extract the minimum cycle basis of a graph (a set of linearly independent cycles with minimum total length)
   - Efficiently handles large-scale sparse graphs
   - Can serve as a high-performance replacement for NetworkX's `minimum_cycle_basis()`

#### 2. **Short Cycle Structure Detection**
   - **Triangle (C3) Detection**: Identify all triangle structures in the graph
   - **Quadrilateral (C4) Detection**: Identify all quadrilateral structures
   - **Pentagon (C5) Detection**: Identify all pentagon structures
   - Support for filtering induced cycles (chordless cycles)

#### 3. **Long Cycle Discovery**
   - Efficiently discover cycles of length C6-C9 using guided DFS
   - Performance-optimized for medium-to-large graphs
   - Configurable search depth and per-edge search limits

#### 4. **Graph Topology Analysis**
   - Calculate the number of connected components
   - Calculate the cycle rank (beta = |E| - |V| + connected_components)
   - Build spanning tree structures

#### 5. **Cycle Reconstruction & Visualization Preparation**
   - Reconstruct complete node sequences from edge ID sets
   - Validate cycle simplicity (each node has degree 2)
   - Output in standard format for subsequent visualization or analysis

#### 6. **Data Input/Output Processing**
   - Read edge lists from files (supports multiple delimiters: comma, space, etc.)
   - Automatic node numbering and formatting (zero-padding)
   - Output cycle basis results to files

### Features

1.  **Graph Construction & Preprocessing**:
    -   Reads edge lists from input, automatically handling node indexing and formatting (zero-padding).
    -   Builds efficient adjacency maps and edge index mappings.
2.  **Cycle Basis Extraction**:
    -   Implements algorithms to extract linearly independent cycles.
    -   Uses bitmask-based linear independence detection to ensure the extracted cycles form a basis for the graph.
3.  **Short Cycle Analysis**:
    -   Specialized identification and classification for short cycle structures like C3 (triangles) and C4 (quadrilaterals).
4.  **Path & Cycle Reconstruction**:
    -   Provides functionality to reconstruct node sequences from sets of Edge IDs, facilitating visualization or further analysis.

### Use Cases (Application Scenarios)

Here are some typical application scenarios:

1. **Network Topology Analysis**: Analyze loop structures in computer networks or social networks
2. **Chemical Molecular Analysis**: Identify cyclic structures in organic compounds (e.g., benzene rings, cyclohexane, etc.)
3. **Circuit Analysis**: Detect fundamental loops in circuit diagrams
4. **Road Network Planning**: Analyze circular routes in urban road networks
5. **Data Structure Optimization**: Optimize storage structures for graph databases or navigation systems
6. **Algorithm Research**: Serve as a foundational tool for other graph algorithms (e.g., planarity testing, coloring problems, etc.)

### How to Use

The project is primarily designed to run as an algorithm script.

#### Requirements
- Python 3.8+ (Recommended)
- Depends only on the Python Standard Library (`collections`, `itertools`, `typing`, etc.), no external packages required.

#### Usage
Typically, the script reads graph data via standard input or files.

**Basic Version**
Suitable for general graph structure analysis.
```bash
python solution_basic.py
```

**Enhanced Version**
`solution_enhanced.py` is the advanced version including:
- **Guided DFS**: Significantly improves efficiency for finding long cycles (C6+).
- **Induced Cycle Support**: Supports filtering for chordless cycles.
- **Performance Optimization**: Deeply optimized for medium-to-large sparse graphs.
- **NetworkX Verification**: Includes built-in comparison with NetworkX (if installed).

```bash
# Run the enhanced algorithm (Runs built-in demo and verification by default)
python solution_enhanced.py
```
