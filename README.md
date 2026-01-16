# MCB Project (Minimum Cycle Basis and Graph Analysis)

[中文说明](#项目介绍) | [English Description](#english-description)

---

## <a id="项目介绍"></a>项目介绍

这是一个基于 Python 的图论分析工具，专门用于无向图的结构分析。主要功能包括构建图的邻接表、识别特定长度的环（如 C3、C4 结构），以及提取图的最小环基（Minimum Cycle Basis）。该工具适用于需要对复杂网络结构进行拓扑分析的场景。

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
