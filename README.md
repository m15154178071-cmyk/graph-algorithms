# Minimum Cycle Basis Algorithm

This repository contains Python implementations for the **Minimum Cycle Basis (MCB)** problem in undirected graphs. It includes two versions suitable for different performance requirements and study purposes.

## Files

### 1. `solution_basic.py` (Prototype)
- **Description**: A basic implementation focusing on algorithmic logic.
- **Features**: 
  - Standard GF(2) Gaussian elimination.
  - Basic cycle enumeration (C3/C4).
  - Clean and flat structure, ideal for learning the core concepts.
- **Usage**: Good for small graphs and educational purposes.

### 2. `solution_enhanced.py` (Production Ready)
- **Description**: An optimized, engineered version designed for performance.
- **Features**: 
  - **O(1) Tree Path XOR**: Optimizes the fundamental cycle calculation using precomputed root-to-node XOR masks.
  - **C3-C5 Enumeration**: Explicitly enumerates cycles up to length 5.
  - **Guided DFS**: Uses structural information (H-structure) to guide Deep First Search for longer cycles (C6+).
  - **Robustness**: Includes strict "induced cycle" checks and memory management.
- **Usage**: Recommended for competitive programming (e.g., Huawei Soft Challenge) or processing medium-to-large graphs (10k - 500k edges).

## How to Run

```bash
python solution_enhanced.py input.txt output.txt
```

## Requirements
- Python 3.8+
- No external dependencies (standard library only).
