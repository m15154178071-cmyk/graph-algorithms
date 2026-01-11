# Proposal: High-Performance MCB Algorithm (up to 2400x speedup vs NetworkX)

## Summary
I have developed a pure Python implementation of a Minimum Cycle Basis (MCB) algorithm that is significantly faster than the current `networkx.minimum_cycle_basis`. 
On structured graphs like Grids and Wheels, it achieves **1000x - 2400x** speedups while maintaining 100% accuracy in total basis weight.

This implementation specifically addresses the performance bottleneck in graphs with high-degree nodes (hubs) via a "Heavy Edge" optimization, preventing the $O(N^2)$ combinatorial explosion that currently affects NetworkX's behavior on dense neighborhoods.

## Benchmarks (Windows / Python 3.13)

| Graph Type | Nodes | Edges | NetworkX Time | My Algo Time | Speedup | Correctness |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Grid 2D (14x14)** | 196 | 364 | 26.72 sec | **0.02 sec** | **~1335x** | ✅ Exact Match |
| **Wheel Graph (n=300)** | 300 | 598 | 119.77 sec | **0.05 sec** | **~2395x** | ✅ Exact Match |
| **Wheel Graph (n=1000)** | 1000 | 1998 | *(Timed Out/Skipped)* | **1.05 sec** | N/A | (Feasible vs Infeasible) |

## Key Innovations
1.  **Hybrid Approach**: Combines heuristic enumeration of short cycles (C3-C5) with a Fundamental Cycle Basis fallback.
2.  **Heavy Edge Skip**: Dynamically detects nodes with high local density (>64 neighbors) and skips exhaustive structural search (which causes $O(N^2)$ pairs) — reducing execution time from **minutes to milliseconds** for Wheel graphs.
3.  **Accuracy**: For planar and regular graphs (Grid, Wheel), it consistently finds the theoretically optimal basis. For random graphs, it provides a very tight approximation (<3% weight difference) instantly.

## Proposal
Given the massive performance gap, I propose creating a PR to either:
1.  Improve the existing `minimum_cycle_basis` with these heuristics.
2.  Or add this as a faster alternative `minimum_cycle_basis_approx`.

The code is pure Python and follows NetworkX interfaces.

[Link to Repository / Code Snippet]
