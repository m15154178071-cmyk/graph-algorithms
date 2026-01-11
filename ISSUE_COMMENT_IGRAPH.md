# Performance Update: Pure Python Implementation Outperforms C Extension (igraph) on Grid Graphs

Following the previous improvements, I benchmarked the algorithm against **igraph** (a highly optimized C library). The results exceeded expectations.

On structured graphs (specifically Grid graphs), my pure Python heuristic implementation is actually **significantly faster than the C-based igraph**, thanks to the optimized short-cycle enumeration strategy which avoids the overhead of general-purpose MCB algorithms.

## Benchmark: Python Heuristic vs C Extension (igraph)

| Graph Type | My Algo (Python) | igraph (C Library) | Comparison | Correctness |
| :--- | :--- | :--- | :--- | :--- |
| **Grid 30x30** (900 Nodes) | **0.11 s** | 1.57 s | **~14x Faster than C** | ✅ Exact Match |
| **Wheel** (n=1000) | 2.10 s | **0.49 s** | ~4x Slower (Same order of magnitude) | ✅ Exact Match |
| **Random** (n=200, p=0.1) | 3.81 s | **0.38 s** | ~10x Slower | Approx (0.7% diff) |

*Environment: Windows / Python 3.13*

## Why is it faster than C?
Grid graphs are dominated by C4 cycles (squares). The heuristic phase of my algorithm (`enumerate_cycles`) identifies these locally in linear time relative to the number of squares, effectively constructing the basis without needing to perform the more expensive, generic search steps that the standard Paton/Horton algorithms (used by igraph) might process.

This demonstrates that for Planar-like and regular graphs, **a smart Python algorithm can beat a brute-force C algorithm.**

This further strengthens the case for including this `minimum_cycle_basis_heuristic` as a standard utility in NetworkX for users dealing with lattice-like or sparse datasets.
