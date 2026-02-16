# Code Improvement Examples (代码改进示例)

This document provides concrete, ready-to-use code examples for improving your graph algorithms project.

本文档提供了具体的、可直接使用的代码示例，用于改进您的图算法项目。

---

## 1. Better Error Handling (更好的错误处理)

### Current Code (当前代码)
```python
# solution_basic.py - DataHelper.read_edges
try:
    u, v = int(parts[0]), int(parts[1])
    if u != v:
        edges.append((min(u, v), max(u, v)))
except:
    pass
```

### Improved Code (改进后的代码)
```python
import logging

logger = logging.getLogger(__name__)

class DataHelper:
    @staticmethod
    def read_edges(filepath):
        """
        Read edges from a file, handling errors gracefully.
        
        Args:
            filepath: Path to the input file
            
        Returns:
            List of edges as tuples (u, v) where u < v
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        import os
        
        # Validate file path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Input file not found: {filepath}")
        if not os.path.isfile(filepath):
            raise ValueError(f"Path is not a file: {filepath}")
        
        edges = []
        line_num = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line_num += 1
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                
                parts = re.split(r'[,\s]+', line)
                if len(parts) < 2:
                    logger.warning(f"Line {line_num}: Invalid format, expected at least 2 values")
                    continue
                
                try:
                    u, v = int(parts[0]), int(parts[1])
                    if u == v:
                        logger.warning(f"Line {line_num}: Self-loop ({u}, {v}) ignored")
                        continue
                    edges.append((min(u, v), max(u, v)))
                except (ValueError, IndexError) as e:
                    logger.warning(f"Line {line_num}: Failed to parse '{line}': {e}")
                    continue
        
        logger.info(f"Successfully read {len(edges)} edges from {filepath}")
        return edges
```

---

## 2. Add Type Hints (添加类型提示)

### Current Code (当前代码)
```python
def bfs_with_detours(u, v, adj, edge_to_id, visited_limit=200, detours=5, max_path_len=None):
    """
    Returns a list of candidate cycles (each is a set of edge IDs).
    """
    candidates = []
    # ...
```

### Improved Code (改进后的代码)
```python
from typing import Dict, List, Set, Tuple, Optional

def bfs_with_detours(
    u: int,
    v: int,
    adj: Dict[int, List[int]],
    edge_to_id: Dict[Tuple[int, int], int],
    visited_limit: int = 200,
    detours: int = 5,
    max_path_len: Optional[int] = None
) -> List[Set[int]]:
    """
    Find candidate cycles using BFS with alternative path exploration.
    
    Args:
        u: Starting node
        v: Ending node
        adj: Adjacency list mapping nodes to neighbors
        edge_to_id: Mapping from edge tuple (u,v) to edge ID
        visited_limit: Maximum nodes to visit in BFS
        detours: Number of alternative paths to explore
        max_path_len: Maximum path length to consider (None = unlimited)
    
    Returns:
        List of candidate cycles, each represented as a set of edge IDs
        
    Time Complexity: O(V + E * detours) where V is nodes, E is edges
    """
    candidates: List[Set[int]] = []
    # ... rest of implementation
```

---

## 3. Use Constants Instead of Magic Numbers (使用常量代替魔法数字)

### Current Code (当前代码)
```python
sys.setrecursionlimit(20000)

def bfs_with_detours(u, v, adj, edge_to_id, visited_limit=200, detours=5, max_path_len=None):
    # ...
```

### Improved Code (改进后的代码)
```python
# At the top of the file (文件顶部)
# =====================================================
# Configuration Constants (配置常量)
# =====================================================

# Recursion settings
MAX_RECURSION_DEPTH = 20000  # Maximum recursion depth for deep graph traversal

# BFS settings
DEFAULT_VISITED_LIMIT = 200   # Default max nodes to visit in BFS
DEFAULT_DETOURS = 5           # Default number of alternative paths
DEFAULT_MAX_PATH_LEN = None   # Default max path length (None = unlimited)

# Cycle detection settings
TRIANGLE_SIZE = 3             # Size of triangle cycles (C3)
SQUARE_SIZE = 4              # Size of square cycles (C4)

# =====================================================

sys.setrecursionlimit(MAX_RECURSION_DEPTH)

def bfs_with_detours(
    u: int,
    v: int,
    adj: Dict[int, List[int]],
    edge_to_id: Dict[Tuple[int, int], int],
    visited_limit: int = DEFAULT_VISITED_LIMIT,
    detours: int = DEFAULT_DETOURS,
    max_path_len: Optional[int] = DEFAULT_MAX_PATH_LEN
) -> List[Set[int]]:
    """..."""
    # ...
```

---

## 4. Complete Documentation Strings (完善文档字符串)

### Current Code (当前代码)
```python
def find_squares(adj, edge_to_id):
    """
    Finds all simple squares (4-cycles). 
    """
    # ...
```

### Improved Code (改进后的代码)
```python
def find_squares(
    adj: Dict[int, List[int]],
    edge_to_id: Dict[Tuple[int, int], int]
) -> Tuple[List[FrozenSet[int]], Set[int]]:
    """
    Find all simple squares (4-cycles) in an undirected graph.
    
    A square is a simple cycle of length 4. This function uses an efficient
    neighbor intersection approach to enumerate all squares without duplication.
    
    Algorithm:
        For each pair of nodes (u,v), find common neighbors that form squares.
        Uses ordering constraints (u < v < w < x) to avoid counting duplicates.
    
    Args:
        adj: Adjacency list mapping each node to its list of neighbors.
             Example: {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
        edge_to_id: Mapping from edge tuple (u, v) where u < v to unique edge ID.
                   Example: {(0,1): 0, (0,2): 1, (1,3): 2, (2,3): 3}
    
    Returns:
        A tuple containing:
        - squares: List of frozensets, each containing 4 edge IDs forming a square
        - covered_edges: Set of edge IDs that participate in at least one square
    
    Example:
        Given a graph with nodes [0,1,2,3] forming a square:
        >>> adj = {0: [1,3], 1: [0,2], 2: [1,3], 3: [0,2]}
        >>> edge_to_id = {(0,1): 0, (1,2): 1, (2,3): 2, (0,3): 3}
        >>> squares, covered = find_squares(adj, edge_to_id)
        >>> len(squares)
        1
        >>> squares[0]
        frozenset({0, 1, 2, 3})
    
    Time Complexity: O(V * deg^2) where V is number of nodes, deg is average degree
    Space Complexity: O(V + E) for adjacency sets
    
    Note:
        - Self-loops are not considered
        - Only simple (chordless) squares are detected
        - Uses canonical ordering to prevent duplicate detection
    """
    # ... implementation
```

---

## 5. Use Logging Instead of Print (使用日志代替打印)

### Current Code (当前代码)
```python
def main():
    # ...
    print(f"Nodes: {num_nodes}, Edges: {num_edges}, Est. Target Rank: {target_rank}")
    print("Extracting Triangles...")
    print(f"Found {len(triangles)} triangles. Covered {len(covered_triangle_edges)} edges.")
```

### Improved Code (改进后的代码)
```python
import logging

# Configure logging at the start of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for MCB extraction."""
    # ...
    
    logger.info(f"Graph loaded: {num_nodes} nodes, {num_edges} edges, "
                f"estimated target rank: {target_rank}")
    
    logger.info("Extracting triangles (C3 cycles)...")
    triangles, covered_triangle_edges = find_triangles(adj, edge_to_id)
    logger.info(f"Found {len(triangles)} triangles covering "
                f"{len(covered_triangle_edges)} edges")
    
    logger.info("Extracting squares (C4 cycles)...")
    squares, covered_square_edges = find_squares(adj, edge_to_id)
    logger.info(f"Found {len(squares)} squares covering "
                f"{len(covered_square_edges)} edges")
    
    # For debugging, use logger.debug()
    logger.debug(f"Covered edges union size: {len(covered_edges)}")
    
    # For warnings
    if len(all_candidates) < target_rank:
        logger.warning(f"Found only {len(all_candidates)} candidates, "
                      f"expected {target_rank}")

# Usage: Control log level from command line
# python solution_basic.py input.txt output.txt --log-level DEBUG
```

---

## 6. Add Unit Tests (添加单元测试)

Create a new file: `test_graph_algorithms.py`

```python
"""
Unit tests for graph algorithms.
Run with: python -m pytest test_graph_algorithms.py
or: python -m unittest test_graph_algorithms.py
"""

import unittest
from solution_basic import LinearBasis, find_triangles, find_squares


class TestLinearBasis(unittest.TestCase):
    """Test cases for LinearBasis class."""
    
    def test_insert_independent_cycles(self):
        """Test inserting linearly independent cycles."""
        lb = LinearBasis(num_edges=10)
        
        # Two independent cycles
        cycle1 = {0, 1, 2}
        cycle2 = {3, 4, 5}
        
        self.assertTrue(lb.insert(cycle1), "First cycle should be independent")
        self.assertTrue(lb.insert(cycle2), "Second cycle should be independent")
        self.assertEqual(lb.basis_count, 2, "Should have 2 basis elements")
    
    def test_insert_dependent_cycle(self):
        """Test inserting a dependent cycle."""
        lb = LinearBasis(num_edges=10)
        
        # Same cycle inserted twice
        cycle1 = {0, 1, 2}
        cycle2 = {0, 1, 2}
        
        self.assertTrue(lb.insert(cycle1))
        self.assertFalse(lb.insert(cycle2), "Duplicate cycle should be dependent")
        self.assertEqual(lb.basis_count, 1)
    
    def test_insert_xor_dependent(self):
        """Test inserting cycles that are XOR-dependent."""
        lb = LinearBasis(num_edges=10)
        
        # Three cycles where C3 = C1 XOR C2
        cycle1 = {0, 1, 2}
        cycle2 = {1, 2, 3}
        cycle3 = {0, 3}  # XOR of cycle1 and cycle2
        
        self.assertTrue(lb.insert(cycle1))
        self.assertTrue(lb.insert(cycle2))
        self.assertFalse(lb.insert(cycle3), "XOR-dependent cycle should fail")
        self.assertEqual(lb.basis_count, 2)
    
    def test_empty_cycle(self):
        """Test inserting an empty cycle."""
        lb = LinearBasis(num_edges=10)
        empty_cycle = set()
        
        self.assertFalse(lb.insert(empty_cycle), "Empty cycle should be dependent")
        self.assertEqual(lb.basis_count, 0)


class TestTriangleFinding(unittest.TestCase):
    """Test cases for triangle detection."""
    
    def test_simple_triangle(self):
        """Test finding a single triangle."""
        # Graph: 0-1-2-0 (a triangle)
        adj = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1]
        }
        edge_to_id = {
            (0, 1): 0,
            (0, 2): 1,
            (1, 2): 2
        }
        
        triangles, covered = find_triangles(adj, edge_to_id)
        
        self.assertEqual(len(triangles), 1, "Should find exactly 1 triangle")
        self.assertEqual(triangles[0], frozenset({0, 1, 2}))
        self.assertEqual(covered, {0, 1, 2})
    
    def test_no_triangles(self):
        """Test graph with no triangles."""
        # Graph: 0-1-2 (a path, no triangle)
        adj = {
            0: [1],
            1: [0, 2],
            2: [1]
        }
        edge_to_id = {
            (0, 1): 0,
            (1, 2): 1
        }
        
        triangles, covered = find_triangles(adj, edge_to_id)
        
        self.assertEqual(len(triangles), 0, "Should find no triangles")
        self.assertEqual(len(covered), 0)
    
    def test_multiple_triangles(self):
        """Test finding multiple triangles."""
        # Graph: Two triangles sharing an edge
        #   0---1
        #   |\ /|
        #   | 2 |
        #   |/ \|
        #   3---4
        adj = {
            0: [1, 2, 3],
            1: [0, 2, 4],
            2: [0, 1, 3, 4],
            3: [0, 2, 4],
            4: [1, 2, 3]
        }
        edge_to_id = {
            (0, 1): 0, (0, 2): 1, (0, 3): 2,
            (1, 2): 3, (1, 4): 4,
            (2, 3): 5, (2, 4): 6,
            (3, 4): 7
        }
        
        triangles, covered = find_triangles(adj, edge_to_id)
        
        # Should find: {0,1,2}, {0,2,3}, {1,2,4}, {2,3,4}
        self.assertGreater(len(triangles), 0, "Should find triangles")
        
        # Verify all triangles are valid (size 3)
        for tri in triangles:
            self.assertEqual(len(tri), 3, "Each triangle should have 3 edges")


class TestSquareFinding(unittest.TestCase):
    """Test cases for square detection."""
    
    def test_simple_square(self):
        """Test finding a single square."""
        # Graph: 0-1-2-3-0 (a square)
        adj = {
            0: [1, 3],
            1: [0, 2],
            2: [1, 3],
            3: [0, 2]
        }
        edge_to_id = {
            (0, 1): 0,
            (1, 2): 1,
            (2, 3): 2,
            (0, 3): 3
        }
        
        squares, covered = find_squares(adj, edge_to_id)
        
        self.assertEqual(len(squares), 1, "Should find exactly 1 square")
        self.assertEqual(squares[0], frozenset({0, 1, 2, 3}))
        self.assertEqual(covered, {0, 1, 2, 3})
    
    def test_no_squares(self):
        """Test graph with no squares."""
        # Graph: A triangle (no square)
        adj = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1]
        }
        edge_to_id = {
            (0, 1): 0,
            (0, 2): 1,
            (1, 2): 2
        }
        
        squares, covered = find_squares(adj, edge_to_id)
        
        self.assertEqual(len(squares), 0, "Should find no squares")
        self.assertEqual(len(covered), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
```

### Running Tests (运行测试)

```bash
# Using unittest
python -m unittest test_graph_algorithms.py -v

# Or using pytest (install with: pip install pytest)
pytest test_graph_algorithms.py -v

# Run specific test
python -m unittest test_graph_algorithms.TestLinearBasis.test_insert_independent_cycles
```

---

## 7. Improved Command-Line Interface (改进的命令行界面)

### Current Code (当前代码)
```python
def main():
    if len(sys.argv) < 3:
        print("Usage: python solution.py input.txt output.txt")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
```

### Improved Code (改进后的代码)
```python
import argparse
import logging

def setup_logging(log_level: str) -> None:
    """Configure logging based on specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Main entry point for graph analysis."""
    parser = argparse.ArgumentParser(
        description='Find Minimum Cycle Basis (MCB) in an undirected graph.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python solution_basic.py input.txt output.txt
  python solution_basic.py input.txt output.txt --verbose
  python solution_basic.py input.txt output.txt --log-level DEBUG
  python solution_basic.py input.txt output.txt --visited-limit 300

For more information, see README.md
        '''
    )
    
    # Required arguments
    parser.add_argument(
        'input',
        help='Input file containing edge list (format: "u v" per line)'
    )
    parser.add_argument(
        'output',
        help='Output file for cycle basis results'
    )
    
    # Optional arguments
    parser.add_argument(
        '--visited-limit',
        type=int,
        default=DEFAULT_VISITED_LIMIT,
        help=f'Maximum nodes to visit in BFS (default: {DEFAULT_VISITED_LIMIT})'
    )
    parser.add_argument(
        '--detours',
        type=int,
        default=DEFAULT_DETOURS,
        help=f'Number of alternative paths to explore (default: {DEFAULT_DETOURS})'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output (equivalent to --log-level DEBUG)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else args.log_level
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting graph analysis: {args.input} -> {args.output}")
    logger.debug(f"Parameters: visited_limit={args.visited_limit}, detours={args.detours}")
    
    # Proceed with the actual work
    try:
        input_file = args.input
        output_file = args.output
        
        # ... rest of main logic
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1
    
    logger.info("Analysis completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

---

## 8. Input Validation Helper (输入验证辅助函数)

```python
import os
from pathlib import Path
from typing import Optional

def validate_input_file(filepath: str, max_size_mb: Optional[int] = None) -> None:
    """
    Validate input file exists and is readable.
    
    Args:
        filepath: Path to the input file
        max_size_mb: Maximum file size in MB (None = no limit)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid (directory, too large, etc.)
        PermissionError: If file is not readable
    """
    path = Path(filepath)
    
    # Check existence
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    # Check if it's a file (not directory)
    if not path.is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    
    # Check readability
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"File is not readable: {filepath}")
    
    # Check file size
    if max_size_mb is not None:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.2f} MB (max: {max_size_mb} MB)"
            )

def validate_output_path(filepath: str, allow_overwrite: bool = False) -> None:
    """
    Validate output file path.
    
    Args:
        filepath: Path to the output file
        allow_overwrite: Whether to allow overwriting existing files
        
    Raises:
        FileExistsError: If file exists and overwrite not allowed
        PermissionError: If directory is not writable
    """
    path = Path(filepath)
    
    # Check if file already exists
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Output file already exists: {filepath}\n"
            f"Use --force to overwrite"
        )
    
    # Check if parent directory exists and is writable
    parent = path.parent
    if not parent.exists():
        raise FileNotFoundError(f"Output directory doesn't exist: {parent}")
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"Output directory not writable: {parent}")

# Usage in main():
def main():
    # ... argparse setup ...
    
    args = parser.parse_args()
    
    # Validate inputs
    try:
        validate_input_file(args.input, max_size_mb=100)
        validate_output_path(args.output, allow_overwrite=args.force)
    except (FileNotFoundError, ValueError, PermissionError) as e:
        logger.error(str(e))
        return 1
    
    # ... continue with processing
```

---

## Summary (总结)

These improvements will make your code:

这些改进将使您的代码：

1. ✅ **More Robust** (更健壮) - Better error handling
2. ✅ **More Maintainable** (更易维护) - Type hints and documentation
3. ✅ **More Testable** (更易测试) - Unit tests
4. ✅ **More Professional** (更专业) - Logging, CLI, validation
5. ✅ **More Readable** (更易读) - Constants, clear documentation

**Priority Order (优先级顺序):**
1. Add error handling (添加错误处理)
2. Add type hints (添加类型提示)
3. Write unit tests (编写单元测试)
4. Improve documentation (改进文档)
5. Add logging (添加日志)
6. Improve CLI (改进命令行界面)

You can implement these improvements gradually, one at a time!

您可以逐步实施这些改进，一次一个！
