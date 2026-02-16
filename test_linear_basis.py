"""
Unit tests for graph algorithms - Example test suite
运行测试: python -m unittest test_linear_basis.py -v
Run tests: python -m unittest test_linear_basis.py -v
"""

import unittest
import sys
import os

# Add parent directory to path to import solution_basic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution_basic import LinearBasis


class TestLinearBasis(unittest.TestCase):
    """Test cases for LinearBasis class - 线性基类的测试用例"""
    
    def test_insert_independent_cycles(self):
        """Test inserting linearly independent cycles - 测试插入线性无关的环"""
        lb = LinearBasis(num_edges=10)
        
        # Two independent cycles - 两个独立的环
        cycle1 = {0, 1, 2}
        cycle2 = {3, 4, 5}
        
        self.assertTrue(lb.insert(cycle1), "First cycle should be independent")
        self.assertTrue(lb.insert(cycle2), "Second cycle should be independent")
        self.assertEqual(lb.basis_count, 2, "Should have 2 basis elements")
    
    def test_insert_dependent_cycle(self):
        """Test inserting a dependent cycle - 测试插入线性相关的环"""
        lb = LinearBasis(num_edges=10)
        
        # Same cycle inserted twice - 相同的环插入两次
        cycle1 = {0, 1, 2}
        cycle2 = {0, 1, 2}
        
        self.assertTrue(lb.insert(cycle1))
        self.assertFalse(lb.insert(cycle2), "Duplicate cycle should be dependent")
        self.assertEqual(lb.basis_count, 1)
    
    def test_insert_xor_dependent(self):
        """Test inserting cycles that are XOR-dependent - 测试异或相关的环"""
        lb = LinearBasis(num_edges=10)
        
        # Three cycles where C3 = C1 XOR C2
        # 三个环，其中 C3 = C1 异或 C2
        cycle1 = {0, 1, 2}
        cycle2 = {1, 2, 3}
        cycle3 = {0, 3}  # XOR of cycle1 and cycle2
        
        self.assertTrue(lb.insert(cycle1))
        self.assertTrue(lb.insert(cycle2))
        self.assertFalse(lb.insert(cycle3), "XOR-dependent cycle should fail")
        self.assertEqual(lb.basis_count, 2)
    
    def test_empty_cycle(self):
        """Test inserting an empty cycle - 测试插入空环"""
        lb = LinearBasis(num_edges=10)
        empty_cycle = set()
        
        self.assertFalse(lb.insert(empty_cycle), "Empty cycle should be dependent")
        self.assertEqual(lb.basis_count, 0)
    
    def test_single_edge_cycle(self):
        """Test single edge (which shouldn't form a cycle) - 测试单边"""
        lb = LinearBasis(num_edges=10)
        single_edge = {5}
        
        # A single edge can be inserted as it's linearly independent
        # 单边可以被插入，因为它是线性无关的
        self.assertTrue(lb.insert(single_edge))
        self.assertEqual(lb.basis_count, 1)
    
    def test_multiple_independent_cycles(self):
        """Test multiple independent cycles - 测试多个独立环"""
        lb = LinearBasis(num_edges=15)
        
        cycles = [
            {0, 1, 2},
            {3, 4, 5},
            {6, 7, 8},
            {9, 10, 11}
        ]
        
        for i, cycle in enumerate(cycles):
            result = lb.insert(cycle)
            self.assertTrue(result, f"Cycle {i} should be independent")
        
        self.assertEqual(lb.basis_count, 4, "Should have 4 independent cycles")
    
    def test_basis_property(self):
        """Test that basis maintains linear independence - 测试基保持线性无关"""
        lb = LinearBasis(num_edges=20)
        
        # Insert several cycles
        # XOR of {0,1,2} and {1,2,3} = {0,3} (edges that appear in one but not both)
        cycles = [
            {0, 1, 2},
            {1, 2, 3},
            {4, 5, 6},
            {0, 3},  # This is XOR of first two, should be rejected
        ]
        
        results = [lb.insert(c) for c in cycles]
        
        # First three should be independent, fourth should be dependent
        self.assertEqual(results, [True, True, True, False])
        self.assertEqual(lb.basis_count, 3)


class TestLinearBasisEdgeCases(unittest.TestCase):
    """Edge case tests for LinearBasis - 边界情况测试"""
    
    def test_large_cycle(self):
        """Test with a large cycle - 测试大环"""
        lb = LinearBasis(num_edges=1000)
        large_cycle = set(range(100))
        
        self.assertTrue(lb.insert(large_cycle))
        self.assertEqual(lb.basis_count, 1)
    
    def test_sequential_inserts(self):
        """Test sequential insertions - 测试顺序插入"""
        lb = LinearBasis(num_edges=20)
        
        # Build a sequence of cycles
        for i in range(5):
            cycle = {i*2, i*2+1, i*2+2}
            lb.insert(cycle)
        
        # All should be independent since they don't share edges
        self.assertEqual(lb.basis_count, 5)
    
    def test_pivot_selection(self):
        """Test that pivot selection works correctly - 测试主元选择正确"""
        lb = LinearBasis(num_edges=10)
        
        cycle1 = {0, 1, 5}  # Max pivot = 5
        cycle2 = {2, 3, 5}  # Max pivot = 5, will collide
        
        self.assertTrue(lb.insert(cycle1))
        # After inserting cycle1, pivot 5 is taken
        # cycle2 will XOR with basis[5], resulting in {0, 1, 2, 3}
        # which should be independent
        self.assertTrue(lb.insert(cycle2))
        self.assertEqual(lb.basis_count, 2)


if __name__ == '__main__':
    # Run tests with verbose output
    # 以详细模式运行测试
    unittest.main(verbosity=2)
