"""
Basic tests to ensure the package structure is working.
"""
import unittest
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestBasicStructure(unittest.TestCase):
    def test_imports(self):
        """Test that basic imports work"""
        try:
            import trainers
            import datasets
            import evaluation
            import utils
        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")

if __name__ == '__main__':
    unittest.main()
