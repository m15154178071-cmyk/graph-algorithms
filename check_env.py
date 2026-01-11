import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"CWD: {os.getcwd()}")
print("Path:")
for p in sys.path:
    print(f"  {p}")

print("-" * 20)

try:
    import igraph
    print(f"SUCCESS: igraph imported. Version: {igraph.__version__}")
    print(f"File: {igraph.__file__}")
except ImportError as e:
    print(f"FAILURE: Could not import igraph. Error: {e}")
except Exception as e:
    print(f"FAILURE: Unexpected error importing igraph. Error: {e}")

print("-" * 20)
try:
    import networkx
    print(f"SUCCESS: networkx imported. Version: {networkx.__version__}")
except ImportError:
    print("FAILURE: Could not import networkx")
