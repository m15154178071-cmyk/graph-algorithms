import sys
try:
    import igraph
    print("igraph imported successfully")
except Exception as e:
    print(f"igraph failed: {e}")

try:
    import rustworkx
    print(f"rustworkx version: {rustworkx.__version__}")
    if hasattr(rustworkx, 'minimum_cycle_basis'):
        print("rustworkx has minimum_cycle_basis")
    else:
        print("rustworkx does NOT have minimum_cycle_basis")
    if hasattr(rustworkx, 'cycle_basis'):
        print("rustworkx has cycle_basis")
except Exception as e:
    print(f"rustworkx failed: {e}")
