#!/usr/bin/env python3
print("Starting import test...")
try:
    from utils.calculate_performance_metrics import calculate_performance_metrics
    print("SUCCESS: Import worked!")
except ImportError as e:
    print(f"FAILED: {e}")
    import sys
    print(f"Python path: {sys.path}")
    print("Let's try manual import...")
    import utils
    print("utils imported successfully")
    from utils import calculate_performance_metrics as cpm_module
    print("calculate_performance_metrics module imported")
    print(dir(cpm_module))