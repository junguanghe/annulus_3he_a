#!/usr/bin/env python3
"""
Test script to verify the Python conversion works correctly.
This runs a minimal test to ensure the code can execute.
"""

import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from propagator import propagator

def test_propagator():
    """Test that propagator function works"""
    print("Testing propagator function...")
    
    # Load gap.txt
    if not os.path.exists('gap.txt'):
        print("Error: gap.txt not found!")
        return False
    
    data = np.loadtxt('gap.txt')
    xspan = data[:, 0]
    Delta1 = data[:, 1]
    Delta2 = data[:, 2]
    print(f"Loaded gap.txt: xspan shape = {xspan.shape}")
    
    # Test propagator with a simple case
    thetap = 0.1
    epsilon = 1j * 0.1
    
    try:
        x, f1, f2, g = propagator(thetap, epsilon, xspan, Delta1, Delta2)
        print(f"✓ Propagator test passed!")
        print(f"  x shape: {x.shape}, f1 shape: {f1.shape}, f2 shape: {f2.shape}, g shape: {g.shape}")
        return True
    except Exception as e:
        print(f"✗ Propagator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("Testing Python conversion of MATLAB code")
    print("=" * 50)
    
    success = test_propagator()
    
    if success:
        print("\n✓ Basic functionality test passed!")
        print("You can now run main.py to perform the full calculation.")
        print("Example: python main.py")
    else:
        print("\n✗ Tests failed. Please check the errors above.")
        sys.exit(1)

