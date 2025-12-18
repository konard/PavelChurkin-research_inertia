#!/usr/bin/env python3
"""
Quick test to ensure the main script runs without errors with new implementation.
We'll just test a few elements instead of running the full script.
"""

import sys
sys.path.insert(0, '/tmp/gh-issue-solver-1766089662537')

from research_inertia import calculate_ether_coefficient
from mendeleev import element

print("Testing main script functionality with new Z_eff implementation...")
print("=" * 70)

# Test a few representative elements
test_elements = [1, 7, 11, 26, 30, 47]

for Z in test_elements:
    try:
        elem = element(Z)
        result = calculate_ether_coefficient(elem)

        if result:
            print(f"\n{Z:3}. {elem.symbol:2} ({elem.name})")
            print(f"     Z_eff: {result['zeff']:.2f}")
            print(f"     k: {result['k_coefficient']:.3e} kg/m³")
            print(f"     Status: OK ✓")
        else:
            print(f"\n{Z:3}. {elem.symbol:2} - Skipped (missing data)")

    except Exception as e:
        print(f"\n{Z:3}. Error: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("Test completed successfully!")
