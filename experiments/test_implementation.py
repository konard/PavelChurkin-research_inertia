#!/usr/bin/env python3
"""
Test the implementation of physical state and ionization energy fixes
"""

import sys
sys.path.insert(0, '/tmp/gh-issue-solver-1766743555657')

from research_inertia import get_physical_state, calculate_ether_coefficient
from mendeleev import element

# Test physical state determination
print("="*60)
print("TESTING PHYSICAL STATE DETERMINATION")
print("="*60)

test_cases = [
    (1, 'H', 'gas'),
    (6, 'C', 'solid'),
    (35, 'Br', 'liquid'),
    (80, 'Hg', 'liquid'),
    (79, 'Au', 'solid'),
]

for z, sym, expected in test_cases:
    elem = element(z)
    state = get_physical_state(elem)
    status = "✓" if state == expected else "✗"
    print(f"{status} {sym}: Expected={expected}, Got={state}")

# Test ionization energy extraction
print("\n" + "="*60)
print("TESTING IONIZATION ENERGY EXTRACTION")
print("="*60)

for z in [1, 6, 26, 80]:
    elem = element(z)
    result = calculate_ether_coefficient(elem)
    if result:
        ie = result.get('ionization_energy')
        ps = result.get('physical_state')
        print(f"{elem.symbol}: IE={ie:.2f} eV, State={ps}")
    else:
        print(f"{elem.symbol}: No data")

print("\n" + "="*60)
print("TEST COMPLETED")
print("="*60)
