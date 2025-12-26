#!/usr/bin/env python3
"""
Experiment to check what physical state information is available in mendeleev library
"""

from mendeleev import element

# Test a few elements with known states
test_elements = [
    (1, 'H', 'gas'),      # Hydrogen - gas
    (6, 'C', 'solid'),    # Carbon - solid
    (80, 'Hg', 'liquid'), # Mercury - liquid
]

print("Checking available physical state attributes in mendeleev:\n")

for atomic_number, symbol, expected_state in test_elements:
    elem = element(atomic_number)
    print(f"\n{symbol} (Z={atomic_number}):")
    print(f"  Expected state: {expected_state}")

    # Check various attributes that might contain state info
    attrs_to_check = [
        'phase',
        'state',
        'physical_state',
        'melting_point',
        'boiling_point',
    ]

    for attr in attrs_to_check:
        if hasattr(elem, attr):
            value = getattr(elem, attr)
            print(f"  {attr}: {value}")

    # List all attributes
    print(f"  All attributes: {[a for a in dir(elem) if not a.startswith('_')][:20]}...")
