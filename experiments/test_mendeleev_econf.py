#!/usr/bin/env python3
"""Experiment to understand mendeleev electron configuration format"""

from mendeleev import element

# Test elements mentioned in the issue
test_elements = [
    ('H', 1),
    ('He', 2),
    ('N', 7),
    ('Ne', 10),
    ('Ar', 18),
    ('Zn', 30),
    ('Kr', 36),
    ('Xe', 54),
    ('Rn', 86)
]

print("Testing electron configurations from mendeleev library:")
print("=" * 70)

for symbol, atomic_number in test_elements:
    elem = element(atomic_number)
    print(f"\n{symbol} (Z={atomic_number}):")
    print(f"  econf: {elem.econf}")
    print(f"  ec.conf: {elem.ec.conf if hasattr(elem, 'ec') else 'N/A'}")

    # Check for any other electron configuration attributes
    attrs = [attr for attr in dir(elem) if 'conf' in attr.lower() or 'elec' in attr.lower()]
    print(f"  Related attributes: {attrs}")
