#!/usr/bin/env python3
"""
Check ionization energy attributes in mendeleev
"""

from mendeleev import element

# Test a few elements
test_elements = [1, 6, 26, 80]

print("Checking ionization energy attributes:\n")

for z in test_elements:
    elem = element(z)
    print(f"\n{elem.symbol} (Z={z}):")

    # Check various ionization-related attributes
    attrs = [a for a in dir(elem) if 'ion' in a.lower() or 'en_' in a.lower()]
    for attr in attrs:
        if not attr.startswith('_'):
            try:
                value = getattr(elem, attr)
                if value is not None:
                    print(f"  {attr}: {value}")
            except:
                pass
