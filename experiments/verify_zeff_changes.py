#!/usr/bin/env python3
"""
Verify that the new Slater's rules implementation produces different results
from the old simplified implementation.
"""

from mendeleev import element

# Old implementation (simplified)
def get_effective_nuclear_charge_old(elem):
    """Оцениваем эффективный заряд ядра Z_eff (старая версия)"""
    Z = elem.atomic_number

    # Упрощенная оценка по правилам Слейтера
    if Z <= 2:  # H, He
        return Z - 0.30
    elif Z <= 10:  # Li до Ne
        return Z - 4.15
    elif Z <= 18:  # Na до Ar
        return Z - 8.85
    elif Z <= 36:  # K до Kr
        return Z - 17.15
    elif Z <= 54:  # Rb до Xe
        return Z - 26.25
    else:
        return Z * 0.85  # Для тяжелых элементов


# Import new implementation
import sys
sys.path.insert(0, '/tmp/gh-issue-solver-1766089662537')
from research_inertia import get_effective_nuclear_charge

print("=" * 80)
print("Comparison of Old vs New Z_eff Implementation")
print("=" * 80)

# Test a representative set of elements
test_elements = [
    (1, 'H'),
    (2, 'He'),
    (7, 'N'),
    (11, 'Na'),
    (17, 'Cl'),
    (26, 'Fe'),
    (30, 'Zn'),
    (36, 'Kr'),
    (47, 'Ag'),
    (79, 'Au'),
]

print(f"\n{'Element':<10} {'Z':<5} {'Old Z_eff':<12} {'New Z_eff':<12} {'Difference':<12} {'Change':<10}")
print("-" * 80)

for Z, symbol in test_elements:
    elem = element(Z)
    old_zeff = get_effective_nuclear_charge_old(elem)
    new_zeff = get_effective_nuclear_charge(elem)
    diff = new_zeff - old_zeff
    change_pct = (diff / old_zeff * 100) if old_zeff != 0 else 0

    print(f"{symbol:<10} {Z:<5} {old_zeff:<12.2f} {new_zeff:<12.2f} {diff:<12.2f} {change_pct:>6.1f}%")

print("\n" + "=" * 80)
print("The new implementation provides element-specific calculations")
print("based on actual electron configurations, rather than fixed ranges.")
print("=" * 80)
