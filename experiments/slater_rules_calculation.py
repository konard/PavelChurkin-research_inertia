#!/usr/bin/env python3
"""
Experiment to understand and verify Slater's rules calculation for effective nuclear charge.

Slater's Rules:
1. For ns and np orbitals:
   - Same group electrons: each contributes 0.35 (except 1s: 0.30)
   - (n-1) shell electrons: each contributes 0.85
   - (n-2) and deeper shells: each contributes 1.00

2. For nd and nf orbitals:
   - Same group electrons: each contributes 0.35
   - All electrons to the left (lower energy): each contributes 1.00
"""

from mendeleev import element

def calculate_slater_zeff(elem):
    """
    Calculate effective nuclear charge using Slater's rules.

    Args:
        elem: Element from mendeleev library

    Returns:
        float: Effective nuclear charge Z_eff
    """
    Z = elem.atomic_number

    # Get electron configuration as OrderedDict {(n, orbital): count}
    ec_conf = elem.ec.conf

    # Convert to list of (n, orbital, count) tuples for easier processing
    config = [(n, orb, count) for (n, orb), count in ec_conf.items()]

    # Find the outermost electron (valence electron)
    # The last entry in the configuration
    if not config:
        return Z

    n_outer, orb_outer, count_outer = config[-1]

    # Calculate shielding constant S
    S = 0.0

    # Group orbitals by Slater groups
    # For s,p orbitals: same (n,s,p) group, then (n-1), then (n-2) and deeper
    # For d,f orbitals: same (n,d,f) group, then all to the left

    if orb_outer in ['s', 'p']:
        # Rules for ns, np orbitals
        for n, orb, count in config:
            if (n, orb) == (n_outer, orb_outer):
                # Same orbital - don't count the electron being shielded
                if n == 1 and orb == 's':
                    # Special case for 1s
                    S += (count - 1) * 0.30
                else:
                    S += (count - 1) * 0.35
            elif n == n_outer and orb in ['s', 'p']:
                # Same shell, different orbital in s,p group
                if n == 1:
                    S += count * 0.30
                else:
                    S += count * 0.35
            elif n == n_outer - 1:
                # One shell below (n-1)
                S += count * 0.85
            elif n < n_outer - 1:
                # Two or more shells below (n-2 and deeper)
                S += count * 1.00

    elif orb_outer in ['d', 'f']:
        # Rules for nd, nf orbitals
        for n, orb, count in config:
            if (n, orb) == (n_outer, orb_outer):
                # Same orbital - don't count the electron being shielded
                S += (count - 1) * 0.35
            elif n == n_outer and orb == orb_outer:
                # Same group
                S += count * 0.35
            else:
                # All other electrons to the left
                S += count * 1.00

    Z_eff = Z - S
    return Z_eff


# Test with examples from the issue
print("=" * 70)
print("Testing Slater's Rules Implementation")
print("=" * 70)

# Test case 1: Nitrogen (Z=7)
# Configuration: 1s2 2s2 2p3
# Expected: S = (4 * 0.35) + (2 * 0.85) = 1.4 + 1.7 = 3.1
# Expected: Z_eff = 7 - 3.1 = 3.9
print("\nTest Case 1: Nitrogen (N)")
N = element(7)
print(f"  Atomic number: {N.atomic_number}")
print(f"  Configuration: {N.econf}")
print(f"  ec.conf: {N.ec.conf}")

# Manual calculation for verification
print("\n  Manual calculation:")
print("    Outermost electron: 2p")
print("    Same group (2s2 2p3): 4 electrons excluding one being shielded")
print("    S from same group: 4 * 0.35 = 1.40")
print("    Shell n-1 (1s2): 2 electrons")
print("    S from n-1: 2 * 0.85 = 1.70")
print("    Total S = 1.40 + 1.70 = 3.10")
print("    Z_eff = 7 - 3.10 = 3.90")

calc_zeff_n = calculate_slater_zeff(N)
print(f"\n  Calculated Z_eff: {calc_zeff_n:.2f}")
print(f"  Expected Z_eff: 3.90")
print(f"  Match: {'✓' if abs(calc_zeff_n - 3.9) < 0.01 else '✗'}")

# Test case 2: Zinc (Z=30)
# Configuration: [Ar] 3d10 4s2
# Expected: S = (1 * 0.35) + (18 * 0.85) + (10 * 1.00) = 0.35 + 15.3 + 10 = 25.65
# Expected: Z_eff = 30 - 25.65 = 4.35
print("\n" + "=" * 70)
print("\nTest Case 2: Zinc (Zn)")
Zn = element(30)
print(f"  Atomic number: {Zn.atomic_number}")
print(f"  Configuration: {Zn.econf}")
print(f"  ec.conf: {Zn.ec.conf}")

print("\n  Manual calculation:")
print("    Outermost electron: 4s")
print("    Same group (4s2): 1 electron excluding one being shielded")
print("    S from same group: 1 * 0.35 = 0.35")
print("    Shell n-1 (3s2 3p6 3d10): 18 electrons")
print("    S from n-1: 18 * 0.85 = 15.30")
print("    Shell n-2 (2s2 2p6): 8 electrons")
print("    S from n-2: 8 * 1.00 = 8.00")
print("    Shell n-3 (1s2): 2 electrons")
print("    S from n-3: 2 * 1.00 = 2.00")
print("    Total S = 0.35 + 15.30 + 8.00 + 2.00 = 25.65")
print("    Z_eff = 30 - 25.65 = 4.35")

calc_zeff_zn = calculate_slater_zeff(Zn)
print(f"\n  Calculated Z_eff: {calc_zeff_zn:.2f}")
print(f"  Expected Z_eff: 4.35")
print(f"  Match: {'✓' if abs(calc_zeff_zn - 4.35) < 0.01 else '✗'}")

# Wait, there's a discrepancy in the issue description!
# Let me recalculate Zn according to what the issue says
print("\n" + "=" * 70)
print("\nRe-checking Zinc calculation from issue:")
print("  Issue states: S = (1 * 0.35) + (18 * 0.85) + (10 * 1.00) = 25.65")
print("  This seems to count:")
print("    - 1 electron from same group (4s²): 0.35")
print("    - 18 electrons from somewhere: 15.30")
print("    - 10 electrons from somewhere: 10.00")
print("\n  But Zn configuration is: 1s² 2s² 2p⁶ 3s² 3p⁶ 3d¹⁰ 4s²")
print("  Total electrons: 2+2+6+2+6+10+2 = 30 ✓")
print("\n  If we're calculating for the outermost 4s electron:")
print("    - Same shell (4s): 1 other electron → 1 × 0.35 = 0.35")
print("    - n-1 shell (3s² 3p⁶ 3d¹⁰): 18 electrons → 18 × 0.85 = 15.30")
print("    - n-2 shell (2s² 2p⁶): 8 electrons → 8 × 1.00 = 8.00")
print("    - n-3 shell (1s²): 2 electrons → 2 × 1.00 = 2.00")
print("    - Total S = 0.35 + 15.30 + 8.00 + 2.00 = 25.65 ✓")
print("\n  The issue description seems to have combined (n-2) and (n-3) into '10'")
print("  which gives 8+2=10, but they should all use coefficient 1.00")
