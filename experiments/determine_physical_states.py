#!/usr/bin/env python3
"""
Determine physical states of elements at normal conditions (25°C = 298.15K, 1 atm)
"""

from mendeleev import element

NORMAL_TEMP_K = 298.15  # 25°C in Kelvin


def get_physical_state(elem):
    """
    Determine physical state at normal conditions based on melting and boiling points.

    Returns:
        str: 'solid', 'liquid', 'gas', or 'unknown'
    """
    mp = elem.melting_point  # Melting point in K
    bp = elem.boiling_point  # Boiling point in K

    # If we don't have data, we can't determine
    if mp is None and bp is None:
        return 'unknown'

    # Gas: boiling point below normal temperature
    if bp is not None and bp < NORMAL_TEMP_K:
        return 'gas'

    # Liquid: melting point below normal temp, boiling point above
    if mp is not None and mp < NORMAL_TEMP_K:
        if bp is None or bp > NORMAL_TEMP_K:
            # If boiling point is unknown but melting point is below normal temp
            # assume it's a solid (most elements are solid)
            if bp is not None:
                return 'liquid'
            else:
                # Need more information
                return 'solid'  # Default assumption

    # Solid: melting point above normal temperature
    if mp is not None and mp >= NORMAL_TEMP_K:
        return 'solid'

    return 'unknown'


# Test with known examples
print("Testing physical state determination:\n")
print(f"Normal conditions: {NORMAL_TEMP_K}K (25°C)\n")

test_cases = [
    # (Z, Symbol, Expected State)
    (1, 'H', 'gas'),       # Hydrogen
    (2, 'He', 'gas'),      # Helium
    (6, 'C', 'solid'),     # Carbon
    (7, 'N', 'gas'),       # Nitrogen
    (8, 'O', 'gas'),       # Oxygen
    (26, 'Fe', 'solid'),   # Iron
    (35, 'Br', 'liquid'),  # Bromine
    (80, 'Hg', 'liquid'),  # Mercury
]

correct = 0
total = 0

for z, sym, expected in test_cases:
    elem = element(z)
    determined = get_physical_state(elem)
    match = "✓" if determined == expected else "✗"

    print(f"{match} {sym:2} (Z={z:3}): Expected={expected:7}, Got={determined:7}")
    print(f"     MP={elem.melting_point}, BP={elem.boiling_point}")

    if determined == expected:
        correct += 1
    total += 1

print(f"\nAccuracy: {correct}/{total} = {100*correct/total:.1f}%")

# Count states for all elements
print("\n" + "="*60)
print("Physical states for all elements:")
print("="*60)

states_count = {'solid': [], 'liquid': [], 'gas': [], 'unknown': []}

for z in range(1, 119):
    try:
        elem = element(z)
        state = get_physical_state(elem)
        states_count[state].append(elem.symbol)
    except:
        pass

for state, elements in states_count.items():
    print(f"\n{state.upper()} ({len(elements)}): {', '.join(elements)}")
