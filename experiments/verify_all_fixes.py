"""
Проверка всех исправлений
"""
from mendeleev import element
import sys
sys.path.insert(0, '/tmp/gh-issue-solver-1766138101227')
from research_inertia import get_effective_nuclear_charge

def test_element(z, name, expected_z_eff):
    """Проверка элемента"""
    elem = element(z)
    z_eff = get_effective_nuclear_charge(elem)
    status = "✓" if abs(z_eff - expected_z_eff) < 0.01 else "✗"
    print(f"{name:10} (Z={z:2}): Z_eff = {z_eff:.2f}, ожидается {expected_z_eff:.2f} {status}")
    return abs(z_eff - expected_z_eff) < 0.01

print("=" * 70)
print("ПРОВЕРКА ИСПРАВЛЕНИЙ")
print("=" * 70)

tests = [
    (7, "Азот", 3.9),      # Из issue
    (82, "Свинец", 4.15),  # Из комментария PavelChurkin
    (30, "Цинк", 2.85),    # Исправлено
    (26, "Железо", 2.85),  # Исправлено
    (21, "Скандий", 2.85), # Исправлено
    (1, "Водород", 1.0),
    (2, "Гелий", 1.7),
]

all_pass = True
for z, name, expected in tests:
    if not test_element(z, name, expected):
        all_pass = False

print("=" * 70)
if all_pass:
    print("✓ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
else:
    print("✗ НЕКОТОРЫЕ ПРОВЕРКИ ПРОВАЛИЛИСЬ")
print("=" * 70)
