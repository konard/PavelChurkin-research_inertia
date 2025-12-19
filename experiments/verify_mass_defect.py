#!/usr/bin/env python
"""
Проверка расчетов дефекта массы
"""
import sqlite3
from mendeleev import element

# Константы
PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg
AMU_TO_KG = 1.66053906660e-27  # Atomic mass unit to kg conversion

def verify_mass_defect_calculations():
    """Проверяет корректность расчета дефекта массы для нескольких элементов"""

    print("=" * 80)
    print("ПРОВЕРКА РАСЧЕТОВ ДЕФЕКТА МАССЫ")
    print("=" * 80)

    # Подключаемся к базе данных
    conn = sqlite3.connect('research_inertia.db')
    cursor = conn.cursor()

    # Проверяем несколько элементов
    test_elements = [1, 6, 26, 82]  # H, C, Fe, Pb

    for z in test_elements:
        elem = element(z)
        print(f"\n{'='*80}")
        print(f"Элемент: {elem.name} ({elem.symbol}), Z = {z}")
        print(f"{'='*80}")

        # Получаем данные из БД
        cursor.execute('''
            SELECT atomic_mass_kg, neutron_count, neutron_rounding_diff,
                   nucleon_mass_sum_kg, mass_defect_kg, mass_defect_per_volume
            FROM elements WHERE atomic_number = ?
        ''', (z,))

        row = cursor.fetchone()
        if row:
            db_mass_kg, db_n, db_rounding, db_nucleon_sum, db_mass_defect, db_md_per_vol = row

            # Пересчитываем вручную
            atomic_mass_amu = elem.atomic_weight
            atomic_mass_kg = (atomic_mass_amu * 1e-3) / 6.02214076e23

            A_exact = atomic_mass_amu
            A_rounded = round(A_exact)
            rounding_diff = A_exact - A_rounded

            N_exact = A_exact - z
            nucleon_mass_sum = z * PROTON_MASS + N_exact * NEUTRON_MASS
            mass_defect = atomic_mass_kg - nucleon_mass_sum

            print(f"\nАтомная масса: {atomic_mass_amu:.6f} а.е.м. = {atomic_mass_kg:.6e} кг")
            print(f"Количество протонов (Z): {z}")
            print(f"Количество нейтронов (N): {N_exact:.6f}")
            print(f"Округленное массовое число (A): {A_rounded}")
            print(f"Разница округления: {rounding_diff:.6f}")

            print(f"\nСумма масс нуклонов:")
            print(f"  Расчет: {z} × {PROTON_MASS:.6e} + {N_exact:.4f} × {NEUTRON_MASS:.6e}")
            print(f"  = {nucleon_mass_sum:.6e} кг")
            print(f"  БД:     {db_nucleon_sum:.6e} кг")
            print(f"  Совпадение: {'✓' if abs(nucleon_mass_sum - db_nucleon_sum) < 1e-35 else '✗'}")

            print(f"\nДефект массы:")
            print(f"  Расчет: {atomic_mass_kg:.6e} - {nucleon_mass_sum:.6e}")
            print(f"  = {mass_defect:.6e} кг")
            print(f"  БД:     {db_mass_defect:.6e} кг")
            print(f"  Совпадение: {'✓' if abs(mass_defect - db_mass_defect) < 1e-35 else '✗'}")

            # Дефект массы всегда отрицательный (энергия связи положительна)
            print(f"\nДефект массы {'отрицательный (правильно)' if mass_defect < 0 else 'положительный (ошибка!)'}")
            print(f"Энергия связи: {abs(mass_defect) * 9e16 / 1.602176634e-13:.2f} МэВ")

        else:
            print(f"Элемент с Z={z} не найден в базе данных")

    conn.close()

    print(f"\n{'='*80}")
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print(f"{'='*80}")

if __name__ == "__main__":
    verify_mass_defect_calculations()
