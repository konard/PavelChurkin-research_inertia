import sqlite3
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mendeleev import element
import csv


def get_atomic_radius(elem):
    """Получаем атомный радиус с приоритетом разных типов радиусов"""
    # Пробуем разные типы радиусов в порядке приоритета
    if hasattr(elem, 'atomic_radius') and elem.atomic_radius is not None:
        return elem.atomic_radius

    # Ковалентные радиусы
    cov_radii = [
        elem.covalent_radius_cordero if hasattr(elem, 'covalent_radius_cordero') else None,
        elem.covalent_radius_pyykko if hasattr(elem, 'covalent_radius_pyykko') else None,
        elem.covalent_radius_pyykko_double if hasattr(elem, 'covalent_radius_pyykko_double') else None,
        elem.covalent_radius_pyykko_triple if hasattr(elem, 'covalent_radius_pyykko_triple') else None,
    ]

    for radius in cov_radii:
        if radius is not None:
            return radius

    # Ван-дер-ваальсовы радиусы
    vdw_radii = [
        elem.vdw_radius if hasattr(elem, 'vdw_radius') else None,
        elem.vdw_radius_bondi if hasattr(elem, 'vdw_radius_bondi') else None,
        elem.vdw_radius_alvarez if hasattr(elem, 'vdw_radius_alvarez') else None,
    ]

    for radius in vdw_radii:
        if radius is not None:
            return radius

    # Если нет радиуса, оценим через атомный объем
    if hasattr(elem, 'atomic_volume') and elem.atomic_volume is not None:
        volume_cm3_per_atom = elem.atomic_volume / 6.02214076e23
        volume_m3_per_atom = volume_cm3_per_atom * 1e-6
        r = (3 * volume_m3_per_atom / (4 * math.pi)) ** (1 / 3)
        radius_pm = r * 1e12
        return radius_pm

    return None


def calculate_atomic_volume(radius_pm):
    """Рассчитывает объем атома как объем шара по радиусу"""
    if radius_pm is None or radius_pm <= 0:
        return None
    r_m = radius_pm * 1e-12
    return (4 / 3) * math.pi * (r_m ** 3)


def get_effective_nuclear_charge(elem):
    """Оцениваем эффективный заряд ядра Z_eff"""
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


def calculate_ether_coefficient(elem):
    """Рассчитывает коэффициент эфирного сцепки k = m / V"""
    atomic_mass = elem.atomic_weight
    if atomic_mass is None:
        return None

    # Переводим массу в килограммы на атом
    mass_per_atom_kg = (atomic_mass * 1e-3) / 6.02214076e23

    # Получаем радиус
    atomic_radius = get_atomic_radius(elem)
    if atomic_radius is None or atomic_radius <= 0:
        return None

    # Рассчитываем объем
    volume_m3 = calculate_atomic_volume(atomic_radius)
    if volume_m3 is None or volume_m3 <= 0:
        return None

    # k = m / V
    k = mass_per_atom_kg / volume_m3
    if k <= 0:
        return None

    # Рассчитываем дополнительные параметры
    Z_eff = get_effective_nuclear_charge(elem)

    return {
        'atomic_number': elem.atomic_number,
        'symbol': elem.symbol,
        'name': elem.name,
        'atomic_mass_kg': mass_per_atom_kg,
        'atomic_radius_pm': atomic_radius,
        'atomic_volume_m3': volume_m3,
        'k_coefficient': k,
        'k_log10': math.log10(k),
        'period': elem.period,
        'group': elem.group_id,
        'zeff': Z_eff,
        'ionization_energy': elem.en_allen if hasattr(elem, 'en_allen') else None,
        'electronegativity_allen': elem.en_allen if hasattr(elem, 'en_allen') else None,
        'density': elem.density if hasattr(elem, 'density') else None,
        'block': elem.block
    }


def analyze_dependencies(elements_data):
    """Анализирует зависимости между k и другими параметрами"""
    valid_data = [e for e in elements_data if e['k_coefficient'] is not None]

    if len(valid_data) < 3:
        print("Недостаточно данных для анализа зависимостей")
        return None

    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВИСИМОСТЕЙ КОЭФФИЦИЕНТА СЦЕПЛЕНИЯ")
    print("=" * 70)

    # Собираем данные для анализа
    analysis_data = []
    for e in valid_data:
        if e['zeff'] is not None and e['atomic_volume_m3'] is not None:
            analysis_data.append({
                'symbol': e['symbol'],
                'k': e['k_coefficient'],
                'zeff': e['zeff'],
                'volume': e['atomic_volume_m3'],
                'zeff_over_v': e['zeff'] / e['atomic_volume_m3'],
                'zeff2_over_v': e['zeff'] ** 2 / e['atomic_volume_m3']
            })

    def safe_correlation(x, y):
        """Безопасный расчет корреляции"""
        if len(x) < 2 or len(y) < 2:
            return 0, 0, 0
        try:
            corr = np.corrcoef(x, y)[0, 1]
            if len(x) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                return corr, r_value ** 2, p_value
            return corr, 0, 0
        except:
            return 0, 0, 0

    # 1. Основная гипотеза: k ∝ Z_eff^n / V
    print("\n" + "=" * 50)
    print("ГИПОТЕЗА: k ∝ Z_eff^n / V")
    print("=" * 50)

    if len(analysis_data) > 2:
        # k vs Z_eff
        zeff_vals = [d['zeff'] for d in analysis_data]
        k_vals = [d['k'] for d in analysis_data]

        # Логарифмируем с проверкой
        log_zeff = np.log10(zeff_vals)
        log_k = np.log10(k_vals)
        corr, r2, pval = safe_correlation(log_zeff, log_k)
        print(f"\nk vs Z_eff (логарифмическая шкала):")
        print(f"  Точек данных: {len(zeff_vals)}")
        print(f"  Корреляция: {corr:.4f}")
        print(f"  R²: {r2:.4f}")

        # k vs Z_eff/V
        zeff_over_v = [d['zeff_over_v'] for d in analysis_data]
        log_zv = np.log10(zeff_over_v)
        corr_zv, r2_zv, pval_zv = safe_correlation(log_zv, log_k)
        print(f"\nk vs Z_eff/V:")
        print(f"  Корреляция: {corr_zv:.4f}")
        print(f"  R²: {r2_zv:.4f}")

        # k vs Z_eff²/V
        zeff2_over_v = [d['zeff2_over_v'] for d in analysis_data]
        log_z2v = np.log10(zeff2_over_v)
        corr_z2v, r2_z2v, pval_z2v = safe_correlation(log_z2v, log_k)
        print(f"\nk vs Z_eff²/V:")
        print(f"  Корреляция: {corr_z2v:.4f}")
        print(f"  R²: {r2_z2v:.4f}")

    # 2. Корреляции в обычной шкале
    print("\n" + "=" * 50)
    print("КОРРЕЛЯЦИИ В ОБЫЧНОЙ ШКАЛЕ")
    print("=" * 50)

    # Z vs k
    z_vals = [e['atomic_number'] for e in valid_data]
    k_all = [e['k_coefficient'] for e in valid_data]
    corr_z, r2_z, _ = safe_correlation(z_vals, k_all)
    print(f"\nZ vs k:")
    print(f"  Корреляция: {corr_z:.4f}")
    print(f"  R²: {r2_z:.4f}")

    # Объем vs k
    volume_vals = [e['atomic_volume_m3'] for e in valid_data]
    corr_v, r2_v, _ = safe_correlation(volume_vals, k_all)
    print(f"\nV vs k:")
    print(f"  Корреляция: {corr_v:.4f}")
    print(f"  R²: {r2_v:.4f}")

    # 3. Статистика по блокам
    print("\n" + "=" * 50)
    print("СТАТИСТИКА ПО БЛОКАМ ЭЛЕМЕНТОВ")
    print("=" * 50)

    elements_by_block = {'s': [], 'p': [], 'd': [], 'f': []}
    for e in valid_data:
        block = e.get('block')
        if block in elements_by_block:
            elements_by_block[block].append(e)

    for block, elements in elements_by_block.items():
        if elements:
            k_values = [e['k_coefficient'] for e in elements]
            symbols = [e['symbol'] for e in elements]
            print(f"\n{block}-элементы ({len(elements)}):")
            print(f"  Элементы: {', '.join(symbols)}")
            print(f"  Средний k: {np.mean(k_values):.3e} кг/м³")
            print(f"  Мин/Макс: {min(k_values):.3e} / {max(k_values):.3e} кг/м³")

    return {'analysis_data': analysis_data, 'valid_data': valid_data}


def plot_all_elements_with_labels(params):
    """Строит графики со всеми подписанными элементами"""
    analysis_data = params.get('analysis_data', [])
    if not analysis_data:
        print("Нет данных для построения графиков")
        return

    symbols = [d['symbol'] for d in analysis_data]
    zeff_vals = [d['zeff'] for d in analysis_data]
    k_vals = [d['k'] for d in analysis_data]
    zeff_over_v = [d['zeff_over_v'] for d in analysis_data]
    zeff2_over_v = [d['zeff2_over_v'] for d in analysis_data]

    # 1. Основной график: k vs Z_eff
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.loglog(zeff_vals, k_vals, 'o', alpha=0.5, markersize=6)

    # Подписываем ВСЕ элементы
    for i, (z, k, sym) in enumerate(zip(zeff_vals, k_vals, symbols)):
        # Сдвигаем позиции подписей для разных элементов
        offset_x = 0
        offset_y = 0

        # Пропускаем перекрывающиеся подписи
        skip = False
        for j in range(i):
            z_prev, k_prev = zeff_vals[j], k_vals[j]
            if abs(z - z_prev) / z_prev < 0.05 and abs(k - k_prev) / k_prev < 0.05:
                skip = True
                break

        if not skip:
            # Добавляем небольшие случайные смещения для лучшей читаемости
            offset_x = np.random.uniform(-0.02, 0.02) * z
            offset_y = np.random.uniform(-0.02, 0.02) * k

            plt.annotate(sym, (z, k),
                         xytext=(offset_x, offset_y),
                         textcoords='offset points',
                         fontsize=7,
                         ha='center',
                         va='bottom',
                         alpha=0.8)

    plt.xlabel('Z_eff (эффективный заряд ядра)', fontsize=12)
    plt.ylabel('Коэффициент k (кг/м³)', fontsize=12)
    plt.title('Зависимость k от эффективного заряда ядра\n(все элементы подписаны)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 2. k vs Z_eff/V
    plt.subplot(2, 2, 2)
    plt.loglog(zeff_over_v, k_vals, 'o', alpha=0.5, markersize=6)

    for i, (zv, k, sym) in enumerate(zip(zeff_over_v, k_vals, symbols)):
        offset_x = 0
        offset_y = 0

        skip = False
        for j in range(i):
            zv_prev, k_prev = zeff_over_v[j], k_vals[j]
            if abs(zv - zv_prev) / zv_prev < 0.05 and abs(k - k_prev) / k_prev < 0.05:
                skip = True
                break

        if not skip:
            offset_x = np.random.uniform(-0.02, 0.02) * zv
            offset_y = np.random.uniform(-0.02, 0.02) * k

            plt.annotate(sym, (zv, k),
                         xytext=(offset_x, offset_y),
                         textcoords='offset points',
                         fontsize=7,
                         ha='center',
                         va='bottom',
                         alpha=0.8)

    plt.xlabel('Z_eff / V (м⁻³)', fontsize=12)
    plt.ylabel('Коэффициент k (кг/м³)', fontsize=12)
    plt.title('k vs Z_eff/V (все элементы подписаны)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 3. k vs Z_eff²/V
    plt.subplot(2, 2, 3)
    plt.loglog(zeff2_over_v, k_vals, 'o', alpha=0.5, markersize=6)

    for i, (z2v, k, sym) in enumerate(zip(zeff2_over_v, k_vals, symbols)):
        offset_x = 0
        offset_y = 0

        skip = False
        for j in range(i):
            z2v_prev, k_prev = zeff2_over_v[j], k_vals[j]
            if abs(z2v - z2v_prev) / z2v_prev < 0.05 and abs(k - k_prev) / k_prev < 0.05:
                skip = True
                break

        if not skip:
            offset_x = np.random.uniform(-0.02, 0.02) * z2v
            offset_y = np.random.uniform(-0.02, 0.02) * k

            plt.annotate(sym, (z2v, k),
                         xytext=(offset_x, offset_y),
                         textcoords='offset points',
                         fontsize=7,
                         ha='center',
                         va='bottom',
                         alpha=0.8)

    plt.xlabel('Z_eff² / V (м⁻³)', fontsize=12)
    plt.ylabel('Коэффициент k (кг/м³)', fontsize=12)
    plt.title('k vs Z_eff²/V (все элементы подписаны)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 4. График по типам элементов (цветами)
    plt.subplot(2, 2, 4)

    valid_data = params.get('valid_data', [])
    colors = {'s': 'red', 'p': 'blue', 'd': 'green', 'f': 'purple'}

    for block in colors.keys():
        block_k = []
        block_z = []
        block_symbols = []

        for e in valid_data:
            if e.get('block') == block and 'zeff' in e and e['zeff'] is not None:
                block_k.append(e['k_coefficient'])
                block_z.append(e['zeff'])
                block_symbols.append(e['symbol'])

        if block_k:
            plt.loglog(block_z, block_k, 'o', color=colors[block],
                       alpha=0.6, markersize=6, label=f'{block}-элементы')

            # Подписываем все элементы в блоке
            for z, k, sym in zip(block_z, block_k, block_symbols):
                offset_x = np.random.uniform(-0.02, 0.02) * z
                offset_y = np.random.uniform(-0.02, 0.02) * k

                plt.annotate(sym, (z, k),
                             xytext=(offset_x, offset_y),
                             textcoords='offset points',
                             fontsize=6,
                             ha='center',
                             va='bottom',
                             alpha=0.7,
                             color=colors[block])

    plt.xlabel('Z_eff', fontsize=12)
    plt.ylabel('k (кг/м³)', fontsize=12)
    plt.title('k vs Z_eff по типам элементов\n(все элементы подписаны)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('all_elements_labeled.png', dpi=300, bbox_inches='tight')
    print("\nГрафик со всеми подписанными элементами сохранен как 'all_elements_labeled.png'")

    # 5. Дополнительный график: Z vs k (линейная шкала)
    plt.figure(figsize=(12, 8))

    z_vals = [e['atomic_number'] for e in valid_data]
    k_vals_all = [e['k_coefficient'] for e in valid_data]
    symbols_all = [e['symbol'] for e in valid_data]

    plt.scatter(z_vals, k_vals_all, alpha=0.6, s=50)

    # Подписываем ВСЕ элементы с адаптивным размещением
    for i, (z, k, sym) in enumerate(zip(z_vals, k_vals_all, symbols_all)):
        # Определяем смещение на основе позиции
        if i % 3 == 0:
            offset_y = 0.05 * k
        elif i % 3 == 1:
            offset_y = -0.05 * k
        else:
            offset_y = 0.1 * k

        # Сдвигаем по X для четных/нечетных
        offset_x = 0.5 if i % 2 == 0 else -0.5

        plt.annotate(sym, (z, k),
                     xytext=(offset_x, offset_y),
                     textcoords='offset points',
                     fontsize=7,
                     ha='center',
                     va='bottom' if offset_y > 0 else 'top',
                     alpha=0.8)

    plt.xlabel('Атомный номер Z', fontsize=12)
    plt.ylabel('Коэффициент k (кг/м³)', fontsize=12)
    plt.title('Зависимость k от атомного номера Z\n(все элементы подписаны)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('k_vs_Z_labeled.png', dpi=300, bbox_inches='tight')
    print("График k vs Z сохранен как 'k_vs_Z_labeled.png'")

    plt.show()


def main():
    print("=" * 70)
    print("РАСЧЕТ И АНАЛИЗ КОЭФФИЦИЕНТОВ ЭФИРНОГО СЦЕПЛЕНИЯ")
    print("=" * 70)

    elements_data = []
    valid_elements = 0

    # Собираем данные для всех элементов
    for atomic_number in range(1, 119):
        try:
            elem = element(atomic_number)
            result = calculate_ether_coefficient(elem)

            if result:
                elements_data.append(result)
                valid_elements += 1
                print(f"{atomic_number:3}. {elem.symbol:2} - OK")
            else:
                print(f"{atomic_number:3}. {elem.symbol:2} - пропущен (нет данных)")

        except Exception as e:
            print(f"{atomic_number:3}. ? - ошибка: {str(e)}")
            continue

    print(f"\nУспешно обработано: {valid_elements} элементов")

    if valid_elements > 0:
        # Анализ зависимостей
        params = analyze_dependencies(elements_data)

        if params:
            # Построение графиков со всеми подписями
            plot_all_elements_with_labels(params)

            # Сохранение данных в CSV
            with open('ether_coefficients_all.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Z', 'Symbol', 'Name', 'k (кг/м³)', 'Z_eff', 'Volume (м³)',
                                 'Radius (пм)', 'Period', 'Group', 'Block'])

                for e in elements_data:
                    if e['k_coefficient']:
                        writer.writerow([
                            e['atomic_number'],
                            e['symbol'],
                            e['name'],
                            e['k_coefficient'],
                            e['zeff'] if 'zeff' in e else '',
                            e['atomic_volume_m3'],
                            e['atomic_radius_pm'],
                            e['period'],
                            e['group'],
                            e.get('block', '')
                        ])

            print(f"\nДанные сохранены в 'ether_coefficients_all.csv'")

    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 70)


if __name__ == "__main__":
    main()
