import sqlite3
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mendeleev import element
import json
import os

CACHE_FILE = 'elements_cache.json'
DB_FILE = 'research_inertia.db'

# Physical constants (in kg)
PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg
AMU_TO_KG = 1.66053906660e-27  # Atomic mass unit to kg conversion

# Normal conditions for physical state determination
NORMAL_TEMP_K = 298.15  # 25°C in Kelvin


def load_cache():
    """Загружает кэшированные данные из JSON файла"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}")
    return None


def save_cache(elements_data):
    """Сохраняет данные элементов в JSON кэш"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(elements_data, f, ensure_ascii=False, indent=2)
        print(f"Кэш сохранен в '{CACHE_FILE}'")
    except Exception as e:
        print(f"Ошибка сохранения кэша: {e}")


def save_to_database(elements_data):
    """Сохраняет данные в SQLite базу данных"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Создаем таблицу
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS elements (
                atomic_number INTEGER PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                atomic_mass_kg REAL,
                atomic_radius_pm REAL,
                atomic_volume_m3 REAL,
                k_coefficient REAL,
                k_log10 REAL,
                period INTEGER,
                group_id INTEGER,
                zeff REAL,
                ionization_energy REAL,
                electronegativity_allen REAL,
                density REAL,
                block TEXT,
                neutron_count REAL,
                neutron_rounding_diff REAL,
                nucleon_mass_sum_kg REAL,
                mass_defect_kg REAL,
                mass_defect_per_volume REAL,
                physical_state TEXT
            )
        ''')

        # Удаляем старые данные
        cursor.execute('DELETE FROM elements')

        # Вставляем новые данные
        for e in elements_data:
            if e.get('k_coefficient') is not None:
                cursor.execute('''
                    INSERT INTO elements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    e['atomic_number'],
                    e['symbol'],
                    e['name'],
                    e['atomic_mass_kg'],
                    e['atomic_radius_pm'],
                    e['atomic_volume_m3'],
                    e['k_coefficient'],
                    e['k_log10'],
                    e['period'],
                    e['group'],
                    e['zeff'],
                    e.get('ionization_energy'),
                    e.get('electronegativity_allen'),
                    e.get('density'),
                    e.get('block'),
                    e.get('neutron_count'),
                    e.get('neutron_rounding_diff'),
                    e.get('nucleon_mass_sum_kg'),
                    e.get('mass_defect_kg'),
                    e.get('mass_defect_per_volume'),
                    e.get('physical_state')
                ))

        conn.commit()
        conn.close()
        print(f"База данных сохранена в '{DB_FILE}'")
        return True
    except Exception as e:
        print(f"Ошибка сохранения в базу данных: {e}")
        return False


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


def calculate_mass_defect(elem, atomic_mass_kg):
    """
    Рассчитывает дефект массы и связанные параметры.

    Дефект массы = атомная масса - сумма масс нуклонов
    (не вычитая массу ядра, как принято в классической физике,
    а вычитая сумму масс протонов и нейтронов)

    Args:
        elem: Элемент из библиотеки mendeleev
        atomic_mass_kg: Атомная масса в килограммах

    Returns:
        dict: Словарь с полями neutron_count, neutron_rounding_diff,
              nucleon_mass_sum_kg, mass_defect_kg
    """
    Z = elem.atomic_number  # количество протонов

    # Получаем атомную массу в а.е.м.
    atomic_mass_amu = elem.atomic_weight
    if atomic_mass_amu is None:
        return None

    # Рассчитываем количество нейтронов
    # N = A - Z, где A - массовое число (округленная атомная масса)
    A_exact = atomic_mass_amu  # точная атомная масса
    A_rounded = round(A_exact)  # округленная атомная масса

    # Разница округления (со знаком)
    rounding_diff = A_exact - A_rounded

    # Количество нейтронов (может быть дробным для точного расчета)
    N_exact = A_exact - Z

    # Сумма масс нуклонов
    nucleon_mass_sum_kg = Z * PROTON_MASS + N_exact * NEUTRON_MASS

    # Дефект массы = атомная масса - сумма масс нуклонов
    mass_defect_kg = atomic_mass_kg - nucleon_mass_sum_kg

    return {
        'neutron_count': N_exact,
        'neutron_rounding_diff': rounding_diff,
        'nucleon_mass_sum_kg': nucleon_mass_sum_kg,
        'mass_defect_kg': mass_defect_kg
    }


def get_physical_state(elem):
    """
    Определяет агрегатное состояние элемента при нормальных условиях (25°C = 298.15K, 1 атм).

    Args:
        elem: Элемент из библиотеки mendeleev

    Returns:
        str: 'solid', 'liquid', 'gas', или 'unknown'
    """
    mp = elem.melting_point  # Температура плавления в K
    bp = elem.boiling_point  # Температура кипения в K

    # Если нет данных, не можем определить
    if mp is None and bp is None:
        return 'unknown'

    # Газ: температура кипения ниже нормальной температуры
    if bp is not None and bp < NORMAL_TEMP_K:
        return 'gas'

    # Жидкость: температура плавления ниже нормальной, кипения выше
    if mp is not None and mp < NORMAL_TEMP_K:
        if bp is not None and bp > NORMAL_TEMP_K:
            return 'liquid'
        # Если температура кипения неизвестна, предполагаем твердое состояние
        return 'solid'

    # Твердое: температура плавления выше нормальной температуры
    if mp is not None and mp >= NORMAL_TEMP_K:
        return 'solid'

    return 'unknown'


def get_effective_nuclear_charge(elem):
    """
    Вычисляет эффективный заряд ядра по правилу Слейтера.

    Правила Слейтера для расчета экранирования S:

    Для электронов на ns или np орбиталях:
    - Электроны той же группы (ns, np): каждый дает вклад 0.35
      (в группе 1s вклад 0.30)
    - Электроны слоя (n-1): каждый дает вклад 0.85
    - Электроны слоя (n-2) и глубже: каждый дает вклад 1.00

    Для электронов на nd или nf орбиталях:
    - Электроны той же группы: каждый дает вклад 0.35
    - Все электроны в группах левее: каждый дает вклад 1.00

    Args:
        elem: Элемент из библиотеки mendeleev

    Returns:
        float: Эффективный заряд ядра Z_eff = Z - S

    Examples:
        Азот (Z=7, конфигурация 1s² 2s² 2p³):
        S = (4 * 0.35) + (2 * 0.85) = 1.4 + 1.7 = 3.1
        Z_eff = 7 - 3.1 = 3.9

        Цинк (Z=30, конфигурация [Ar] 3d¹⁰ 4s²):
        S = (1 * 0.35) + (18 * 0.85) + (10 * 1.00) = 0.35 + 15.3 + 10 = 25.65
        Z_eff = 30 - 25.65 = 4.35
    """
    Z = elem.atomic_number

    # Получаем электронную конфигурацию как OrderedDict {(n, orbital): count}
    ec_conf = elem.ec.conf

    # Преобразуем в список для удобства обработки
    config = [(n, orb, count) for (n, orb), count in ec_conf.items()]

    if not config:
        return float(Z)

    # Находим внешний (валентный) электрон - последний в конфигурации
    n_outer, orb_outer, count_outer = config[-1]

    # Вычисляем константу экранирования S
    S = 0.0

    if orb_outer in ['s', 'p']:
        # Правила для ns, np орбиталей
        for n, orb, count in config:
            if (n, orb) == (n_outer, orb_outer):
                # Та же орбиталь - не учитываем экранируемый электрон
                if n == 1 and orb == 's':
                    # Особый случай для 1s
                    S += (count - 1) * 0.30
                else:
                    S += (count - 1) * 0.35
            elif n == n_outer and orb in ['s', 'p']:
                # Та же оболочка, другая орбиталь в группе s,p
                if n == 1:
                    S += count * 0.30
                else:
                    S += count * 0.35
            elif n == n_outer - 1:
                # Одна оболочка ниже (n-1)
                # Важно: для s/p орбиталей коэффициент 0.85, для d/f - 1.00
                if orb in ['s', 'p']:
                    S += count * 0.85
                else:
                    # d, f орбитали считаются как внутренний слой
                    S += count * 1.00
            elif n < n_outer - 1:
                # Две и более оболочек ниже (n-2 и глубже)
                S += count * 1.00

    elif orb_outer in ['d', 'f']:
        # Правила для nd, nf орбиталей
        for n, orb, count in config:
            if (n, orb) == (n_outer, orb_outer):
                # Та же орбиталь - не учитываем экранируемый электрон
                S += (count - 1) * 0.35
            elif n == n_outer and orb == orb_outer:
                # Та же группа
                S += count * 0.35
            else:
                # Все остальные электроны левее
                S += count * 1.00

    Z_eff = Z - S
    return Z_eff


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

    # Рассчитываем дефект массы
    mass_defect_data = calculate_mass_defect(elem, mass_per_atom_kg)

    # Получаем физическое состояние
    physical_state = get_physical_state(elem)

    # Получаем первую энергию ионизации
    ionization_energy = None
    if hasattr(elem, 'ionenergies') and elem.ionenergies:
        ionization_energy = elem.ionenergies.get(1)  # Первая энергия ионизации

    # Создаем базовый результат
    result = {
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
        'ionization_energy': ionization_energy,
        'electronegativity_allen': elem.en_allen if hasattr(elem, 'en_allen') else None,
        'density': elem.density if hasattr(elem, 'density') else None,
        'block': elem.block,
        'physical_state': physical_state
    }

    # Добавляем данные о дефекте массы, если они есть
    if mass_defect_data:
        result['neutron_count'] = mass_defect_data['neutron_count']
        result['neutron_rounding_diff'] = mass_defect_data['neutron_rounding_diff']
        result['nucleon_mass_sum_kg'] = mass_defect_data['nucleon_mass_sum_kg']
        result['mass_defect_kg'] = mass_defect_data['mass_defect_kg']
        result['mass_defect_per_volume'] = mass_defect_data['mass_defect_kg'] / volume_m3
    else:
        result['neutron_count'] = None
        result['neutron_rounding_diff'] = None
        result['nucleon_mass_sum_kg'] = None
        result['mass_defect_kg'] = None
        result['mass_defect_per_volume'] = None

    return result


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
        # Фильтруем элементы с положительным Z_eff для логарифмического анализа
        if e['zeff'] is not None and e['atomic_volume_m3'] is not None and e['zeff'] > 0:
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
        # k vs Z_eff/V
        zeff_over_v = [d['zeff_over_v'] for d in analysis_data]
        k_vals = [d['k'] for d in analysis_data]
        log_zv = np.log10(zeff_over_v)
        log_k = np.log10(k_vals)
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

    # 2. Статистика по блокам
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
    valid_data = params.get('valid_data', [])

    if not analysis_data:
        print("Нет данных для построения графиков")
        return

    symbols = [d['symbol'] for d in analysis_data]
    k_vals = [d['k'] for d in analysis_data]
    zeff_over_v = [d['zeff_over_v'] for d in analysis_data]

    # Цвета периодов для легенды
    period_colors = {
        1: '#FF0000',  # Красный
        2: '#FF7F00',  # Оранжевый
        3: '#FFFF00',  # Желтый
        4: '#00FF00',  # Зеленый
        5: '#0000FF',  # Синий
        6: '#4B0082',  # Индиго
        7: '#9400D3'   # Фиолетовый
    }

    # 1. График k vs Z_eff/V
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.loglog(zeff_over_v, k_vals, 'o', alpha=0.5, markersize=6)

    # Подписываем элементы с простым размещением
    for i, (zv, k, sym) in enumerate(zip(zeff_over_v, k_vals, symbols)):
        # Адаптивное размещение подписей
        offset_angle = (i * 30) % 360  # Разные углы для разных элементов
        offset_r = 8  # Радиус смещения в пикселях
        offset_x = offset_r * np.cos(np.radians(offset_angle))
        offset_y = offset_r * np.sin(np.radians(offset_angle))

        plt.annotate(sym, (zv, k),
                     xytext=(offset_x, offset_y),
                     textcoords='offset points',
                     fontsize=6,
                     ha='center',
                     va='center',
                     alpha=0.8)

    plt.xlabel('Z_eff / V (м⁻³)', fontsize=12)
    plt.ylabel('Коэффициент k (кг/м³)', fontsize=12)
    plt.title('k vs Z_eff/V (все элементы подписаны)', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 2. График по типам элементов (цветами блоков)
    plt.subplot(1, 2, 2)

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

            # Подписываем элементы
            for idx, (z, k, sym) in enumerate(zip(block_z, block_k, block_symbols)):
                offset_angle = (idx * 30) % 360
                offset_r = 8
                offset_x = offset_r * np.cos(np.radians(offset_angle))
                offset_y = offset_r * np.sin(np.radians(offset_angle))

                plt.annotate(sym, (z, k),
                             xytext=(offset_x, offset_y),
                             textcoords='offset points',
                             fontsize=5,
                             ha='center',
                             va='center',
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

    # 3. Основной график: Z vs k с цветами по периодам и легендой
    plt.figure(figsize=(16, 10))

    z_vals = [e['atomic_number'] for e in valid_data]
    k_vals_all = [e['k_coefficient'] for e in valid_data]
    periods = [e['period'] for e in valid_data]
    symbols_all = [e['symbol'] for e in valid_data]

    # Группируем по периодам для легенды
    plotted_periods = set()
    for period in sorted(set(periods)):
        period_z = []
        period_k = []
        period_symbols = []

        for z, k, p, sym in zip(z_vals, k_vals_all, periods, symbols_all):
            if p == period:
                period_z.append(z)
                period_k.append(k)
                period_symbols.append(sym)

        if period_z:
            color = period_colors.get(period, '#808080')
            plt.scatter(period_z, period_k, alpha=0.7, s=60, c=color,
                       label=f'Период {period}', edgecolors='black', linewidth=0.5)

            # Подписываем элементы с адаптивным размещением
            for idx, (z, k, sym) in enumerate(zip(period_z, period_k, period_symbols)):
                offset_angle = (idx * 40) % 360
                offset_r = 10
                offset_x = offset_r * np.cos(np.radians(offset_angle))
                offset_y = offset_r * np.sin(np.radians(offset_angle))

                plt.annotate(sym, (z, k),
                             xytext=(offset_x, offset_y),
                             textcoords='offset points',
                             fontsize=7,
                             ha='center',
                             va='center',
                             alpha=0.9,
                             weight='bold')

    plt.xlabel('Атомный номер Z', fontsize=14)
    plt.ylabel('Коэффициент k (кг/м³)', fontsize=14)
    plt.title('Зависимость К от атомного номера\n(цвет соответствует периоду элемента)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('k_vs_Z_labeled.png', dpi=300, bbox_inches='tight')
    print("График k vs Z с легендой периодов сохранен как 'k_vs_Z_labeled.png'")

    plt.show()


def plot_mass_defect_dependencies(valid_data):
    """
    Строит графики зависимостей с дефектом массы:
    1. Коэффициент сцепления k vs дефект массы/объём
    2. Z_eff vs дефект массы
    """
    # Фильтруем элементы с данными о дефекте массы
    data_with_mass_defect = [e for e in valid_data
                             if e.get('mass_defect_kg') is not None
                             and e.get('mass_defect_per_volume') is not None]

    if len(data_with_mass_defect) < 3:
        print("Недостаточно данных для построения графиков с дефектом массы")
        return

    print(f"\nПостроение графиков с дефектом массы для {len(data_with_mass_defect)} элементов...")

    # Извлекаем данные
    symbols = [e['symbol'] for e in data_with_mass_defect]
    k_vals = [e['k_coefficient'] for e in data_with_mass_defect]
    mass_defect_per_volume = [e['mass_defect_per_volume'] for e in data_with_mass_defect]
    mass_defect = [e['mass_defect_kg'] for e in data_with_mass_defect]
    zeff = [e['zeff'] for e in data_with_mass_defect]

    # Создаем новое окно с двумя графиками
    plt.figure(figsize=(16, 8))

    # График 1: Коэффициент сцепления vs дефект массы/объём
    plt.subplot(1, 2, 1)

    # Используем scatter с цветом по атомному номеру
    atomic_numbers = [e['atomic_number'] for e in data_with_mass_defect]
    scatter1 = plt.scatter(mass_defect_per_volume, k_vals,
                          c=atomic_numbers, cmap='viridis',
                          alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Подписываем элементы
    for i, (mdv, k, sym) in enumerate(zip(mass_defect_per_volume, k_vals, symbols)):
        offset_angle = (i * 30) % 360
        offset_r = 8
        offset_x = offset_r * np.cos(np.radians(offset_angle))
        offset_y = offset_r * np.sin(np.radians(offset_angle))

        plt.annotate(sym, (mdv, k),
                     xytext=(offset_x, offset_y),
                     textcoords='offset points',
                     fontsize=6,
                     ha='center',
                     va='center',
                     alpha=0.8)

    plt.xlabel('Дефект массы / Объём (кг/м³)', fontsize=12)
    plt.ylabel('Коэффициент сцепления k (кг/м³)', fontsize=12)
    plt.title('Зависимость коэффициента сцепления\nот дефекта массы на единицу объема', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter1, label='Атомный номер Z')

    # График 2: Z_eff vs дефект массы
    plt.subplot(1, 2, 2)

    scatter2 = plt.scatter(mass_defect, zeff,
                          c=atomic_numbers, cmap='viridis',
                          alpha=0.6, s=60, edgecolors='black', linewidth=0.5)

    # Подписываем элементы
    for i, (md, z, sym) in enumerate(zip(mass_defect, zeff, symbols)):
        offset_angle = (i * 30) % 360
        offset_r = 8
        offset_x = offset_r * np.cos(np.radians(offset_angle))
        offset_y = offset_r * np.sin(np.radians(offset_angle))

        plt.annotate(sym, (md, z),
                     xytext=(offset_x, offset_y),
                     textcoords='offset points',
                     fontsize=6,
                     ha='center',
                     va='center',
                     alpha=0.8)

    plt.xlabel('Дефект массы (кг)', fontsize=12)
    plt.ylabel('Эффективный заряд ядра Z_eff', fontsize=12)
    plt.title('Зависимость эффективного заряда ядра\nот дефекта массы', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter2, label='Атомный номер Z')

    plt.tight_layout()
    plt.savefig('mass_defect_dependencies.png', dpi=300, bbox_inches='tight')
    print("Графики с дефектом массы сохранены как 'mass_defect_dependencies.png'")

    plt.show()


def plot_volume_vs_ionization_energy(valid_data):
    """
    Строит графики зависимости объема от энергии ионизации
    с легендой агрегатного состояния элемента.
    """
    # Фильтруем элементы с данными об объеме и энергии ионизации
    data_with_ie = [e for e in valid_data
                    if e.get('ionization_energy') is not None
                    and e.get('atomic_volume_m3') is not None
                    and e.get('physical_state') is not None]

    if len(data_with_ie) < 3:
        print("Недостаточно данных для построения графиков объем vs энергия ионизации")
        return

    print(f"\nПостроение графиков объем vs энергия ионизации для {len(data_with_ie)} элементов...")

    # Цвета и маркеры для агрегатных состояний
    state_colors = {
        'solid': '#1f77b4',    # Синий
        'liquid': '#ff7f0e',   # Оранжевый
        'gas': '#2ca02c',      # Зеленый
        'unknown': '#7f7f7f'   # Серый
    }

    state_names_ru = {
        'solid': 'Твердые',
        'liquid': 'Жидкие',
        'gas': 'Газообразные',
        'unknown': 'Неизвестно'
    }

    # Группируем по агрегатному состоянию
    data_by_state = {'solid': [], 'liquid': [], 'gas': [], 'unknown': []}
    for e in data_with_ie:
        state = e['physical_state']
        data_by_state[state].append(e)

    # Создаем фигуру с двумя графиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # График 1: Объем vs Энергия ионизации (линейная шкала)
    for state in ['solid', 'liquid', 'gas', 'unknown']:
        if not data_by_state[state]:
            continue

        volumes = [e['atomic_volume_m3'] for e in data_by_state[state]]
        ionization_energies = [e['ionization_energy'] for e in data_by_state[state]]
        symbols = [e['symbol'] for e in data_by_state[state]]

        ax1.scatter(ionization_energies, volumes,
                   color=state_colors[state],
                   label=state_names_ru[state],
                   alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

        # Подписываем элементы
        for i, (ie, vol, sym) in enumerate(zip(ionization_energies, volumes, symbols)):
            offset_angle = (i * 30) % 360
            offset_r = 8
            offset_x = offset_r * np.cos(np.radians(offset_angle))
            offset_y = offset_r * np.sin(np.radians(offset_angle))

            ax1.annotate(sym, (ie, vol),
                        xytext=(offset_x, offset_y),
                        textcoords='offset points',
                        fontsize=6,
                        ha='center',
                        va='center',
                        alpha=0.8)

    ax1.set_xlabel('Энергия ионизации (эВ)', fontsize=12)
    ax1.set_ylabel('Атомный объём (м³)', fontsize=12)
    ax1.set_title('Зависимость объёма от энергии ионизации\n(линейная шкала)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)

    # График 2: Объем vs Энергия ионизации (логарифмическая шкала)
    for state in ['solid', 'liquid', 'gas', 'unknown']:
        if not data_by_state[state]:
            continue

        volumes = [e['atomic_volume_m3'] for e in data_by_state[state]]
        ionization_energies = [e['ionization_energy'] for e in data_by_state[state]]
        symbols = [e['symbol'] for e in data_by_state[state]]

        ax2.scatter(ionization_energies, volumes,
                   color=state_colors[state],
                   label=state_names_ru[state],
                   alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

        # Подписываем элементы
        for i, (ie, vol, sym) in enumerate(zip(ionization_energies, volumes, symbols)):
            offset_angle = (i * 30) % 360
            offset_r = 8
            offset_x = offset_r * np.cos(np.radians(offset_angle))
            offset_y = offset_r * np.sin(np.radians(offset_angle))

            ax2.annotate(sym, (ie, vol),
                        xytext=(offset_x, offset_y),
                        textcoords='offset points',
                        fontsize=6,
                        ha='center',
                        va='center',
                        alpha=0.8)

    ax2.set_xlabel('Энергия ионизации (эВ)', fontsize=12)
    ax2.set_ylabel('Атомный объём (м³)', fontsize=12)
    ax2.set_title('Зависимость объёма от энергии ионизации\n(логарифмическая шкала)', fontsize=14)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('volume_vs_ionization_energy.png', dpi=300, bbox_inches='tight')
    print("Графики объем vs энергия ионизации сохранены как 'volume_vs_ionization_energy.png'")

    # Статистика по агрегатным состояниям
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ПО АГРЕГАТНЫМ СОСТОЯНИЯМ")
    print("=" * 60)

    for state in ['solid', 'liquid', 'gas', 'unknown']:
        if data_by_state[state]:
            count = len(data_by_state[state])
            symbols = [e['symbol'] for e in data_by_state[state]]
            print(f"\n{state_names_ru[state]} ({count}):")
            print(f"  Элементы: {', '.join(symbols)}")

    plt.show()


def main():
    print("=" * 70)
    print("РАСЧЕТ И АНАЛИЗ КОЭФФИЦИЕНТОВ ЭФИРНОГО СЦЕПЛЕНИЯ")
    print("=" * 70)

    # Попытка загрузить данные из кэша
    elements_data = load_cache()

    if elements_data:
        print(f"\nДанные загружены из кэша '{CACHE_FILE}'")
        print(f"Загружено элементов: {len(elements_data)}")
        valid_elements = len([e for e in elements_data if e.get('k_coefficient') is not None])
    else:
        print("\nКэш не найден. Собираем данные для всех элементов...")
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

        # Сохраняем в кэш
        save_cache(elements_data)

    if valid_elements > 0:
        # Сохраняем в SQLite базу данных
        save_to_database(elements_data)

        # Анализ зависимостей
        params = analyze_dependencies(elements_data)

        if params:
            # Построение графиков со всеми подписями
            plot_all_elements_with_labels(params)

            # Построение дополнительных графиков с дефектом массы
            plot_mass_defect_dependencies(params['valid_data'])

            # Построение графиков объем vs энергия ионизации
            plot_volume_vs_ionization_energy(params['valid_data'])

    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 70)


if __name__ == "__main__":
    main()
