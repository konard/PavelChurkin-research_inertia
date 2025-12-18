import sqlite3
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display issues
import matplotlib.pyplot as plt
from scipy import stats
from mendeleev import element
import csv
import json
import os
import time
from datetime import datetime
import argparse


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_database(db_path='atomic_data.db'):
    """Инициализирует базу данных SQLite"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Создание таблицы элементов
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS elements (
        atomic_number INTEGER PRIMARY KEY,
        symbol TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        atomic_mass REAL,
        atomic_mass_kg REAL,
        atomic_radius_pm REAL,
        atomic_volume_m3 REAL,
        k_coefficient REAL,
        k_log10 REAL,
        period INTEGER,
        element_group INTEGER,
        zeff REAL,
        block TEXT,

        -- Additional properties
        density REAL,
        en_allen REAL,
        en_pauling REAL,
        electron_affinity REAL,
        vdw_radius REAL,
        covalent_radius_cordero REAL,
        metallic_radius REAL,
        atomic_volume REAL,

        -- Structural properties
        lattice_structure TEXT,
        lattice_constant REAL,

        -- Thermodynamic properties
        melting_point REAL,
        boiling_point REAL,
        specific_heat_capacity REAL,
        thermal_conductivity REAL,

        -- Electromagnetic properties
        electrical_resistivity REAL,
        magnetic_ordering TEXT,

        -- Computed derived parameters
        zeff_over_v REAL,
        zeff2_over_v REAL,
        k_over_zeff REAL,
        surface_area REAL,
        cross_section REAL,
        mass_density_atomic REAL,

        -- Mass defect calculations (nucleon-based)
        neutron_count INTEGER,
        proton_count INTEGER,
        nucleon_mass_sum REAL,
        mass_defect REAL,
        mass_rounding_difference REAL,

        -- Metadata
        calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Создание таблицы метаданных
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cache_metadata (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Создание индексов
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_k_coefficient ON elements(k_coefficient)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_zeff ON elements(zeff)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_block ON elements(block)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_period ON elements(period)')

    # Сохранение версии схемы
    cursor.execute('''
    INSERT OR REPLACE INTO cache_metadata (key, value, updated_at)
    VALUES ('schema_version', '2.0', CURRENT_TIMESTAMP)
    ''')

    conn.commit()
    conn.close()
    print(f"База данных инициализирована: {db_path}")


def save_element_to_db(element_data, db_path='atomic_data.db'):
    """Сохраняет данные элемента в базу данных"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Подготовка данных для вставки
    columns = list(element_data.keys())
    placeholders = ','.join(['?' for _ in columns])
    column_names = ','.join(columns)

    # Специальная обработка для поля 'group' -> 'element_group'
    if 'group' in element_data:
        element_data['element_group'] = element_data.pop('group')
        columns = list(element_data.keys())
        column_names = ','.join(columns)

    values = [element_data[col] for col in columns]

    try:
        cursor.execute(f'''
        INSERT OR REPLACE INTO elements ({column_names})
        VALUES ({placeholders})
        ''', values)
        conn.commit()
    except Exception as e:
        print(f"Ошибка сохранения элемента {element_data.get('symbol', '?')}: {e}")
    finally:
        conn.close()


def load_elements_from_db(db_path='atomic_data.db'):
    """Загружает все элементы из базы данных"""
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT * FROM elements ORDER BY atomic_number')
        rows = cursor.fetchall()

        elements_data = []
        for row in rows:
            element_dict = dict(row)
            # Преобразуем element_group обратно в group
            if 'element_group' in element_dict:
                element_dict['group'] = element_dict.pop('element_group')
            elements_data.append(element_dict)

        print(f"Загружено {len(elements_data)} элементов из базы данных")
        return elements_data
    except Exception as e:
        print(f"Ошибка загрузки из базы данных: {e}")
        return None
    finally:
        conn.close()


def get_db_metadata(db_path='atomic_data.db'):
    """Получает метаданные из базы данных"""
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute('SELECT key, value, updated_at FROM cache_metadata')
        metadata = {row[0]: {'value': row[1], 'updated_at': row[2]} for row in cursor.fetchall()}
        return metadata
    except:
        return None
    finally:
        conn.close()


# ============================================================================
# DATA LOADING AND CALCULATION
# ============================================================================

def load_or_calculate_data(force_reload=False, use_db=True, db_path='atomic_data.db'):
    """Загружает данные из базы данных, JSON или рассчитывает заново"""
    cache_file = 'atomic_data_cache.json'

    # Приоритет 1: Загрузка из базы данных
    if use_db and not force_reload:
        elements_data = load_elements_from_db(db_path)
        if elements_data:
            print(f"Данные успешно загружены из базы данных ({len(elements_data)} элементов)")
            return elements_data

    # Приоритет 2: Загрузка из JSON кэша
    if os.path.exists(cache_file) and not force_reload:
        print(f"Загрузка данных из кэша: {cache_file}")
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Проверяем версию кэша
            cache_version = data.get('cache_version', 1)
            if cache_version >= 1:
                print(f"Кэш загружен успешно (версия {cache_version})")
                print(f"Дата создания: {data.get('created_at', 'неизвестно')}")
                print(f"Элементов в кэше: {len(data.get('elements', []))}")
                return data['elements']
            else:
                print("Устаревшая версия кэша, пересчитываем...")
        except Exception as e:
            print(f"Ошибка загрузки кэша: {e}")
            print("Пересчитываем данные...")

    # Если нет кэша или нужен пересчет
    print("Расчет данных атомов...")
    elements_data = []
    valid_elements = 0

    for atomic_number in range(1, 119):
        try:
            elem = element(atomic_number)
            result = calculate_atomic_data(elem)

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

    # Сохраняем в базу данных
    if use_db:
        print("\nСохранение данных в базу данных...")
        init_database(db_path)
        for elem_data in elements_data:
            save_element_to_db(elem_data.copy(), db_path)
        print(f"Данные сохранены в базу данных: {db_path}")

    # Сохраняем в JSON кэш (для обратной совместимости)
    cache_data = {
        'cache_version': 2,
        'created_at': datetime.now().isoformat(),
        'elements_count': valid_elements,
        'elements': elements_data
    }

    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"Данные также сохранены в JSON кэш: {cache_file}")
    except Exception as e:
        print(f"Ошибка сохранения JSON кэша: {e}")

    return elements_data


def calculate_neutron_count(atomic_mass, proton_count):
    """Рассчитывает количество нейтронов округлением атомной массы"""
    return round(atomic_mass) - proton_count


def calculate_mass_rounding_difference(atomic_mass, proton_count):
    """Рассчитывает разницу округления атомной массы

    Отрицательная при округлении в меньшую сторону,
    положительная при округлении в большую сторону
    """
    rounded_mass = round(atomic_mass)
    return atomic_mass - rounded_mass


def calculate_atomic_data(elem):
    """Рассчитывает все данные для атома"""
    atomic_mass = elem.atomic_weight
    if atomic_mass is None:
        return None

    # Получаем радиус
    atomic_radius = get_atomic_radius(elem)
    if atomic_radius is None or atomic_radius <= 0:
        return None

    # Рассчитываем объем
    volume_m3 = calculate_atomic_volume(atomic_radius)
    if volume_m3 is None or volume_m3 <= 0:
        return None

    # Переводим массу в килограммы на атом
    mass_per_atom_kg = (atomic_mass * 1e-3) / 6.02214076e23

    # k = m / V
    k = mass_per_atom_kg / volume_m3
    if k <= 0:
        return None

    # Рассчитываем дополнительные параметры
    Z_eff = get_effective_nuclear_charge(elem)

    # Рассчитываем массу дефекта и связанные параметры
    proton_count = elem.atomic_number
    neutron_count = calculate_neutron_count(atomic_mass, proton_count)
    mass_rounding_diff = calculate_mass_rounding_difference(atomic_mass, proton_count)

    # Константы масс нуклонов в а.е.м.
    PROTON_MASS_AMU = 1.007276466812  # масса протона в а.е.м.
    NEUTRON_MASS_AMU = 1.00866491595  # масса нейтрона в а.е.м.

    # Сумма масс нуклонов
    nucleon_mass_sum = (proton_count * PROTON_MASS_AMU) + (neutron_count * NEUTRON_MASS_AMU)

    # Дефект массы: атомная масса минус сумма масс нуклонов
    mass_defect = atomic_mass - nucleon_mass_sum

    # Собираем все доступные данные
    atomic_data = {
        'atomic_number': elem.atomic_number,
        'symbol': elem.symbol,
        'name': elem.name,
        'atomic_mass': atomic_mass,
        'atomic_mass_kg': mass_per_atom_kg,
        'atomic_radius_pm': atomic_radius,
        'atomic_volume_m3': volume_m3,
        'k_coefficient': k,
        'k_log10': math.log10(k),
        'period': int(elem.period) if hasattr(elem, 'period') and elem.period else None,
        'group': int(elem.group_id) if hasattr(elem, 'group_id') and elem.group_id else None,
        'zeff': Z_eff,
        'block': str(elem.block) if hasattr(elem, 'block') and elem.block else None,

        # Дополнительные свойства
        'density': float(elem.density) if hasattr(elem, 'density') and elem.density else None,
        'en_allen': float(elem.en_allen) if hasattr(elem, 'en_allen') and elem.en_allen else None,
        'en_pauling': float(elem.en_pauling) if hasattr(elem, 'en_pauling') and elem.en_pauling else None,
        'electron_affinity': float(elem.electron_affinity) if hasattr(elem,
                                                                      'electron_affinity') and elem.electron_affinity else None,
        'vdw_radius': float(elem.vdw_radius) if hasattr(elem, 'vdw_radius') and elem.vdw_radius else None,
        'covalent_radius_cordero': float(elem.covalent_radius_cordero) if hasattr(elem,
                                                                                  'covalent_radius_cordero') and elem.covalent_radius_cordero else None,
        'metallic_radius': float(elem.metallic_radius) if hasattr(elem,
                                                                  'metallic_radius') and elem.metallic_radius else None,
        'atomic_volume': float(elem.atomic_volume) if hasattr(elem, 'atomic_volume') and elem.atomic_volume else None,

        # Структурные свойства
        'lattice_structure': str(elem.lattice_structure) if hasattr(elem,
                                                                    'lattice_structure') and elem.lattice_structure else None,
        'lattice_constant': float(elem.lattice_constant) if hasattr(elem,
                                                                    'lattice_constant') and elem.lattice_constant else None,

        # Термодинамические свойства
        'melting_point': float(elem.melting_point) if hasattr(elem, 'melting_point') and elem.melting_point else None,
        'boiling_point': float(elem.boiling_point) if hasattr(elem, 'boiling_point') and elem.boiling_point else None,
        'specific_heat_capacity': float(elem.specific_heat_capacity) if hasattr(elem,
                                                                                'specific_heat_capacity') and elem.specific_heat_capacity else None,
        'thermal_conductivity': float(elem.thermal_conductivity) if hasattr(elem,
                                                                            'thermal_conductivity') and elem.thermal_conductivity else None,

        # Электромагнитные свойства
        'electrical_resistivity': float(elem.electrical_resistivity) if hasattr(elem,
                                                                                'electrical_resistivity') and elem.electrical_resistivity else None,
        'magnetic_ordering': str(elem.magnetic_ordering) if hasattr(elem,
                                                                    'magnetic_ordering') and elem.magnetic_ordering else None,

        # Вычисляемые производные параметры
        'zeff_over_v': Z_eff / volume_m3 if volume_m3 != 0 else None,
        'zeff2_over_v': Z_eff ** 2 / volume_m3 if volume_m3 != 0 else None,
        'k_over_zeff': k / Z_eff if Z_eff != 0 else None,
        'surface_area': 4 * math.pi * (atomic_radius * 1e-12) ** 2,
        'cross_section': math.pi * (atomic_radius * 1e-12) ** 2,
        'mass_density_atomic': k,  # Это и есть k

        # Расчеты дефекта массы (нуклонные)
        'neutron_count': neutron_count,
        'proton_count': proton_count,
        'nucleon_mass_sum': nucleon_mass_sum,
        'mass_defect': mass_defect,
        'mass_rounding_difference': mass_rounding_diff,
    }

    return atomic_data


def get_atomic_radius(elem):
    """Получаем атомный радиус с приоритетом разных типов радиусов"""
    # Пробуем разные типы радиусов в порядке приоритета
    if hasattr(elem, 'atomic_radius') and elem.atomic_radius is not None:
        return float(elem.atomic_radius)

    # Ковалентные радиусы
    cov_radii = [
        elem.covalent_radius_cordero if hasattr(elem, 'covalent_radius_cordero') else None,
        elem.covalent_radius_pyykko if hasattr(elem, 'covalent_radius_pyykko') else None,
        elem.covalent_radius_pyykko_double if hasattr(elem, 'covalent_radius_pyykko_double') else None,
        elem.covalent_radius_pyykko_triple if hasattr(elem, 'covalent_radius_pyykko_triple') else None,
    ]

    for radius in cov_radii:
        if radius is not None:
            return float(radius)

    # Ван-дер-ваальсовы радиусы
    vdw_radii = [
        elem.vdw_radius if hasattr(elem, 'vdw_radius') else None,
        elem.vdw_radius_bondi if hasattr(elem, 'vdw_radius_bondi') else None,
        elem.vdw_radius_alvarez if hasattr(elem, 'vdw_radius_alvarez') else None,
    ]

    for radius in vdw_radii:
        if radius is not None:
            return float(radius)

    # Если нет радиуса, оценим через атомный объем
    if hasattr(elem, 'atomic_volume') and elem.atomic_volume is not None:
        volume_cm3_per_atom = elem.atomic_volume / 6.02214076e23
        volume_m3_per_atom = volume_cm3_per_atom * 1e-6
        r = (3 * volume_m3_per_atom / (4 * math.pi)) ** (1 / 3)
        radius_pm = r * 1e12
        return float(radius_pm)

    return None


def calculate_atomic_volume(radius_pm):
    """Рассчитывает объем атома как объем шара по радиусу"""
    if radius_pm is None or radius_pm <= 0:
        return None
    r_m = float(radius_pm) * 1e-12
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


def analyze_dependencies(elements_data):
    """Анализирует зависимости между k и другими параметрами"""
    # Фильтруем элементы с данными
    valid_data = [e for e in elements_data if e.get('k_coefficient') is not None]

    if len(valid_data) < 3:
        print("Недостаточно данных для анализа зависимостей")
        return None

    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВИСИМОСТЕЙ КОЭФФИЦИЕНТА СЦЕПЛЕНИЯ")
    print("=" * 70)

    # Подготавливаем данные для анализа
    analysis_data = []
    for e in valid_data:
        if e.get('zeff') is not None and e.get('atomic_volume_m3') is not None:
            analysis_data.append({
                'symbol': e['symbol'],
                'k': e['k_coefficient'],
                'zeff': e['zeff'],
                'volume': e['atomic_volume_m3'],
                'zeff_over_v': e.get('zeff_over_v'),
                'zeff2_over_v': e.get('zeff2_over_v'),
                'z': e['atomic_number'],
                'radius': e['atomic_radius_pm'],
                'ionization': e.get('en_allen'),
                'density': e.get('density')
            })

    def safe_correlation(x, y, log_scale=False):
        """Безопасный расчет корреляции"""
        # Фильтруем None значения
        filtered = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        if len(filtered) < 2:
            return 0, 0, 0, []

        x_vals, y_vals = zip(*filtered)

        if log_scale:
            # Фильтруем неположительные значения для логарифма
            log_filtered = [(a, b) for a, b in zip(x_vals, y_vals) if a > 0 and b > 0]
            if len(log_filtered) < 2:
                return 0, 0, 0, []

            x_log, y_log = zip(*log_filtered)
            try:
                corr = np.corrcoef(np.log10(x_log), np.log10(y_log))[0, 1]
                slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(x_log), np.log10(y_log))
                return corr, r_value ** 2, p_value, list(zip(x_log, y_log))
            except:
                return 0, 0, 0, list(zip(x_log, y_log))
        else:
            try:
                corr = np.corrcoef(x_vals, y_vals)[0, 1]
                if len(x_vals) > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                    return corr, r_value ** 2, p_value, list(zip(x_vals, y_vals))
                return corr, 0, 0, list(zip(x_vals, y_vals))
            except:
                return 0, 0, 0, list(zip(x_vals, y_vals))

    # 1. Основная гипотеза: k ∝ Z_eff^n / V
    print("\n" + "=" * 50)
    print("ГИПОТЕЗА: k ∝ Z_eff^n / V")
    print("=" * 50)

    if len(analysis_data) > 2:
        # k vs Z_eff
        zeff_vals = [d['zeff'] for d in analysis_data]
        k_vals = [d['k'] for d in analysis_data]

        corr, r2, pval, data_points = safe_correlation(zeff_vals, k_vals, log_scale=True)
        print(f"\nk vs Z_eff (логарифмическая шкала):")
        print(f"  Точек данных: {len(data_points)}")
        print(f"  Корреляция: {corr:.4f}")
        print(f"  R²: {r2:.4f}")

        # k vs Z_eff/V
        zeff_over_v = [d['zeff_over_v'] for d in analysis_data if d['zeff_over_v']]
        corr_zv, r2_zv, pval_zv, data_zv = safe_correlation(zeff_over_v, k_vals[:len(zeff_over_v)], log_scale=True)
        print(f"\nk vs Z_eff/V:")
        print(f"  Точек данных: {len(data_zv)}")
        print(f"  Корреляция: {corr_zv:.4f}")
        print(f"  R²: {r2_zv:.4f}")

        # k vs Z_eff²/V
        zeff2_over_v = [d['zeff2_over_v'] for d in analysis_data if d['zeff2_over_v']]
        corr_z2v, r2_z2v, pval_z2v, data_z2v = safe_correlation(zeff2_over_v, k_vals[:len(zeff2_over_v)],
                                                                log_scale=True)
        print(f"\nk vs Z_eff²/V:")
        print(f"  Точек данных: {len(data_z2v)}")
        print(f"  Корреляция: {corr_z2v:.4f}")
        print(f"  R²: {r2_z2v:.4f}")

    # 2. Другие корреляции
    print("\n" + "=" * 50)
    print("ДРУГИЕ КОРРЕЛЯЦИИ")
    print("=" * 50)

    # Z vs k
    z_all = [e['atomic_number'] for e in valid_data]
    k_all = [e['k_coefficient'] for e in valid_data]
    corr_z, r2_z, _, data_z = safe_correlation(z_all, k_all, log_scale=True)
    print(f"\nZ vs k:")
    print(f"  Точек данных: {len(data_z)}")
    print(f"  Корреляция: {corr_z:.4f}")
    print(f"  R²: {r2_z:.4f}")

    # Объем vs k
    volume_all = [e['atomic_volume_m3'] for e in valid_data]
    corr_v, r2_v, _, data_v = safe_correlation(volume_all, k_all, log_scale=True)
    print(f"\nV vs k:")
    print(f"  Точек данных: {len(data_v)}")
    print(f"  Корреляция: {corr_v:.4f}")
    print(f"  R²: {r2_v:.4f}")

    # Энергия ионизации vs k
    ionization_vals = [e.get('en_allen') for e in valid_data]
    corr_i, r2_i, _, data_i = safe_correlation(ionization_vals, k_all, log_scale=False)
    if len(data_i) > 0:
        print(f"\nЭнергия ионизации vs k:")
        print(f"  Точек данных: {len(data_i)}")
        print(f"  Корреляция: {corr_i:.4f}")
        print(f"  R²: {r2_i:.4f}")

    return {'analysis_data': analysis_data, 'valid_data': valid_data}


def plot_all_elements_with_labels(params, label_every_nth=1, reduced_dpi=150):
    """Строит графики с подписями ВСЕХ элементов (каждый график в отдельном окне)

    Args:
        params: Словарь с данными для анализа
        label_every_nth: Подписывать каждый N-й элемент (по умолчанию 1 = ВСЕ элементы)
        reduced_dpi: Разрешение изображения (по умолчанию 150)
    """
    # Проверяем наличие графиков в текущей директории
    graph_files = [
        'graph_1_k_vs_zeff.png',
        'graph_2_k_vs_zeff_over_v.png',
        'graph_3_k_vs_z.png',
        'graph_4_k_vs_zeff_by_blocks.png',
        'graph_5_k_distribution.png',
        'graph_6_k_coefficient_dependence.png',
        'graph_7_mass_defect_per_volume.png'
    ]

    # Проверяем наличие всех графиков
    all_graphs_exist = all(os.path.exists(f) for f in graph_files)

    if all_graphs_exist:
        print("Все графики уже существуют в текущей директории. Пропускаем генерацию.")
        return

    analysis_data = params.get('analysis_data', [])
    valid_data = params.get('valid_data', [])

    if not analysis_data:
        print("Нет данных для построения графиков")
        return

    print(f"Генерация графиков с метками ВСЕХ элементов, DPI={reduced_dpi}...")

    symbols = [d['symbol'] for d in analysis_data]
    zeff_vals = [d['zeff'] for d in analysis_data]
    k_vals = [d['k'] for d in analysis_data]
    zeff_over_v = [d.get('zeff_over_v') for d in analysis_data]
    zeff2_over_v = [d.get('zeff2_over_v') for d in analysis_data]

    # Фильтруем None значения и неположительные для логарифмирования
    filtered_data = []
    for i, (sym, zeff, k, zv, z2v) in enumerate(zip(symbols, zeff_vals, k_vals, zeff_over_v, zeff2_over_v)):
        if zeff is not None and k is not None and zeff > 0 and k > 0:
            filtered_data.append({
                'symbol': sym,
                'zeff': zeff,
                'k': k,
                'zeff_over_v': zv,
                'zeff2_over_v': z2v,
                'index': i
            })

    if not filtered_data:
        print("Нет валидных данных для построения графиков")
        return

    symbols_f = [d['symbol'] for d in filtered_data]
    zeff_f = [d['zeff'] for d in filtered_data]
    k_f = [d['k'] for d in filtered_data]

    # ============================================================================
    # ГРАФИК 1: k vs Z_eff (отдельное окно)
    # ============================================================================
    print("\n1. Создание графика: k vs Z_eff")
    fig1 = plt.figure(figsize=(16, 12))
    ax1 = fig1.add_subplot(111)
    ax1.loglog(zeff_f, k_f, 'o', alpha=0.3, markersize=8)

    # Подписываем ВСЕ элементы
    for i, (z, k, sym) in enumerate(zip(zeff_f, k_f, symbols_f)):
        if i % label_every_nth == 0:
            ax1.text(z, k, sym,
                     fontsize=8,
                     ha='center',
                     va='center',
                     alpha=0.8,
                     transform=ax1.transData)

    ax1.set_xlabel('Z_eff (эффективный заряд ядра)', fontsize=14)
    ax1.set_ylabel('Коэффициент k (кг/м³)', fontsize=14)
    ax1.set_title('Зависимость k от Z_eff (все элементы подписаны)', fontsize=16, pad=20)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('graph_1_k_vs_zeff.png', dpi=reduced_dpi, bbox_inches='tight')
    print(f"   Сохранен: graph_1_k_vs_zeff.png")
    plt.close(fig1)

    # ============================================================================
    # ГРАФИК 2: k vs Z_eff/V (отдельное окно)
    # ============================================================================
    print("2. Создание графика: k vs Z_eff/V")

    # Собираем данные для этого графика
    zv_data = []
    for d in filtered_data:
        if d['zeff_over_v'] is not None and d['zeff_over_v'] > 0:
            zv_data.append((d['zeff_over_v'], d['k'], d['symbol']))

    if zv_data:
        fig2 = plt.figure(figsize=(16, 12))
        ax2 = fig2.add_subplot(111)

        zv_vals, k_zv, sym_zv = zip(*zv_data)
        ax2.loglog(zv_vals, k_zv, 'o', alpha=0.3, markersize=8)

        # Подписываем ВСЕ элементы
        for i, (zv, k, sym) in enumerate(zip(zv_vals, k_zv, sym_zv)):
            if i % label_every_nth == 0:
                ax2.text(zv, k, sym,
                         fontsize=8,
                         ha='center',
                         va='center',
                         alpha=0.8,
                         transform=ax2.transData)

        ax2.set_xlabel('Z_eff / V (м⁻³)', fontsize=14)
        ax2.set_ylabel('Коэффициент k (кг/м³)', fontsize=14)
        ax2.set_title('k vs Z_eff/V (все элементы подписаны)', fontsize=16, pad=20)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('graph_2_k_vs_zeff_over_v.png', dpi=reduced_dpi, bbox_inches='tight')
        print(f"   Сохранен: graph_2_k_vs_zeff_over_v.png")
        plt.close(fig2)

    # ============================================================================
    # ГРАФИК 3: k vs Z (отдельное окно)
    # ============================================================================
    print("3. Создание графика: k vs Z")

    # Собираем данные для графика Z vs k
    zk_data = []
    for e in valid_data:
        if e.get('k_coefficient') and e.get('atomic_number'):
            zk_data.append((e['atomic_number'], e['k_coefficient'], e['symbol']))

    if zk_data:
        fig3 = plt.figure(figsize=(16, 12))
        ax3 = fig3.add_subplot(111)

        z_vals, k_z, sym_z = zip(*zk_data)
        ax3.loglog(z_vals, k_z, 'o', alpha=0.3, markersize=8)

        # Подписываем ВСЕ элементы
        for i, (z, k, sym) in enumerate(zip(z_vals, k_z, sym_z)):
            if i % label_every_nth == 0:
                ax3.text(z, k, sym,
                         fontsize=8,
                         ha='center',
                         va='center',
                         alpha=0.8)

        ax3.set_xlabel('Атомный номер Z', fontsize=14)
        ax3.set_ylabel('Коэффициент k (кг/м³)', fontsize=14)
        ax3.set_title('Зависимость k от Z (все элементы подписаны)', fontsize=16, pad=20)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('graph_3_k_vs_z.png', dpi=reduced_dpi, bbox_inches='tight')
        print(f"   Сохранен: graph_3_k_vs_z.png")
        plt.close(fig3)

    # ============================================================================
    # ГРАФИК 4: k vs Z_eff по типам элементов (отдельное окно)
    # ============================================================================
    print("4. Создание графика: k vs Z_eff по типам элементов")

    fig4 = plt.figure(figsize=(16, 12))
    ax4 = fig4.add_subplot(111)

    colors = {'s': '#FF6B6B', 'p': '#4ECDC4', 'd': '#45B7D1', 'f': '#96CEB4'}
    marker_size = 8

    # Собираем данные по блокам
    blocks_data = {'s': [], 'p': [], 'd': [], 'f': []}
    for e in valid_data:
        block = e.get('block')
        if block in blocks_data and e.get('zeff') and e.get('k_coefficient'):
            blocks_data[block].append({
                'zeff': e['zeff'],
                'k': e['k_coefficient'],
                'symbol': e['symbol']
            })

    # Рисуем каждый блок
    for block, data in blocks_data.items():
        if data:
            zeff_block = [d['zeff'] for d in data]
            k_block = [d['k'] for d in data]
            symbols_block = [d['symbol'] for d in data]

            ax4.loglog(zeff_block, k_block, 'o',
                       color=colors[block],
                       alpha=0.6,
                       markersize=marker_size,
                       label=f'{block}-элементы')

            # Подписываем ВСЕ элементы в блоке
            for i, (z, k, sym) in enumerate(zip(zeff_block, k_block, symbols_block)):
                if i % label_every_nth == 0:
                    ax4.text(z, k, sym,
                             fontsize=7,
                             ha='center',
                             va='center',
                             alpha=0.7,
                             color=colors[block])

    ax4.set_xlabel('Z_eff', fontsize=14)
    ax4.set_ylabel('k (кг/м³)', fontsize=14)
    ax4.set_title('k vs Z_eff по типам элементов (все элементы подписаны)', fontsize=16, pad=20)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=12)

    plt.tight_layout()
    plt.savefig('graph_4_k_vs_zeff_by_blocks.png', dpi=reduced_dpi, bbox_inches='tight')
    print(f"   Сохранен: graph_4_k_vs_zeff_by_blocks.png")
    plt.close(fig4)

    # ============================================================================
    # ГРАФИК 5: Распределение коэффициентов k (отдельное окно)
    # ============================================================================
    print("5. Создание графика: Распределение коэффициентов k")

    k_values = [e['k_coefficient'] for e in valid_data if e.get('k_coefficient')]

    if k_values:
        fig5 = plt.figure(figsize=(14, 10))
        ax5 = fig5.add_subplot(111)

        # Используем логарифмические бины для гистограммы
        log_k = np.log10(k_values)
        bins = np.logspace(np.log10(min(k_values)), np.log10(max(k_values)), 30)

        ax5.hist(k_values, bins=bins, edgecolor='black', alpha=0.7)
        ax5.set_xscale('log')
        ax5.set_xlabel('Коэффициент k (кг/м³)', fontsize=14)
        ax5.set_ylabel('Количество элементов', fontsize=14)
        ax5.set_title('Распределение коэффициентов k (логарифмическая шкала)', fontsize=16)
        ax5.grid(True, alpha=0.3, which='both')

        # Добавляем статистические линии
        mean_k = np.mean(k_values)
        median_k = np.median(k_values)

        ax5.axvline(mean_k, color='red', linestyle='--',
                    linewidth=2, label=f'Среднее: {mean_k:.2e} кг/м³')
        ax5.axvline(median_k, color='green', linestyle='-.',
                    linewidth=2, label=f'Медиана: {median_k:.2e} кг/м³')

        # Добавляем аннотации для интересных областей
        ax5.annotate(f'Всего элементов: {len(k_values)}',
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax5.legend(loc='upper right', fontsize=12)

        plt.tight_layout()
        plt.savefig('graph_5_k_distribution.png', dpi=reduced_dpi, bbox_inches='tight')
        print(f"   Сохранен: graph_5_k_distribution.png")
        plt.close(fig5)

    # ============================================================================
    # ГРАФИК 6: Зависимость коэффициента сцепления от атомного номера (отдельное окно)
    # ============================================================================
    print("6. Создание графика: Коэффициент сцепления vs Z")

    z_all = [e['atomic_number'] for e in valid_data if e.get('k_coefficient')]
    k_all = [e['k_coefficient'] for e in valid_data if e.get('k_coefficient')]
    symbols_all = [e['symbol'] for e in valid_data if e.get('k_coefficient')]

    if z_all and k_all:
        fig6 = plt.figure(figsize=(16, 12))
        ax6 = fig6.add_subplot(111)

        ax6.loglog(z_all, k_all, 'o', alpha=0.3, markersize=8)

        # Подписываем ВСЕ элементы
        for i, (z, k, sym) in enumerate(zip(z_all, k_all, symbols_all)):
            if i % label_every_nth == 0:
                ax6.text(z, k, sym,
                         fontsize=8,
                         ha='center',
                         va='center',
                         alpha=0.8)

        ax6.set_xlabel('Атомный номер Z', fontsize=14)
        ax6.set_ylabel('Коэффициент сцепления k (кг/м³)', fontsize=14)
        ax6.set_title('Зависимость коэффициента сцепления от атомного номера', fontsize=16, pad=20)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('graph_6_k_coefficient_dependence.png', dpi=reduced_dpi, bbox_inches='tight')
        print(f"   Сохранен: graph_6_k_coefficient_dependence.png")
        plt.close(fig6)

    # ============================================================================
    # ГРАФИК 7: Дефект массы / объём vs атомный номер (отдельное окно)
    # ============================================================================
    print("7. Создание графика: Дефект массы/объём vs Z")

    # Собираем данные для дефекта массы на объем
    mass_defect_per_volume_data = []
    for e in valid_data:
        if (e.get('mass_defect') is not None and
                e.get('atomic_volume_m3') is not None and
                e.get('atomic_volume_m3') > 0):
            # Дефект массы в кг
            mass_defect_kg = (e['mass_defect'] * 1e-3) / 6.02214076e23
            defect_per_volume = abs(mass_defect_kg / e['atomic_volume_m3'])
            mass_defect_per_volume_data.append({
                'z': e['atomic_number'],
                'defect_per_volume': defect_per_volume,
                'symbol': e['symbol']
            })

    if mass_defect_per_volume_data:
        fig7 = plt.figure(figsize=(16, 12))
        ax7 = fig7.add_subplot(111)

        z_vals = [d['z'] for d in mass_defect_per_volume_data]
        defect_vals = [d['defect_per_volume'] for d in mass_defect_per_volume_data]
        symbols_vals = [d['symbol'] for d in mass_defect_per_volume_data]

        ax7.loglog(z_vals, defect_vals, 'o', alpha=0.3, markersize=8, color='purple')

        # Подписываем ВСЕ элементы
        for i, (z, defect, sym) in enumerate(zip(z_vals, defect_vals, symbols_vals)):
            if i % label_every_nth == 0:
                ax7.text(z, defect, sym,
                         fontsize=8,
                         ha='center',
                         va='center',
                         alpha=0.8)

        ax7.set_xlabel('Атомный номер Z', fontsize=14)
        ax7.set_ylabel('|Дефект массы| / Объём (кг/м³)', fontsize=14)
        ax7.set_title('Зависимость дефекта массы на объём от атомного номера', fontsize=16, pad=20)
        ax7.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('graph_7_mass_defect_per_volume.png', dpi=reduced_dpi, bbox_inches='tight')
        print(f"   Сохранен: graph_7_mass_defect_per_volume.png")
        plt.close(fig7)

    print("\nВсе графики успешно созданы и сохранены!")


def export_data_to_csv(elements_data):
    """Экспортирует данные в CSV"""
    filename = f'atomic_data_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

    # Определяем все возможные поля
    all_fields = set()
    for elem in elements_data:
        all_fields.update(elem.keys())

    # Сортируем поля для удобства
    field_order = ['atomic_number', 'symbol', 'name', 'k_coefficient', 'zeff',
                   'atomic_volume_m3', 'atomic_radius_pm', 'period', 'group', 'block']
    other_fields = sorted([f for f in all_fields if f not in field_order])
    fieldnames = field_order + other_fields

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for elem in elements_data:
            # Конвертируем все значения в строки
            row = {}
            for field in fieldnames:
                value = elem.get(field)
                if value is None:
                    row[field] = ''
                elif isinstance(value, (int, float)):
                    row[field] = str(value)
                else:
                    row[field] = str(value)
            writer.writerow(row)

    print(f"\nДанные экспортированы в: {filename}")
    return filename




def main():
    """Главная функция с поддержкой аргументов командной строки"""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Анализ коэффициентов эфирного сцепления атомов')
    parser.add_argument('--force-reload', action='store_true',
                        help='Принудительный пересчет данных (игнорировать кэш и БД)')
    parser.add_argument('--no-db', action='store_true',
                        help='Не использовать базу данных (только JSON кэш)')
    parser.add_argument('--label-every', type=int, default=1,
                        help='Подписывать каждый N-й элемент на графиках (по умолчанию: 1 = все элементы)')
    parser.add_argument('--dpi', type=int, default=150,
                        help='Разрешение изображений (по умолчанию: 150)')
    parser.add_argument('--skip-plots', action='store_true',
                        help='Пропустить построение графиков')
    parser.add_argument('--db-path', type=str, default='atomic_data.db',
                        help='Путь к файлу базы данных (по умолчанию: atomic_data.db)')

    args = parser.parse_args()

    print("=" * 70)
    print("АНАЛИЗ КОЭФФИЦИЕНТОВ ЭФИРНОГО СЦЕПЛЕНИЯ")
    print("=" * 70)
    print(f"Настройки: label_every={args.label_every}, dpi={args.dpi}, use_db={not args.no_db}")

    # Загружаем данные (из кэша или рассчитываем)
    elements_data = load_or_calculate_data(
        force_reload=args.force_reload,
        use_db=not args.no_db,
        db_path=args.db_path
    )

    if not elements_data:
        print("Не удалось получить данные")
        return

    print(f"\nЗагружено данных: {len(elements_data)} элементов")

    # Анализ зависимостей
    params = analyze_dependencies(elements_data)

    if params and not args.skip_plots:
        # Построение графиков с настраиваемыми параметрами
        print("\n" + "=" * 70)
        print("ПОСТРОЕНИЕ ГРАФИКОВ (КАЖДЫЙ В ОТДЕЛЬНОМ ФАЙЛЕ)")
        print("=" * 70)
        plot_all_elements_with_labels(params, label_every_nth=args.label_every, reduced_dpi=args.dpi)

        # Экспорт данных
        export_data_to_csv(elements_data)

        # Статистика
        valid_elements = [e for e in elements_data if e.get('k_coefficient')]
        if valid_elements:
            k_values = [e['k_coefficient'] for e in valid_elements]
            print(f"\nСТАТИСТИКА КОЭФФИЦИЕНТОВ k:")
            print(f"  Всего элементов с k: {len(k_values)}")
            print(f"  Минимальный k: {min(k_values):.3e} кг/м³")
            print(f"  Максимальный k: {max(k_values):.3e} кг/м³")
            print(f"  Средний k: {np.mean(k_values):.3e} кг/м³")
            print(f"  Медиана k: {np.median(k_values):.3e} кг/м³")

    print("\n" + "=" * 70)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 70)


if __name__ == "__main__":
    main()