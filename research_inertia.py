import sqlite3
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from mendeleev import element
import csv
import json
import os
import time
from datetime import datetime


def load_or_calculate_data(force_reload=False):
    """Загружает данные из JSON или рассчитывает заново"""
    cache_file = 'atomic_data_cache.json'

    # Если есть кэш и не принудительная перезагрузка
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

    # Сохраняем в кэш
    cache_data = {
        'cache_version': 2,
        'created_at': datetime.now().isoformat(),
        'elements_count': valid_elements,
        'elements': elements_data
    }

    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        print(f"\nДанные сохранены в кэш: {cache_file}")
    except Exception as e:
        print(f"Ошибка сохранения кэша: {e}")

    return elements_data


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


def plot_all_elements_with_labels(params):
    """Строит графики со всеми подписанными элементами"""
    analysis_data = params.get('analysis_data', [])
    if not analysis_data:
        print("Нет данных для построения графиков")
        return

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

    # Создаем фигуру с адаптивным размером
    fig = plt.figure(figsize=(18, 14))

    # 1. Основной график: k vs Z_eff
    ax1 = plt.subplot(2, 2, 1)
    scatter1 = ax1.loglog(zeff_f, k_f, 'o', alpha=0.3, markersize=8)

    # Подписываем элементы с интеллектуальным позиционированием
    for i, (z, k, sym) in enumerate(zip(zeff_f, k_f, symbols_f)):
        # Более контролируемые смещения
        angle = (i * 137.5) % 360  # Используем золотой угол для распределения
        distance = 0.04  # 4% от размера точки

        # Конвертируем полярные координаты в смещения
        offset_x = distance * z * math.cos(math.radians(angle))
        offset_y = distance * k * math.sin(math.radians(angle))

        # Используем textcoords='data' для привязки к данным
        ax1.text(z, k, sym,
                 fontsize=7,
                 ha='center',
                 va='center',
                 alpha=0.8,
                 transform=ax1.transData)

    ax1.set_xlabel('Z_eff (эффективный заряд ядра)', fontsize=12)
    ax1.set_ylabel('Коэффициент k (кг/м³)', fontsize=12)
    ax1.set_title('Зависимость k от Z_eff\n(все элементы подписаны)', fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3)

    # 2. k vs Z_eff/V
    ax2 = plt.subplot(2, 2, 2)

    # Собираем данные для этого графика
    zv_data = []
    for d in filtered_data:
        if d['zeff_over_v'] is not None and d['zeff_over_v'] > 0:
            zv_data.append((d['zeff_over_v'], d['k'], d['symbol']))

    if zv_data:
        zv_vals, k_zv, sym_zv = zip(*zv_data)
        ax2.loglog(zv_vals, k_zv, 'o', alpha=0.3, markersize=8)

        for i, (zv, k, sym) in enumerate(zip(zv_vals, k_zv, sym_zv)):
            angle = (i * 137.5) % 360
            distance = 0.04

            offset_x = distance * zv * math.cos(math.radians(angle))
            offset_y = distance * k * math.sin(math.radians(angle))

            ax2.text(zv, k, sym,
                     fontsize=7,
                     ha='center',
                     va='center',
                     alpha=0.8,
                     transform=ax2.transData)

    ax2.set_xlabel('Z_eff / V (м⁻³)', fontsize=12)
    ax2.set_ylabel('Коэффициент k (кг/м³)', fontsize=12)
    ax2.set_title('k vs Z_eff/V\n(все элементы подписаны)', fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3)

    # 3. k vs Z
    ax3 = plt.subplot(2, 2, 3)
    valid_data = params.get('valid_data', [])

    # Собираем данные для графика Z vs k
    zk_data = []
    for e in valid_data:
        if e.get('k_coefficient') and e.get('atomic_number'):
            zk_data.append((e['atomic_number'], e['k_coefficient'], e['symbol']))

    if zk_data:
        z_vals, k_z, sym_z = zip(*zk_data)
        ax3.loglog(z_vals, k_z, 'o', alpha=0.3, markersize=8)

        for i, (z, k, sym) in enumerate(zip(z_vals, k_z, sym_z)):
            # Для графика Z vs k используем фиксированные смещения
            offset_x = 0.3 if i % 2 == 0 else -0.3
            offset_y = 0.03 * k if i % 3 == 0 else -0.03 * k

            ax3.text(z + offset_x, k + offset_y, sym,
                     fontsize=7,
                     ha='center',
                     va='center',
                     alpha=0.8)

    ax3.set_xlabel('Атомный номер Z', fontsize=12)
    ax3.set_ylabel('Коэффициент k (кг/м³)', fontsize=12)
    ax3.set_title('Зависимость k от Z\n(все элементы подписаны)', fontsize=14, pad=20)
    ax3.grid(True, alpha=0.3)

    # 4. График по блокам
    ax4 = plt.subplot(2, 2, 4)

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

            # Подписываем элементы в блоке
            for i, (z, k, sym) in enumerate(zip(zeff_block, k_block, symbols_block)):
                angle = (i * 137.5) % 360
                distance = 0.05

                offset_x = distance * z * math.cos(math.radians(angle))
                offset_y = distance * k * math.sin(math.radians(angle))

                ax4.text(z, k, sym,
                         fontsize=6,
                         ha='center',
                         va='center',
                         alpha=0.7,
                         color=colors[block])

    ax4.set_xlabel('Z_eff', fontsize=12)
    ax4.set_ylabel('k (кг/м³)', fontsize=12)
    ax4.set_title('k vs Z_eff по типам элементов\n(все элементы подписаны)', fontsize=14, pad=20)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')

    # Настраиваем layout с учетом подписей
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08,
                        wspace=0.25, hspace=0.3)

    # Сохраняем с правильными параметрами
    plt.savefig('all_elements_analysis.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    print("\nОсновной график сохранен как 'all_elements_analysis.png'")

    # Дополнительный график: гистограмма распределения k
    fig2, ax5 = plt.subplots(figsize=(12, 8))

    k_values = [e['k_coefficient'] for e in valid_data if e.get('k_coefficient')]

    if k_values:
        # Используем логарифмические бины для гистограммы
        log_k = np.log10(k_values)
        bins = np.logspace(np.log10(min(k_values)), np.log10(max(k_values)), 30)

        ax5.hist(k_values, bins=bins, edgecolor='black', alpha=0.7)
        ax5.set_xscale('log')
        ax5.set_xlabel('Коэффициент k (кг/м³)', fontsize=12)
        ax5.set_ylabel('Количество элементов', fontsize=12)
        ax5.set_title('Распределение коэффициентов k (логарифмическая шкала)', fontsize=14)
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
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax5.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('k_distribution.png', dpi=300, bbox_inches='tight')
        print("Гистограмма сохранена как 'k_distribution.png'")

    plt.show()


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


def plot_optimized_labels(params):
    """Строит графики с оптимизированным размещением подписей"""
    analysis_data = params.get('analysis_data', [])
    valid_data = params.get('valid_data', [])

    if not analysis_data:
        return

    # Подготовка данных
    symbols = [d['symbol'] for d in analysis_data]
    zeff_vals = [d['zeff'] for d in analysis_data]
    k_vals = [d['k'] for d in analysis_data]

    # Создаем большой график с тремя панелями
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))

    # 1. График k vs Z_eff с умным позиционированием
    ax1.loglog(zeff_vals, k_vals, 'o', alpha=0.5, markersize=10, color='blue')

    # Группируем близкие точки и подписываем только по одной из группы
    points = list(zip(zeff_vals, k_vals, symbols))
    points.sort(key=lambda x: x[0])  # Сортируем по Z_eff

    labeled_points = set()
    cluster_threshold = 0.05  # 5% разницы для кластеризации

    for i, (x, y, sym) in enumerate(points):
        # Проверяем, не находимся ли мы в уже размеченном кластере
        in_cluster = False
        for labeled in labeled_points:
            lx, ly, lsym = points[labeled]
            if abs(x - lx) / lx < cluster_threshold and abs(y - ly) / ly < cluster_threshold:
                in_cluster = True
                break

        if not in_cluster:
            # Подписываем эту точку
            offset_x = x * 0.02
            offset_y = y * 0.02

            # Чередуем направление смещения
            if i % 4 == 0:
                offset_x, offset_y = offset_x, offset_y
            elif i % 4 == 1:
                offset_x, offset_y = -offset_x, offset_y
            elif i % 4 == 2:
                offset_x, offset_y = offset_x, -offset_y
            else:
                offset_x, offset_y = -offset_x, -offset_y

            ax1.text(x + offset_x, y + offset_y, sym,
                     fontsize=8,
                     ha='center',
                     va='center',
                     alpha=0.9,
                     bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
            labeled_points.add(i)

    ax1.set_xlabel('Z_eff', fontsize=12)
    ax1.set_ylabel('k (кг/м³)', fontsize=12)
    ax1.set_title('k vs Z_eff\n(умное позиционирование подписей)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 2. График k vs Z
    ax2.loglog([e['atomic_number'] for e in valid_data if e.get('k_coefficient')],
               [e['k_coefficient'] for e in valid_data if e.get('k_coefficient')],
               'o', alpha=0.5, markersize=10, color='green')

    # Подписываем каждый 3-й элемент для Z графика
    zk_points = [(e['atomic_number'], e['k_coefficient'], e['symbol'])
                 for e in valid_data if e.get('k_coefficient')]

    for i, (z, k, sym) in enumerate(zk_points):
        if i % 3 == 0:  # Подписываем каждый третий элемент
            ax2.text(z, k, sym,
                     fontsize=8,
                     ha='center',
                     va='center',
                     alpha=0.9,
                     bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))

    ax2.set_xlabel('Атомный номер Z', fontsize=12)
    ax2.set_ylabel('k (кг/м³)', fontsize=12)
    ax2.set_title('k vs Z\n(каждый 3-й элемент подписан)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 3. График по блокам с подписями только характерных элементов
    colors = {'s': '#FF6B6B', 'p': '#4ECDC4', 'd': '#45B7D1', 'f': '#96CEB4'}

    # Характерные элементы для каждого блока (для подписей)
    characteristic_elements = {
        's': ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
        'p': ['B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'Ga', 'Ge', 'As', 'Se', 'Br', 'In', 'Sn', 'Sb', 'Te',
              'I', 'Tl', 'Pb', 'Bi'],
        'd': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
              'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
              'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
        'f': ['Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
              'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    }

    for block in colors.keys():
        block_data = [(e['zeff'], e['k_coefficient'], e['symbol'])
                      for e in valid_data
                      if e.get('block') == block and e.get('zeff') and e.get('k_coefficient')]

        if block_data:
            zeff_b, k_b, sym_b = zip(*block_data)
            ax3.loglog(zeff_b, k_b, 'o', color=colors[block],
                       alpha=0.6, markersize=8, label=f'{block}-элементы')

            # Подписываем только характерные элементы
            for z, k, sym in zip(zeff_b, k_b, sym_b):
                if sym in characteristic_elements[block]:
                    ax3.text(z, k, sym,
                             fontsize=7,
                             ha='center',
                             va='center',
                             alpha=0.8,
                             color=colors[block],
                             bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.5))

    ax3.set_xlabel('Z_eff', fontsize=12)
    ax3.set_ylabel('k (кг/м³)', fontsize=12)
    ax3.set_title('k vs Z_eff по типам элементов\n(подписаны характерные элементы)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')

    plt.tight_layout()
    plt.savefig('optimized_labels.png', dpi=300, bbox_inches='tight')
    print("График с оптимизированными подписями сохранен как 'optimized_labels.png'")
    plt.show()


def main():
    print("=" * 70)
    print("АНАЛИЗ КОЭФФИЦИЕНТОВ ЭФИРНОГО СЦЕПЛЕНИЯ")
    print("=" * 70)

    # Загружаем данные (из кэша или рассчитываем)
    elements_data = load_or_calculate_data(force_reload=False)

    if not elements_data:
        print("Не удалось получить данные")
        return

    print(f"\nЗагружено данных: {len(elements_data)} элементов")

    # Анализ зависимостей
    params = analyze_dependencies(elements_data)

    if params:
        # Построение графиков с разными стратегиями подписей
        plot_all_elements_with_labels(params)
        plot_optimized_labels(params)  # Новая функция с оптимизированными подписями

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