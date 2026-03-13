import numpy as np
import logging
logger = logging.getLogger(__name__)


def generate_lattice(a, n=3):
    """Генерация атомов в кубе решетки (n - число периодов в каждую сторону)."""
    atoms = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                atoms.append(np.array([i * a, j * a, k * a]))
    return np.array(atoms)


def spherical_to_cartesian(alpha, beta):
    """Преобразование углов в вектор направления."""
    x = np.sin(alpha) * np.cos(beta)
    y = np.sin(alpha) * np.sin(beta)
    z = np.cos(alpha)
    return np.array([x, y, z])


def distance_point_to_line(point, line_point, line_dir):
    """Расстояние от точки до прямой."""
    return np.linalg.norm(np.cross(point - line_point, line_dir)) / np.linalg.norm(line_dir)


def nearest_atoms(a, interaction_radius, alpha, beta, n=3, d_layer=0, max_longitudinal=None):
    """
    Возвращает атомы в пределах interaction_radius от луча движения.

    Параметры:
    ----------
    a : float
        Постоянная решётки (Å).
    interaction_radius : float
        Радиус взаимодействия вокруг траектории (Å).
    alpha, beta : float
        Углы направления траектории в радианах.
    n : int
        Размер куба решётки: [-n..n]^3.
    d_layer : int
        Слой источника по оси z. Старт электрона: (0, 0, d_layer * a).
    max_longitudinal : float | None
        Максимальная длина вдоль траектории, до которой учитываем атомы.
        Если None, берём весь луч внутри построенного куба.
    """
    logger.info(
        "lattice.nearest_atoms | a=%.4f, R_int=%.4f, alpha=%.4f, beta=%.4f, n=%d, d=%d, s_max=%s",
        a, interaction_radius, alpha, beta, n, d_layer,
        "None" if max_longitudinal is None else f"{max_longitudinal:.4f}"
    )

    origin = np.array([0.0, 0.0, float(d_layer) * a], dtype=float)

    dir_vec = spherical_to_cartesian(alpha, beta)
    dir_norm = np.linalg.norm(dir_vec)
    if dir_norm <= 1e-15:
        raise ValueError("Направляющий вектор траектории имеет почти нулевую длину.")

    dir_unit = dir_vec / dir_norm
    atoms = generate_lattice(a, n=n)

    results = []
    for atom in atoms:
        # пропускаем атом в точке старта, если совпал
        if np.allclose(atom, origin):
            continue

        displacement = atom - origin
        longitudinal = float(np.dot(displacement, dir_unit))

        # оставляем только атомы впереди по направлению движения
        if longitudinal < 0.0:
            continue

        # если задано ограничение по длине пролёта — учитываем
        if max_longitudinal is not None and longitudinal > max_longitudinal:
            continue

        # расстояние от атома до луча
        dist_to_line = np.linalg.norm(np.cross(displacement, dir_unit))

        if dist_to_line <= interaction_radius:
            results.append(
                {
                    "coords": atom,
                    "distance_to_line": float(dist_to_line),
                    "distance_to_origin": float(np.linalg.norm(displacement)),
                    "longitudinal_distance": longitudinal,
                }
            )

    results.sort(key=lambda item: item["longitudinal_distance"])

    logger.info(
        "lattice.nearest_atoms | origin=%s, dir=%s, всего узлов=%d, отобрано=%d",
        np.array2string(origin, precision=4, suppress_small=True),
        np.array2string(dir_unit, precision=4, suppress_small=True),
        len(atoms), len(results)
    )

    return results


# lattice.py
import math

def compute_lattice_n_auto(a_ang: float, R_bohr: float,
                           alpha: float, beta: float,
                           d_layer: int,
                           margin_bohr: float = 5.0,
                           max_atoms: int = 100_000) -> int:
    """
    Возвращает минимальный n для куба [-n..n]^3, который накрывает
    путь электрона от (0,0,0) до плоскости z = d*a + 5*R_bohr,
    с запасом ±5*R_bohr по X и Y. Ограничивает (2n+1)^3 <= max_atoms.
    """
    vx = math.sin(alpha) * math.cos(beta)
    vy = math.sin(alpha) * math.sin(beta)
    vz = math.cos(alpha) if abs(math.cos(alpha)) > 1e-8 else 1e-8  # избегаем деления на 0

    z_end = d_layer * a_ang + margin_bohr * R_bohr
    t_end = z_end / vz
    x_end = abs(t_end * vx)
    y_end = abs(t_end * vy)

    Lx = x_end + margin_bohr * R_bohr
    Ly = y_end + margin_bohr * R_bohr
    Lz = z_end

    nx = math.ceil(Lx / a_ang)
    ny = math.ceil(Ly / a_ang)
    nz = math.ceil(Lz / a_ang)
    n  = max(nx, ny, nz)

    def nodes(nn: int) -> int:
        side = 2*nn + 1
        return side*side*side

    # держим в лимите по количеству узлов
    while nodes(n) > max_atoms and n > nz:
        n -= 1
    while nodes(n) > max_atoms and n > 1:
        n -= 1
    return max(1, int(n))

