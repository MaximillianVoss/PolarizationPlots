import numpy as np


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


def nearest_atoms(a, interaction_radius, alpha, beta, n=3):
    """Возвращает атомы в пределах interaction_radius от прямой движения."""
    origin = np.zeros(3)
    dir_vec = spherical_to_cartesian(alpha, beta)
    dir_norm = dir_vec / np.linalg.norm(dir_vec)
    atoms = generate_lattice(a, n=n)

    results = []
    for atom in atoms:
        if np.allclose(atom, origin):
            continue

        displacement = atom - origin
        longitudinal = np.dot(displacement, dir_norm)
        if longitudinal < 0:
            continue

        dist_to_line = distance_point_to_line(atom, origin, dir_vec)
        if dist_to_line <= interaction_radius:
            distance_to_origin = np.linalg.norm(displacement)
            results.append(
                {
                    "coords": atom,
                    "distance_to_line": dist_to_line,
                    "distance_to_origin": distance_to_origin,
                    "longitudinal_distance": longitudinal,
                }
            )

    results.sort(key=lambda item: item["longitudinal_distance"])
    return results
