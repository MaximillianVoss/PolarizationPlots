import numpy as np

def generate_lattice(a, n=3):
    """Генерация атомов в кубе решетки (n - число периодов в каждую сторону)."""
    atoms = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            for k in range(-n, n+1):
                atoms.append(np.array([i*a, j*a, k*a]))
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

def nearest_atoms(a, R_bohr, alpha, beta, d=5):
    origin = np.zeros(3)
    dir_vec = spherical_to_cartesian(alpha, beta)
    atoms = generate_lattice(a, n=3)

    results = []
    for atom in atoms:
        if np.allclose(atom, origin):
            continue
        dist = distance_point_to_line(atom, origin, dir_vec)
        if dist < d * R_bohr:
            results.append((atom, dist))

    results.sort(key=lambda x: np.dot(x[0]-origin, dir_vec))
    return results
