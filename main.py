from lattice import nearest_atoms
from transitions import transition_matrix


def main():
    # Пример параметров
    a = 4.75  # постоянная решетки (Å)
    R_bohr = 0.53  # один радиус Бора (Å)
    interaction_radius = 5 * R_bohr  # радиус взаимодействия (5 радиусов Бора)

    alpha, beta = 0.5, 1.0  # радиальный и азимутальный углы

    # 1. Работа с решеткой
    atoms = nearest_atoms(a, interaction_radius, alpha, beta)
    print(f"Ближайшие атомы (в пределах {interaction_radius:.2f} Å):")
    for atom, dist in atoms[:5]:
        print(atom, "->", dist)

    # 2. Работа с матрицами переходов
    D, D_inv = transition_matrix(L=1)
    print("\nМатрица D:\n", D)
    if D_inv is not None:
        print("\nОбратная матрица D^-1:\n", D_inv)
    else:
        print("\n⚠️ Обратная матрица не существует (матрица D вырожденная).")


if __name__ == "__main__":
    main()
