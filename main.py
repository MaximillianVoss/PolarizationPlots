import tkinter as tk
from tkinter import ttk
import numpy as np

from lattice import nearest_atoms
from transitions import transition_matrices


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crystal Transitions GUI (Tkinter)")

        # Параметры по умолчанию
        self.a = tk.DoubleVar(value=4.75)
        self.R_bohr = tk.DoubleVar(value=0.53)
        self.alpha = tk.DoubleVar(value=0.5)
        self.beta = tk.DoubleVar(value=1.0)
        self.lattice_radius = tk.IntVar(value=3)

        # Слайдеры
        self._make_slider("Постоянная решётки a (Å)", self.a, 1, 10, 0,
                           description="Расстояние между узлами решётки")
        self._make_slider("Радиус Бора R_bohr (Å)", self.R_bohr, 0.1, 2.0, 1,
                           description="Радиус взаимодействия (×5 для поиска атомов)")
        self._make_slider("Полярный угол α (рад)", self.alpha, 0, np.pi, 2,
                           description="Угол между направлением электрона и осью z")
        self._make_slider("Азимутальный угол β (рад)", self.beta, 0, np.pi, 3,
                           description="Угол разворота направления вокруг оси z")
        self._make_slider("Размер решётки n", self.lattice_radius, 1, 8, 4,
                           description="Число периодов по каждой оси", resolution=1)

        # Кнопка обновления
        btn = ttk.Button(self, text="Пересчитать", command=self.update_output)
        btn.grid(row=5, column=0, columnspan=4, pady=5, sticky="ew")

        # Поле вывода
        self.output = tk.Text(self, width=80, height=25, font=("Consolas", 10), wrap="word")
        self.output.grid(row=6, column=0, columnspan=4, pady=5, sticky="nsew")
        self.grid_rowconfigure(6, weight=1)

        self.update_output()

    def _make_slider(self, label, var, mn, mx, row, description="", resolution=0.01):
        frame = ttk.Frame(self)
        frame.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(2, 0))
        frame.columnconfigure(1, weight=1)

        label_frame = ttk.Frame(frame)
        label_frame.grid(row=0, column=0, sticky="w")

        lbl = ttk.Label(label_frame, text=label)
        lbl.grid(row=0, column=0, sticky="w")

        if description:
            hint = ttk.Label(label_frame, text=f"({description})", foreground="#555")
            hint.grid(row=1, column=0, sticky="w")

        sld = ttk.Scale(frame, from_=mn, to=mx, orient="horizontal", variable=var,
                        command=lambda v: self.update_output())
        sld.grid(row=0, column=1, sticky="ew", rowspan=2, padx=8)

        def format_value(value):
            if isinstance(var, tk.IntVar) or resolution >= 1:
                return f"{int(round(float(value)))}"
            return f"{float(value):.2f}"

        val_lbl = ttk.Label(frame, text=format_value(var.get()))
        val_lbl.grid(row=0, column=2, rowspan=2, sticky="e")

        # сохраняем ссылку, чтобы потом обновлять
        var.trace_add("write", lambda *args, v=var, l=val_lbl: l.config(text=format_value(v.get())))

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_columnconfigure(3, weight=0)

    def update_output(self):
        a = self.a.get()
        R_bohr = self.R_bohr.get()
        alpha = self.alpha.get()
        beta = self.beta.get()
        interaction_radius = 5 * R_bohr

        n = self.lattice_radius.get()
        atoms = nearest_atoms(a, interaction_radius, alpha, beta, n=n)
        matrices, inverses = transition_matrices(L_source=1)

        txt = (
            f"Ближайшие атомы (расстояние до прямой ≤ {interaction_radius:.2f} Å):\n"
            f"Всего найдено: {len(atoms)}\n"
        )

        preview = atoms[:10]
        for item in preview:
            coord = np.array2string(item["coords"], precision=2, suppress_small=True)
            txt += (
                f"{coord} -> d_прямой={item['distance_to_line']:.2f} Å, "
                f"d_исток={item['distance_to_origin']:.2f} Å, "
                f"s={item['longitudinal_distance']:.2f} Å\n"
            )

        if len(atoms) > len(preview):
            txt += "...\n"

        txt += "\nМатрицы переходов D (L_s = 1):\n"
        for Ln, D in matrices.items():
            txt += f"\nLn = {Ln}:\n"
            txt += np.array2string(D, precision=4, suppress_small=True)
            inv = inverses[Ln]
            if inv is not None:
                txt += "\nD^-1:\n"
                txt += np.array2string(inv, precision=4, suppress_small=True)
            else:
                txt += "\n⚠️ Обратная матрица не существует."
            txt += "\n"

        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, txt)


if __name__ == "__main__":
    app = App()
    app.mainloop()
