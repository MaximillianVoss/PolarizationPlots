# main.py
import tkinter as tk
from tkinter import ttk
import numpy as np

from lattice import nearest_atoms
from transitions import transition_matrices
# ▼ Новое:
from polarization_part2 import compute_grid

# ▼ Новое: матплотлиб в Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crystal Transitions GUI (Tkinter)")
        self.geometry("1100x720")

        # ====== ЛЕВАЯ ПАНЕЛЬ: существующие параметры (решётка/матрицы) ======
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        self.a = tk.DoubleVar(value=4.75)
        self.R_bohr = tk.DoubleVar(value=0.53)
        self.alpha = tk.DoubleVar(value=0.5)
        self.beta = tk.DoubleVar(value=1.0)
        self.lattice_radius = tk.IntVar(value=3)

        self._make_slider(left, "Постоянная решётки a (Å)", self.a, 1, 10, 0,
                           description="Расстояние между узлами решётки")
        self._make_slider(left, "Радиус Бора R_bohr (Å)", self.R_bohr, 0.1, 2.0, 1,
                           description="Радиус взаимодействия (×5 для поиска атомов)")
        self._make_slider(left, "Полярный угол α (рад)", self.alpha, 0, np.pi, 2,
                           description="Угол между направлением электрона и осью z")
        self._make_slider(left, "Азимутальный угол β (рад)", self.beta, 0, np.pi, 3,
                           description="Угол разворота направления вокруг оси z")
        self._make_slider(left, "Размер решётки n", self.lattice_radius, 1, 8, 4,
                           description="Число периодов по каждой оси", resolution=1)

        btn = ttk.Button(left, text="Пересчитать (решётка/матрицы)", command=self.update_output_left)
        btn.grid(row=5, column=0, columnspan=4, pady=5, sticky="ew")

        self.output = tk.Text(left, width=70, height=24, font=("Consolas", 10), wrap="word")
        self.output.grid(row=6, column=0, columnspan=4, pady=5, sticky="nsew")
        left.grid_rowconfigure(6, weight=1)

        # ====== ПРАВАЯ ПАНЕЛЬ: 'Часть 2' (интегралы и график) ======
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        # Параметры части 2
        self.Z = tk.DoubleVar(value=29.0)
        self.b = tk.DoubleVar(value=0.53)
        self.c1 = tk.DoubleVar(value=1.0)
        self.c2 = tk.DoubleVar(value=1.0)
        self.dr = tk.DoubleVar(value=0.01)
        self.rmax = tk.DoubleVar(value=15.0)
        self.Emin = tk.DoubleVar(value=10.0)
        self.Emax = tk.DoubleVar(value=100000.0)
        self.Npts = tk.IntVar(value=160)
        self.auto = tk.BooleanVar(value=True)

        row = 0
        self._make_slider(right, "Z (заряд ядра)", self.Z, 1, 92, row); row += 1
        self._make_slider(right, "b (Å)", self.b, 0.1, 2.0, row); row += 1
        self._make_slider(right, "c1", self.c1, 0.1, 3.0, row); row += 1
        self._make_slider(right, "c2", self.c2, 0.1, 3.0, row); row += 1
        self._make_slider(right, "dr (Å)", self.dr, 0.001, 0.1, row); row += 1
        self._make_slider(right, "r_max (Å)", self.rmax, 5.0, 30.0, row); row += 1
        self._make_slider(right, "Emin (эВ)", self.Emin, 1.0, 1000.0, row); row += 1
        self._make_slider(right, "Emax (эВ)", self.Emax, 1000.0, 200000.0, row); row += 1
        self._make_slider(right, "N точек", self.Npts, 20, 400, row, resolution=1); row += 1

        ttk.Checkbutton(right, text="Автопересчёт при движении ползунков",
                        variable=self.auto).grid(row=row, column=0, columnspan=3, sticky="w"); row += 1

        ttk.Button(right, text="Пересчитать (Часть 2)", command=self.update_output_right)\
            .grid(row=row, column=0, columnspan=3, sticky="ew", pady=4); row += 1

        # Площадка для графика
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xscale("log"); self.ax.set_yscale("log")
        self.ax.set_xlabel("Энергия, эВ"); self.ax.set_ylabel("|I_total| (усл. ед.)")
        self.ax.grid(True, which="both")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=row, column=0, columnspan=3, sticky="nsew")
        right.grid_rowconfigure(row, weight=1)

        # Первичный вывод
        self.update_output_left()
        self.update_output_right()

    def _make_slider(self, parent, label, var, mn, mx, row, description="", resolution=0.01):
        frame = ttk.Frame(parent); frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        frame.columnconfigure(1, weight=1)
        label_frame = ttk.Frame(frame); label_frame.grid(row=0, column=0, sticky="w")
        ttk.Label(label_frame, text=label).grid(row=0, column=0, sticky="w")
        if description:
            ttk.Label(label_frame, text=f"({description})", foreground="#555").grid(row=1, column=0, sticky="w")

        def on_change(_):
            # автопересчёт графика части 2
            if hasattr(self, "auto") and self.auto.get():
                self.update_output_right()

        sld = ttk.Scale(frame, from_=mn, to=mx, orient="horizontal", variable=var, command=on_change)
        sld.grid(row=0, column=1, sticky="ew", rowspan=2, padx=8)

        def format_value(value):
            if isinstance(var, tk.IntVar) or resolution >= 1:
                return f"{int(round(float(value)))}"
            return f"{float(value):.3g}"

        val_lbl = ttk.Label(frame, text=format_value(var.get()))
        val_lbl.grid(row=0, column=2, rowspan=2, sticky="e")
        var.trace_add("write", lambda *args, v=var, l=val_lbl: l.config(text=format_value(v.get())))

    # ====== ЛЕВАЯ ПАНЕЛЬ ======
    def update_output_left(self):
        a = self.a.get()
        R_bohr = self.R_bohr.get()
        alpha = self.alpha.get()
        beta = self.beta.get()
        interaction_radius = 5 * R_bohr
        n = self.lattice_radius.get()

        atoms = nearest_atoms(a, interaction_radius, alpha, beta, n=n)
        matrices, inverses = transition_matrices(L_source=1)

        txt = (f"Ближайшие атомы (расстояние до прямой ≤ {interaction_radius:.2f} Å):\n"
               f"Всего найдено: {len(atoms)}\n")
        preview = atoms[:10]
        for item in preview:
            coord = np.array2string(item["coords"], precision=2, suppress_small=True)
            txt += (f"{coord} -> d_прямой={item['distance_to_line']:.2f} Å, "
                    f"d_исток={item['distance_to_origin']:.2f} Å, "
                    f"s={item['longitudinal_distance']:.2f} Å\n")
        if len(atoms) > len(preview):
            txt += "...\n"

        txt += "\nМатрицы переходов D (L_s = 1):\n"
        for Ln, D in matrices.items():
            txt += f"\nLn = {Ln}:\n{np.array2string(D, precision=4, suppress_small=True)}"
            inv = inverses[Ln]
            if inv is not None:
                txt += "\nD^-1:\n" + np.array2string(inv, precision=4, suppress_small=True)
            else:
                txt += "\n⚠️ Обратная матрица не существует."
            txt += "\n"

        self.output.delete(1.0, tk.END)
        self.output.insert(tk.END, txt)

    # ====== ПРАВАЯ ПАНЕЛЬ (Часть 2) ======
    def update_output_right(self):
        try:
            df = compute_grid(
                self.Emin.get(), self.Emax.get(), int(self.Npts.get()),
                Z=self.Z.get(), a_ang=self.a.get(), b_ang=self.b.get(),
                c1=self.c1.get(), c2=self.c2.get(),
                dr_ang=self.dr.get(), r_max_ang=self.rmax.get(),
            )
        except Exception as ex:
            # Покажем ошибку на графике
            self.ax.clear()
            self.ax.text(0.05, 0.95, f"Ошибка: {ex}", transform=self.ax.transAxes, va="top", ha="left")
            self.canvas.draw()
            return

        # Обновить график |I_total|(E)
        self.ax.clear()
        self.ax.set_xscale("log"); self.ax.set_yscale("log")
        self.ax.set_xlabel("Энергия, эВ"); self.ax.set_ylabel("|I_total| (усл. ед.)")
        self.ax.grid(True, which="both")
        self.ax.plot(df["E_eV"].values, np.abs(df["I_total"].values))
        self.fig.tight_layout()
        self.canvas.draw()

        # Небольшая сводка внизу левого Text (чтобы точка привязки к форме не терялась)
        self.output.insert(tk.END,
            f"\n[Часть 2] E∈[{self.Emin.get():.3g},{self.Emax.get():.3g}] эВ, "
            f"N={int(self.Npts.get())}, Z={self.Z.get():.3g}, a={self.a.get():.3g} Å, b={self.b.get():.3g} Å\n"
        )


if __name__ == "__main__":
    app = App()
    app.mainloop()
