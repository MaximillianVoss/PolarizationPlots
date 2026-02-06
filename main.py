# main.py
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import numpy as np
import os
from datetime import datetime

from lattice import nearest_atoms, compute_lattice_n_auto
from transitions import transition_matrices
from polarization_part2 import (
    compute_grid,
    chi_table_interp,
    chi_default,
    spin_amplitudes_both,  # обе подготовки спина
)

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import logging

logging.basicConfig(
    filename="log.txt",
    filemode="a",  # дописываем
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crystal Transitions GUI (Tkinter)")
        self.geometry("1180x800")

        # ====== ЛЕВАЯ ПАНЕЛЬ: решётка/матрицы ======
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        # Параметры решётки/геометрии
        self.a = tk.DoubleVar(value=4.75)
        self.R_bohr = tk.DoubleVar(value=0.53)
        self.alpha = tk.DoubleVar(value=0.5)
        self.beta = tk.DoubleVar(value=1.0)
        self.lattice_radius = tk.IntVar(value=3)   # n
        self.d_layer = tk.IntVar(value=0)          # слой источника d
        self.auto_n = tk.BooleanVar(value=True)    # автоподбор n

        row = 0
        self._make_slider(left, "Постоянная решётки a (Å)", self.a, 1, 10, row,
                          description="Расстояние между узлами решётки"); row += 1
        self._make_slider(left, "Радиус Бора R_bohr (Å)", self.R_bohr, 0.1, 2.0, row,
                          description="Радиус взаимодействия (×5 для поиска атомов)"); row += 1
        self._make_slider(left, "Полярный угол α (рад)", self.alpha, 0, np.pi, row,
                          description="Угол между направлением электрона и осью z"); row += 1
        self._make_slider(left, "Азимутальный угол β (рад)", self.beta, 0, np.pi, row,
                          description="Угол разворота направления вокруг оси z"); row += 1
        self._make_slider(left, "Размер решётки n", self.lattice_radius, 1, 20, row,
                          description="Число периодов по каждой оси", resolution=1); row += 1
        self._make_slider(left, "Слой источника d", self.d_layer, 0, 20, row,
                          description="Номер слоя, откуда вылетает электрон", resolution=1); row += 1

        ttk.Checkbutton(left, text="Автовыбор n по углам/слою",
                        variable=self.auto_n, command=self._recompute_n)\
            .grid(row=row, column=0, columnspan=4, sticky="w"); row += 1

        self.n_auto_label = ttk.Label(left, text="")
        self.n_auto_label.grid(row=row, column=0, columnspan=4, sticky="w"); row += 1

        ttk.Button(left, text="Пересчитать (решётка/матрицы)",
                   command=self.update_output_left).grid(row=row, column=0, columnspan=4, pady=5, sticky="ew"); row += 1

        self.output = tk.Text(left, width=70, height=28, font=("Consolas", 10), wrap="word")
        self.output.grid(row=row, column=0, columnspan=4, pady=5, sticky="nsew")
        left.grid_rowconfigure(row, weight=1)

        # ====== ПРАВАЯ ПАНЕЛЬ: Часть 2 ======
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        # Параметры Часть 2
        self.Z = tk.DoubleVar(value=29.0)
        self.b = tk.DoubleVar(value=0.53)
        self.c1 = tk.DoubleVar(value=1.0)
        self.c2 = tk.DoubleVar(value=1.0)
        self.dr = tk.DoubleVar(value=0.01)
        self.rmax = tk.DoubleVar(value=15.0)
        self.Emin = tk.DoubleVar(value=10.0)
        self.Emax = tk.DoubleVar(value=1.0e5)
        self.Npts = tk.IntVar(value=160)
        self.auto = tk.BooleanVar(value=True)

        row_r = 0
        self._make_slider(right, "Z (заряд ядра)", self.Z, 1, 92, row_r); row_r += 1
        self._make_slider(right, "b (Å)", self.b, 0.1, 2.0, row_r); row_r += 1
        self._make_slider(right, "c1", self.c1, 0.1, 3.0, row_r); row_r += 1
        self._make_slider(right, "c2", self.c2, 0.1, 3.0, row_r); row_r += 1
        self._make_slider(right, "dr (Å)", self.dr, 0.001, 0.1, row_r); row_r += 1
        self._make_slider(right, "r_max (Å)", self.rmax, 5.0, 40.0, row_r); row_r += 1
        self._make_slider(right, "Emin (эВ)", self.Emin, 1.0, 1000.0, row_r); row_r += 1
        self._make_slider(right, "Emax (эВ)", self.Emax, 1000.0, 200000.0, row_r); row_r += 1
        self._make_slider(right, "N точек", self.Npts, 20, 600, row_r, resolution=1); row_r += 1

        ttk.Checkbutton(right, text="Автопересчёт при движении ползунков",
                        variable=self.auto).grid(row=row_r, column=0, columnspan=3, sticky="w"); row_r += 1

        self.use_table_chi = tk.BooleanVar(value=True)
        self.i3_mode_sum = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="χ(x): табличная интерполяция (Thomas–Fermi)",
                        variable=self.use_table_chi, command=self.update_output_right)\
            .grid(row=row_r, column=0, columnspan=3, sticky="w"); row_r += 1
        ttk.Checkbutton(right, text="I3 как «сумма с усреднением» (шаг = dr)",
                        variable=self.i3_mode_sum, command=self.update_output_right)\
            .grid(row=row_r, column=0, columnspan=3, sticky="w"); row_r += 1

        ttk.Button(right, text="Пересчитать (Часть 2)",
                   command=self.update_output_right).grid(row=row_r, column=0, columnspan=3, sticky="ew", pady=4); row_r += 1

        # Два графика (на каждом две кривые: подготовка ↑ и ↓)
        self.fig = Figure(figsize=(6.4, 5.2), dpi=100)
        self.ax_sum = self.fig.add_subplot(211)   # проверочный: P↑+P↓
        self.ax_spin = self.fig.add_subplot(212)  # «удвоенный спин»: P↑−P↓
        for ax in (self.ax_sum, self.ax_spin):
            ax.grid(True, which="both")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=row_r, column=0, columnspan=3, sticky="nsew")
        right.grid_rowconfigure(row_r, weight=1)

        # Автопересчёт n при изменении геометрии
        for v in (self.a, self.R_bohr, self.alpha, self.beta, self.d_layer):
            v.trace_add("write", lambda *_: self._recompute_n())

        # Первый запуск
        self._recompute_n()
        self.update_output_left()
        self.update_output_right()

    # -------- UI helpers --------
    def _make_slider(self, parent, label, var, mn, mx, row, description="", resolution=0.01):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        frame.columnconfigure(1, weight=1)

        label_frame = ttk.Frame(frame)
        label_frame.grid(row=0, column=0, sticky="w")
        ttk.Label(label_frame, text=label).grid(row=0, column=0, sticky="w")
        if description:
            ttk.Label(label_frame, text=f"({description})", foreground="#555").grid(row=1, column=0, sticky="w")

        def on_change(_):
            if hasattr(self, "auto") and self.auto.get():
                self.update_output_right()

        sld = ttk.Scale(frame, from_=mn, to=mx, orient="horizontal", variable=var, command=on_change)
        sld.grid(row=0, column=1, sticky="ew", rowspan=2, padx=8)

        def format_value(value):
            try:
                if isinstance(var, tk.IntVar) or resolution >= 1:
                    return f"{int(round(float(value)))}"
                return f"{float(value):.3g}"
            except Exception:
                return str(value)

        val_lbl = ttk.Label(frame, text=format_value(var.get()))
        val_lbl.grid(row=0, column=2, rowspan=2, sticky="e")
        var.trace_add("write", lambda *_: val_lbl.config(text=format_value(var.get())))

    # -------- Автоподбор n --------
    def _recompute_n(self):
        """Пересчитать n по углам/слою и обновить слайдер+подпись (если включён авто-режим)."""
        if not self.auto_n.get():
            return
        a = float(self.a.get())
        R = float(self.R_bohr.get())
        alpha = float(self.alpha.get())
        beta = float(self.beta.get())
        d = int(self.d_layer.get())

        n_auto = compute_lattice_n_auto(a, R, alpha, beta, d)
        if n_auto != int(self.lattice_radius.get()):
            self.lattice_radius.set(n_auto)

        side = 2 * n_auto + 1
        approx_nodes = side ** 3
        self.n_auto_label.config(text=f"Авто n = {n_auto}  (~{approx_nodes:,} узлов)")

    # -------- Левая панель --------
    def update_output_left(self):
        a = float(self.a.get())
        R_bohr = float(self.R_bohr.get())
        alpha = float(self.alpha.get())
        beta = float(self.beta.get())
        interaction_radius = 5 * R_bohr
        n = int(self.lattice_radius.get())

        logger.info(
            "LEFT | a=%.4f Å, R_bohr=%.4f Å, interaction_radius=%.4f Å, "
            "alpha=%.4f rad, beta=%.4f rad, n=%d",
            a, R_bohr, interaction_radius, alpha, beta, n
        )

        atoms = nearest_atoms(a, interaction_radius, alpha, beta, n=n)
        logger.info("LEFT | найдено атомов: %d", len(atoms))

        matrices, inverses = transition_matrices(L_source=1)
        for Ln, D in matrices.items():
            logger.info(
                "LEFT | D(Ln=%d), det=%.6e, invertible=%s",
                Ln, np.linalg.det(D),
                inverses[Ln] is not None
            )

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

    # -------- Правая панель (Часть 2) --------
    def update_output_right(self):
        try:
            logger.info(
                "RIGHT | Emin=%.3g eV, Emax=%.3g eV, N=%d, Z=%.3g, a=%.3g Å, b=%.3g Å, "
                "c1=%.3g, c2=%.3g, dr=%.3g Å, rmax=%.3g Å",
                self.Emin.get(), self.Emax.get(), int(self.Npts.get()),
                self.Z.get(), self.a.get(), self.b.get(),
                self.c1.get(), self.c2.get(),
                self.dr.get(), self.rmax.get()
            )
            chi_fn = chi_table_interp if self.use_table_chi.get() else chi_default
            i3_mode = "sum_avg" if self.i3_mode_sum.get() else "trapz"

            df = compute_grid(
                self.Emin.get(), self.Emax.get(), int(self.Npts.get()),
                Z=self.Z.get(), a_ang=self.a.get(), b_ang=self.b.get(),
                c1=self.c1.get(), c2=self.c2.get(),
                dr_ang=self.dr.get(), r_max_ang=self.rmax.get(),
                chi=chi_fn, i3_mode=i3_mode
            )

            logger.info("RIGHT | сетка рассчитана: %d точек", len(df))
        except Exception as ex:
            logger.exception("RIGHT | ошибка при расчёте")
            for ax in (self.ax_sum, self.ax_spin):
                ax.clear()
                ax.text(0.05, 0.95, f"Ошибка: {ex}", transform=ax.transAxes,
                        va="top", ha="left")
                ax.grid(True, which="both")
            self.canvas.draw()
            return

        # Матрица D и её Ln (Ln используем как L_atom в фазовой матрице)
        matrices, _ = transition_matrices(L_source=1)
        #TODO: добавить комбобокс с выбором L
        if 0 in matrices:
            Ln, D = 1, matrices[1]
        else:
            Ln = next(iter(matrices.keys()))
            D = matrices[Ln]

        E = df["E_eV"].values
        Phi = df["Phi"].values  # = I_total

        sp = spin_amplitudes_both(E, Phi, D, L_atom=Ln)

        # График 1: проверочный спектр (две подготовки)
        self.ax_sum.clear()
        self.ax_sum.set_title("Проверочный спектр: P↑+P↓  (начальный ↑ и ↓)")
        self.ax_sum.plot(E, sp["sum_check_up"], label="начальный ↑")
        self.ax_sum.plot(E, sp["sum_check_dn"], label="начальный ↓")
        self.ax_sum.set_xlabel("Энергия, эВ")
        self.ax_sum.set_ylabel("Σ вероятностей")
        self.ax_sum.grid(True, which="both")
        self.ax_sum.legend()

        # График 2: «средний удвоенный спин» (две подготовки)
        self.ax_spin.clear()
        self.ax_spin.set_title("Средний удвоенный спин: P↑−P↓  (начальный ↑ и ↓)")
        self.ax_spin.plot(E, sp["spin_mean_up"], label="начальный ↑")
        self.ax_spin.plot(E, sp["spin_mean_dn"], label="начальный ↓")
        self.ax_spin.set_xlabel("Энергия, эВ")
        self.ax_spin.set_ylabel("P↑ − P↓")
        self.ax_spin.grid(True, which="both")
        self.ax_spin.legend()

        self.fig.tight_layout()
        self.canvas.draw()

        # краткая сводка в левый лог
        self.output.insert(
            tk.END,
            f"\n[Часть 2] E∈[{self.Emin.get():.3g},{self.Emax.get():.3g}] эВ, "
            f"N={int(self.Npts.get())}, Z={self.Z.get():.3g}, a={self.a.get():.3g} Å, b={self.b.get():.3g} Å, "
            f"χ={'table' if self.use_table_chi.get() else 'exp'}, "
            f"I3={'sum' if self.i3_mode_sum.get() else 'trapz'}, Ln={Ln}\n"
        )


if __name__ == "__main__":
    App().mainloop()
