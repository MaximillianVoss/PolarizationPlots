# polarization_part2.py
# -*- coding: utf-8 -*-
import os
from datetime import datetime
from typing import Callable, Dict, Tuple, Literal
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


# -------------------- Физические константы --------------------
E_CHARGE = 1.602176634e-19  # Кл
M_E = 9.1093837015e-31      # кг

# -------------------- E <-> v --------------------
def energy_eV_to_speed_mps(E_eV: np.ndarray) -> np.ndarray:
    """Энергия (эВ) -> скорость (м/с) по нерелятивистской формуле E = 1/2 m v^2."""
    E_J = np.asarray(E_eV, dtype=float) * E_CHARGE
    return np.sqrt(2.0 * E_J / M_E)

def speed_mps_to_energy_eV(V: np.ndarray) -> np.ndarray:
    """Скорость (м/с) -> энергия (эВ) по нерелятивистской формуле."""
    E_J = 0.5 * M_E * np.asarray(V, dtype=float) ** 2
    return E_J / E_CHARGE

# -------------------- χ(x): таблица + интерполяция --------------------
# Усечённая табличка Томаса–Ферми (можно расширять при необходимости)
_CHI_X = np.array([
    0.00,0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,
    0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.0,1.2,1.4,1.6,1.8,2.0,
    2.2,2.4,2.6,2.8,3.0,4.0,5.0,6.0,8.0,10.0,15.0,20.0,30.0,40.0,50.0,60.0
], dtype=float)
_CHI_Y = np.array([
    1.000,0.972,0.947,0.924,0.902,0.882,0.863,0.845,0.828,0.812,0.797,
    0.721,0.667,0.621,0.580,0.544,0.512,0.482,0.454,0.374,0.333,0.298,0.268,0.243,
    0.221,0.202,0.185,0.170,0.157,0.105,0.0788,0.0594,0.0366,0.0243,0.0123,0.0088,0.0035,0.0022,0.00063,0.00039
], dtype=float)

def chi_table_interp(x: np.ndarray, params: Dict[str, float] | None = None) -> np.ndarray:
    """
    χ(x) по таблице с линейной интерполяцией.
    Вне диапазона таблицы применяется клэмп к крайним значениям.
    """
    x = np.asarray(x, dtype=float)
    return np.interp(x, _CHI_X, _CHI_Y, left=_CHI_Y[0], right=_CHI_Y[-1])

def chi_default(x: np.ndarray, params: Dict[str, float] | None = None) -> np.ndarray:
    """Простая экспоненциальная аппроксимация на случай отладки."""
    x = np.asarray(x, dtype=float)
    return np.exp(-x)

# -------------------- Интегралы I1, I2, I3 --------------------
BOHR_TO_ANG = 0.52917721092
C_LIGHT = 299792458.0
INV_ALPHA = 137.04

def compute_I_components(
    V: float,
    a_ang: float,
    Z: float,
    b_ang: float,
    c1: float,
    c2: float,
    dr_ang: float,
    r_max_ang: float,
    chi,
    chi_params=None,
    i3_mode: str = "sum_avg",
):
    """
    Считает I1, I2, I3.

    ВАЖНО (исправление единиц):
    - входные длины заданы в Å, внутри переводим в a0 (атомные единицы): x[a0] = x[Å] / 0.529177...
    - V приходит в м/с (из energy_eV_to_speed_mps), переводим в атомные единицы скорости:
          V_au = (V / c) * 137.04
    """

    # --- скорость: м/с -> a.u.  ---
    V = float(V)
    V_au = (V / C_LIGHT) * INV_ALPHA
    if not np.isfinite(V_au) or V_au <= 0.0:
        raise ValueError(f"Некорректная скорость: V={V} м/с, V_au={V_au}")

    # --- длины: Å -> a0 ---
    a = float(a_ang) / BOHR_TO_ANG
    b = float(b_ang) / BOHR_TO_ANG
    dr = float(dr_ang) / BOHR_TO_ANG
    r_max = float(r_max_ang) / BOHR_TO_ANG

    # --- проверки ---
    EPS_A = 1e-6  # a0
    if not np.isfinite(a) or a <= EPS_A:
        raise ValueError(f"a слишком мал/некорректен: a_ang={a_ang}, a={a} a0")
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError(f"b некорректен: b_ang={b_ang}, b={b} a0")
    if not np.isfinite(dr) or dr <= 0.0:
        raise ValueError(f"dr некорректен: dr_ang={dr_ang}, dr={dr} a0")
    if not np.isfinite(r_max) or r_max <= a:
        raise ValueError(f"r_max должен быть > a: r_max_ang={r_max_ang}, r_max={r_max} a0, a={a} a0")

    z13 = Z ** (1.0 / 3.0)
    pref = 1.0 / (a ** 5)

    # сетка r в a0
    n_r = int(max(2, np.ceil((r_max - a) / dr) + 1))
    r = np.linspace(a, r_max, n_r)

    # χ аргумент безразмерный: Z^(1/3) * r/b (оба в a0)
    x = z13 * r / b
    chi_r = chi(x, chi_params) if chi_params is not None else chi(x)
    chi_r = np.asarray(chi_r, dtype=float)
    chi_r = np.clip(chi_r, 0.0, None)          # защита от отрицательных значений
    chi_r_32 = np.power(chi_r, 1.5)

    term = (r ** 2 - a ** 2)

    # I1, I2
    f1 = term / np.power(r, 2.5) * chi_r_32
    f2 = term / np.power(r, 4.0) * chi_r

    I1_int = np.trapz(f1, r)
    I2_int = np.trapz(f2, r)

    # I3
    if i3_mode == "trapz":
        r_shift = np.minimum(r + dr, r_max)
        x_shift = z13 * r_shift / b
        chi_shift = chi(x_shift, chi_params) if chi_params is not None else chi(x_shift)
        chi_shift = np.asarray(chi_shift, dtype=float)
        chi_shift = np.clip(chi_shift, 0.0, None)

        f3 = term / np.power(r, 3.0) * (chi_shift - chi_r)
        I3_int = np.trapz(f3, r)

    elif i3_mode == "sum_avg":
        p = 0.0
        count = 0
        r_val = a
        while (r_val + dr) <= r_max:
            x0 = z13 * r_val / b
            x1 = z13 * (r_val + dr) / b

            chi0 = chi(np.array([x0]), chi_params)[0] if chi_params is not None else chi(np.array([x0]))[0]
            chi1 = chi(np.array([x1]), chi_params)[0] if chi_params is not None else chi(np.array([x1]))[0]
            chi0 = float(max(0.0, chi0))
            chi1 = float(max(0.0, chi1))

            term0 = (r_val ** 2 - a ** 2)
            p += term0 / (r_val ** 3) * (chi1 - chi0)

            count += 1
            r_val += dr

        I3_int = p / max(count, 1)

    else:
        raise ValueError(f"Неизвестный i3_mode: {i3_mode}")

    # --- итоговые I1,I2,I3 (везде используем V_au) ---
    I1 = (-(2.0 * c1 * c2) / V_au) * (Z ** 1.5) * pref * I1_int
    I2 = (-(6.0 * c1 * Z) / V_au) * pref * I2_int
    I3 = ((6.0 * c1 * Z * b) / (V_au * (Z ** (1.0 / 3.0)))) * pref * I3_int

    It = I1 + I2 + I3
    return I1, I2, I3, It


def compute_grid_atoms(
    Emin_eV: float,
    Emax_eV: float,
    N: int,
    *,
    a_list_ang: list[float],   # список d_прямой (Å)
    Z: float,
    b_ang: float,
    c1: float,
    c2: float,
    dr_ang: float,
    r_max_ang: float,
    chi: Callable[[np.ndarray, Dict[str, float] | None], np.ndarray] = chi_table_interp,
    chi_params: Dict[str, float] | None = None,
    i3_mode: Literal["trapz", "sum_avg"] = "sum_avg",
    dump_atom_phi_csv: bool = True,   # <-- НОВОЕ: сохранять промежуточные Φ_k
    max_atoms_dump: int = 200,        # <-- НОВОЕ: ограничить кол-во атомов в дампе
) -> pd.DataFrame:
    """
    Сетка по энергии, но Phi(E) = Σ_k Phi_k(E), где для каждого атома k:
      a = d_прямой(k) (impact parameter).

    Дополнительно (опционально) сохраняет промежуточные Φ_k(E) в CSV.
    """

    if Emin_eV <= 0 or Emax_eV <= 0 or Emax_eV <= Emin_eV:
        raise ValueError("Требуется 0 < Emin < Emax.")
    if not a_list_ang:
        raise ValueError("Список a_list_ang пуст: нет атомов для суммирования.")
    if any(a <= 0 for a in a_list_ang):
        raise ValueError("В a_list_ang найдены некорректные значения (<=0).")

    E = np.logspace(np.log10(Emin_eV), np.log10(Emax_eV), int(N))
    V = energy_eV_to_speed_mps(E)

    a_arr = np.asarray(a_list_ang, dtype=float)

    # ограничим число атомов (и для скорости, и для сохранения)
    if a_arr.size > int(max_atoms_dump):
        a_arr = np.sort(a_arr)[:int(max_atoms_dump)]
    else:
        a_arr = np.sort(a_arr)

    I1, I2, I3, It = [], [], [], []

    # матрицы вкладов по атомам (по желанию)
    Phi_atoms = np.empty((len(E), len(a_arr)), dtype=float) if dump_atom_phi_csv else None

    for iE, v in enumerate(V):
        s1 = s2 = s3 = st = 0.0

        for jA, a_imp in enumerate(a_arr):
            i1, i2, i3, it = compute_I_components(
                float(v),
                Z=Z, a_ang=float(a_imp), b_ang=b_ang, c1=c1, c2=c2,
                dr_ang=dr_ang, r_max_ang=r_max_ang,
                chi=chi, chi_params=chi_params, i3_mode=i3_mode
            )
            s1 += i1; s2 += i2; s3 += i3; st += it

            if Phi_atoms is not None:
                Phi_atoms[iE, jA] = float(it)  # вклад k-го атома в Φ(E)

        I1.append(s1); I2.append(s2); I3.append(s3); It.append(st)

    logger.info(
        "compute_grid_atoms | Emin=%.3g eV, Emax=%.3g eV, N=%d, atoms=%d, i3_mode=%s",
        Emin_eV, Emax_eV, int(N), int(len(a_arr)), i3_mode
    )

    phi = np.asarray(It, dtype=float)
    logger.info(
        "compute_grid_atoms | Phi stats: min=%.6g, max=%.6g, mean=%.6g",
        float(phi.min()), float(phi.max()), float(phi.mean())
    )

    df_sum = pd.DataFrame({
        "E_eV": E,
        "V_m_per_s": V,
        "I1": np.asarray(I1, dtype=float),
        "I2": np.asarray(I2, dtype=float),
        "I3": np.asarray(I3, dtype=float),
        "Phi": phi,
    })

    # --- сохранение промежуточных Φ_k(E) ---
    if dump_atom_phi_csv and Phi_atoms is not None:
        try:
            os.makedirs("data", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # (1) "широкий" формат: одна строка = энергия, колонки = атомы (a_k)
            cols = ["E_eV"] + [f"a_{k}_A={a_arr[k]:.6g}" for k in range(len(a_arr))]
            wide = np.column_stack([E, Phi_atoms])
            df_wide = pd.DataFrame(wide, columns=cols)
            path_wide = os.path.join("data", f"phi_atoms_matrix_{ts}.csv")
            df_wide.to_csv(path_wide, index=False, encoding="utf-8")
            logger.info("compute_grid_atoms | saved per-atom Phi matrix: %s", os.path.abspath(path_wide))

            # (2) "длинный" формат: удобно фильтровать/группировать
            # E_i, atom_idx, a, Phi_atom
            df_long = pd.DataFrame({
                "E_eV": np.repeat(E, len(a_arr)),
                "atom_idx": np.tile(np.arange(len(a_arr)), len(E)),
                "a_ang": np.tile(a_arr, len(E)),
                "Phi_atom": Phi_atoms.reshape(-1),
            })
            path_long = os.path.join("data", f"phi_atoms_long_{ts}.csv")
            df_long.to_csv(path_long, index=False, encoding="utf-8")
            logger.info("compute_grid_atoms | saved per-atom Phi long: %s", os.path.abspath(path_long))

        except Exception:
            logger.exception("compute_grid_atoms | failed to save per-atom Phi CSV")

    return df_sum



def compute_grid(
    Emin_eV: float,
    Emax_eV: float,
    N: int,
    *,
    Z: float,
    a_ang: float,
    b_ang: float,
    c1: float,
    c2: float,
    dr_ang: float,
    r_max_ang: float,
    chi: Callable[[np.ndarray, Dict[str, float] | None], np.ndarray] = chi_table_interp,
    chi_params: Dict[str, float] | None = None,
    i3_mode: Literal["trapz", "sum_avg"] = "sum_avg",
) -> pd.DataFrame:
    """Считает сетку по энергии и возвращает таблицу со столбцами E, V, I1, I2, I3 и Phi = I_total."""

    if Emin_eV <= 0 or Emax_eV <= 0 or Emax_eV <= Emin_eV:
        raise ValueError("Требуется 0 < Emin < Emax.")
    E = np.logspace(np.log10(Emin_eV), np.log10(Emax_eV), int(N))
    V = energy_eV_to_speed_mps(E)

    I1, I2, I3, It = [], [], [], []
    for v in V:
        i1, i2, i3, it = compute_I_components(
            v,
            Z=Z, a_ang=a_ang, b_ang=b_ang, c1=c1, c2=c2,
            dr_ang=dr_ang, r_max_ang=r_max_ang,
            chi=chi, chi_params=chi_params, i3_mode=i3_mode
        )
        I1.append(i1); I2.append(i2); I3.append(i3); It.append(it)

    logger.info(
        "compute_grid | Emin=%.3g eV, Emax=%.3g eV, N=%d, i3_mode=%s",
        Emin_eV, Emax_eV, N, i3_mode
    )

    # ===== Вариант A: логирование сводки по Phi + несколько контрольных точек =====
    phi = np.asarray(It, dtype=float)

    logger.info(
        "compute_grid | Phi stats: min=%.6g, max=%.6g, mean=%.6g",
        float(phi.min()), float(phi.max()), float(phi.mean())
    )

    if len(phi) > 0:
        idx0 = 0
        idxm = len(phi) // 2
        idxe = len(phi) - 1
        logger.info(
            "compute_grid | Phi samples: "
            "E0=%.6g Phi0=%.6g | Emid=%.6g Phimid=%.6g | Eend=%.6g Phiend=%.6g",
            float(E[idx0]), float(phi[idx0]),
            float(E[idxm]), float(phi[idxm]),
            float(E[idxe]), float(phi[idxe]),
        )
    # ============================================================================

    # ===== Сохранение массива Phi (и I1/I2/I3) в CSV =====
    try:
        os.makedirs("data", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join("data", f"phi_dump_{ts}.csv")

        df_dump = pd.DataFrame({
            "E_eV": E,
            "V_m_per_s": V,
            "I1": np.asarray(I1, dtype=float),
            "I2": np.asarray(I2, dtype=float),
            "I3": np.asarray(I3, dtype=float),
            "Phi": phi,
        })
        df_dump.to_csv(csv_path, index=False, encoding="utf-8")

        logger.info(
            "compute_grid | Phi saved to CSV: %s (%d rows)",
            os.path.abspath(csv_path),
            len(df_dump)
        )
    except Exception:
        logger.exception("compute_grid | failed to save Phi CSV")
    # ======================================================

    return pd.DataFrame({
        "E_eV": E,
        "V_m_per_s": V,
        "I1": np.array(I1),
        "I2": np.array(I2),
        "I3": np.array(I3),
        "Phi": phi,   # суммарная матричная часть
    })


# -------------------- Амплитуды/вероятности для двух подготовок спина --------------------
def spin_amplitudes_both(
    E: np.ndarray,
    Phi: np.ndarray,
    D: np.ndarray,
    L_atom: int = 0,
):
    """
    Строим амплитуды для двух начальных спинов |↑>=[1,0] и |↓>=[0,1].
    Учитываем мнимую фазу (диагональная матрица):
        U(E) = diag( exp(+i * L_atom * Phi(E)),  exp(-i * (L_atom + 1) * Phi(E)) )
    Вычисляем вероятности как |амплитуда|^2 (сначала модуль, затем квадрат).
    Возвращает словарь с четырьмя кривыми: sum/spin для ↑ и ↓.
    """
    D = np.asarray(D, dtype=complex)
    Dinv = np.linalg.inv(D)

    ket_up = np.array([0.5+0j, 0.0+0j])
    ket_dn = np.array([0.0+0j, 0.5+0j])

    Phi = np.asarray(Phi, dtype=float)

    P_up_from_up   = np.empty_like(Phi, dtype=float)
    P_dn_from_up   = np.empty_like(Phi, dtype=float)
    P_up_from_down = np.empty_like(Phi, dtype=float)
    P_dn_from_down = np.empty_like(Phi, dtype=float)

    for i, phi in enumerate(Phi):
        # Фазовая матрица (комплексная)
        U = np.array([
            [np.exp(1j * L_atom * phi),            0.0+0j],
            [0.0+0j, np.exp(-1j * (L_atom + 1) * phi)]
        ], dtype=complex)

        # ψ_out = 2 * D^{-1} * U * D * |ψ_in>
        amp_up   = 2.0 * (Dinv @ (U @ (D @ ket_up)))
        amp_down = 2.0 * (Dinv @ (U @ (D @ ket_dn)))

        # Вероятности: сначала модуль, потом квадрат
        P_up_from_up[i]   = np.abs(amp_up[0])**2
        P_dn_from_up[i]   = np.abs(amp_up[1])**2
        P_up_from_down[i] = np.abs(amp_down[0])**2
        P_dn_from_down[i] = np.abs(amp_down[1])**2

    return {
        # начальный ↑
        "sum_check_up":  P_up_from_up + P_dn_from_up,
        "spin_mean_up":  P_up_from_up - P_dn_from_up,
        # начальный ↓
        "sum_check_dn":  P_up_from_down + P_dn_from_down,
        "spin_mean_dn":  P_up_from_down - P_dn_from_down,
    }
