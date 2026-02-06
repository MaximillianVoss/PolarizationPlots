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
def compute_I_components(
    V: float,
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
    n_r: int = 4000,
    i3_mode: Literal["trapz", "sum_avg"] = "sum_avg",
) -> Tuple[float, float, float, float]:
    """
    Реализация формул «Части 2».
    I1 = (−(2*c1*c2)/V) * Z^(3/2) / a^5 * ∫ (r^2 − a^2)/r^(5/2) * χ^(3/2)( Z^(1/3) * r/b ) dr
    I2 = (−(6*c1*Z)/V) * 1/a^5 * ∫ (r^2 − a^2)/r^4 * χ( Z^(1/3) * r/b ) dr
    I3 = ( 6*c1*Z*b/(V*Z^(1/3)) ) * 1/a^5 * ∫ (r^2 − a^2)/r^3 * [ χ( Z^(1/3)*(r+dr)/b ) − χ( Z^(1/3)*r/b ) ] dr

    Режим I3:
      - "trapz"   : обычное интегрирование разности χ(r+dr)-χ(r) по r-сетке (trapz);
      - "sum_avg" : «интеграл суммой» из примечания — идём r=a..r_max шагом dr, суммируем выражение и усредняем.
    """



    if r_max_ang <= a_ang:
        raise ValueError("r_max_ang должен быть больше a_ang.")

    a = float(a_ang); b = float(b_ang); dr = float(dr_ang)
    z13 = Z ** (1.0/3.0)
    pref = 1.0 / (a**5)

    # r-сетка для I1 и I2 (trapz)
    r = np.linspace(a, r_max_ang, int(n_r))
    x = z13 * r / b
    chi_r = chi(x, chi_params)
    chi_r_32 = np.power(chi_r, 1.5)

    term = (r**2 - a**2)
    f1 = term / np.power(r, 2.5) * chi_r_32
    f2 = term / np.power(r, 4.0) * chi_r

    I1_int = np.trapz(f1, r)
    I2_int = np.trapz(f2, r)

    if i3_mode == "trapz":
        chi_r_shift = chi(z13 * (r + dr) / b, chi_params)
        f3 = term / np.power(r, 3.0) * (chi_r_shift - chi_r)
        I3_int = np.trapz(f3, r)
    else:
        # Суммирование по шагам dr с последующим усреднением (как в примечании)
        p = 0.0
        count = 0
        r_val = a
        while (r_val + dr) <= r_max_ang:
            x0 = z13 * r_val / b
            x1 = z13 * (r_val + dr) / b
            chi0 = chi(np.array([x0]), chi_params)[0]
            chi1 = chi(np.array([x1]), chi_params)[0]
            term0 = (r_val**2 - a**2)
            p += term0 / (r_val**3) * (chi1 - chi0)
            count += 1
            r_val += dr
        I3_int = p / max(count, 1)

    I1 = (-(2.0 * c1 * c2) / V) * (Z ** 1.5) * pref * I1_int
    I2 = (-(6.0 * c1 * Z) / V) * pref * I2_int
    I3 = ((6.0 * c1 * Z * b) / (V * (Z ** (1.0/3.0)))) * pref * I3_int

    logger.debug(
        "I_components | V=%.3e, Z=%.3g, a=%.3g, b=%.3g, dr=%.3g, rmax=%.3g",
        V, Z, a_ang, b_ang, dr_ang, r_max_ang
    )

    return I1, I2, I3, (I1 + I2 + I3)

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
