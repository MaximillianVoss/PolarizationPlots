# polarization_part2.py
# -*- coding: utf-8 -*-
from typing import Callable, Dict, Tuple
import numpy as np
import pandas as pd

E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31

def energy_eV_to_speed_mps(E_eV: np.ndarray) -> np.ndarray:
    E_J = E_eV * E_CHARGE
    return np.sqrt(2.0 * E_J / M_E)

def speed_mps_to_energy_eV(V: np.ndarray) -> np.ndarray:
    E_J = 0.5 * M_E * V**2
    return E_J / E_CHARGE

def chi_default(x: np.ndarray, params: Dict[str, float] | None = None) -> np.ndarray:
    return np.exp(-x)

def compute_I_components(
    V: float,
    *, Z: float, a_ang: float, b_ang: float, c1: float, c2: float,
    dr_ang: float, r_max_ang: float,
    chi: Callable[[np.ndarray, Dict[str, float] | None], np.ndarray] = chi_default,
    chi_params: Dict[str, float] | None = None,
    n_r: int = 4000,
) -> Tuple[float, float, float, float]:
    if r_max_ang <= a_ang:
        raise ValueError("r_max_ang должен быть больше a_ang.")
    r = np.linspace(a_ang, r_max_ang, n_r)
    a, b, dr = float(a_ang), float(b_ang), float(dr_ang)
    z13 = Z ** (1/3)
    pref = 1.0 / (a**5)

    x = z13 * r / b
    chi_r = chi(x, chi_params)
    chi_r_32 = np.power(chi_r, 1.5)

    term = (r**2 - a**2)
    f1 = term / np.power(r, 2.5) * chi_r_32
    f2 = term / np.power(r, 4.0) * chi_r
    chi_r_shift = chi(z13 * (r + dr) / b, chi_params)
    f3 = term / np.power(r, 3.0) * (chi_r_shift - chi_r)

    I1_int = np.trapz(f1, r)
    I2_int = np.trapz(f2, r)
    I3_int = np.trapz(f3, r)

    I1 = (-(2*c1*c2)/V) * (Z**1.5) * pref * I1_int
    I2 = (-(6*c1*Z)/V) * pref * I2_int
    I3 = ((6*c1*Z*b)/(V*(Z**(1/3)))) * pref * I3_int
    return I1, I2, I3, (I1 + I2 + I3)

def compute_grid(
    Emin_eV: float, Emax_eV: float, N: int,
    *, Z: float, a_ang: float, b_ang: float, c1: float, c2: float,
    dr_ang: float, r_max_ang: float,
    chi: Callable[[np.ndarray, Dict[str, float] | None], np.ndarray] = chi_default,
    chi_params: Dict[str, float] | None = None,
) -> pd.DataFrame:
    if Emin_eV <= 0 or Emax_eV <= 0 or Emax_eV <= Emin_eV:
        raise ValueError("Требуется 0 < Emin < Emax.")
    E = np.logspace(np.log10(Emin_eV), np.log10(Emax_eV), N)
    V = energy_eV_to_speed_mps(E)
    I1, I2, I3, It = [], [], [], []
    for v in V:
        i1, i2, i3, it = compute_I_components(
            v, Z=Z, a_ang=a_ang, b_ang=b_ang, c1=c1, c2=c2,
            dr_ang=dr_ang, r_max_ang=r_max_ang, chi=chi, chi_params=chi_params
        )
        I1.append(i1); I2.append(i2); I3.append(i3); It.append(it)
    import pandas as pd
    return pd.DataFrame({
        "E_eV": E, "V_m_per_s": V,
        "I1": np.array(I1), "I2": np.array(I2), "I3": np.array(I3),
        "I_total": np.array(It),
    })
