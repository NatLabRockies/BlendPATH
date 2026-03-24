from typing import Literal

import numpy as np
import numpy.typing as npt

RE_LAMI = 2_300.0
RE_TURB = 4_000.0
FF_TYPES = Literal["hofer", "chen"]
F_LAMINAR = 64.0 / RE_LAMI


def get_friction_factor_vector(
    Re: npt.NDArray[np.float64],
    roughness_mm: npt.NDArray[np.float64],
    diameter_mm: npt.NDArray[np.float64],
    ff_type: FF_TYPES = "hofer",
) -> npt.NDArray[np.float64]:
    """
    Multi-regime friction factor:
      - laminar (Re < 2300): f = 64/Re
      - transitional (2300 ≤ Re < 4000): linear interpolation between laminar at Re=2300
        and turbulent at Re=4000
      - turbulent (Re ≥ 4000): Hofer correlation (similar to Colebrook)
    """

    # Initially set all to laminar
    f_out = get_f_laminar(Re=Re)

    # Transitional
    trans_ind = np.logical_and(Re >= RE_LAMI, Re < RE_TURB)
    f_out[trans_ind] = get_f_transitional(
        Re=Re[trans_ind],
        e=roughness_mm[trans_ind],
        d=diameter_mm[trans_ind],
        ff_type=ff_type,
    )
    # Turbulant
    turb_ind = Re >= RE_TURB
    f_out[turb_ind] = friction_factor_correlation(
        Re=Re[turb_ind],
        e=roughness_mm[turb_ind],
        d=diameter_mm[turb_ind],
        ff_type=ff_type,
    )

    return f_out


def get_f_laminar(Re: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 64.0 / Re


def get_f_transitional(
    Re: npt.NDArray[np.float64],
    e: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    ff_type: FF_TYPES,
) -> npt.NDArray[np.float64]:
    f_turbulent = friction_factor_correlation(
        Re=RE_TURB,
        e=e,
        d=d,
        ff_type=ff_type,
    )
    weight = (Re - RE_LAMI) / (RE_TURB - RE_LAMI)
    return F_LAMINAR + weight * (f_turbulent - F_LAMINAR)


def hofer_friction(
    Re: npt.NDArray[np.float64], e: npt.NDArray[np.float64], d: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (-2 * np.log10(4.518 / Re * np.log10(Re / 7) + e / (3.71 * d))) ** (-2)


def chen_friction(
    Re: npt.NDArray[np.float64],
    e: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return (
        1
        / (
            -2
            * np.log10(
                e / (3.7065 * d)
                - 5.0452
                / Re
                * np.log10((1 / 2.8257 * (e / d) ** 1.1098) + (5.8506 / Re**0.8981))
            )
        )
    ) ** 2


friction_factor_hash = {"hofer": hofer_friction, "chen": chen_friction}


def friction_factor_correlation(
    Re: npt.NDArray[np.float64],
    e: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    ff_type: FF_TYPES,
) -> npt.NDArray[np.float64]:
    if e.size == 0:
        return np.array([])
    if ff_type not in friction_factor_hash:
        raise ValueError(f"{ff_type} is not a valid friction factor correlation")
    return friction_factor_hash[ff_type](Re=Re, e=e, d=d)
