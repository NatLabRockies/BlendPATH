from typing import Literal, Union, get_args

import cantera as ct
import numpy as np

# Local BlendPATH helpers -----------------------------------------------------
from . import Composition
from . import cantera_util as ctu

# Optional: CoolProp for GERG-2008 -------------------------------------------
try:
    import CoolProp.CoolProp as cp

    _HAS_COOLPROP = True
except ModuleNotFoundError:  # keep BlendPATH runnable without CoolProp
    _HAS_COOLPROP = False

# -----------------------------------------------------------------------------
# Supported equations of state
# -----------------------------------------------------------------------------
_EOS_OPTIONS = Literal["rk", "papay", "gerg"]

# -----------------------------------------------------------------------------
# Map BlendPATH species keys → CoolProp GERG fluid names
# Extend as needed when you introduce additional components
# -----------------------------------------------------------------------------
_COOLPROP_NAMES = {
    "H2": "Hydrogen",
    "CH4": "Methane",
    "C2H6": "Ethane",
    "C3H8": "Propane",
    "C4H10": "Butane",
    "IC4H10": "Isobutane",
    "C5H12": "Pentane",
    "CO2": "CarbonDioxide",
    "N2": "Nitrogen",
}

_CP_PROP_FILTER = {
    "density": "D",
    "s": "S",
    "h": "H",
    "mean_molecular_weight": "M",
    "viscosity": "V",
}

_CP_PROP_UNITFIX = {
    "mean_molecular_weight": 1000,
}


# -----------------------------------------------------------------------------
# Public helper
# -----------------------------------------------------------------------------


def get_rz(
    p_gauge: float = ct.one_atm,
    T_K: float = 273.15,
    X: Composition = None,
    eos: _EOS_OPTIONS = "rk",
    mw: float = 0.0,
) -> tuple[float, float]:
    """Return density [kg/m³] and compressibility Z for the requested EOS."""
    eos_lc = eos.lower()
    if eos_lc not in get_args(_EOS_OPTIONS):
        raise ValueError(f"{eos} is not a valid EOS option…")

    if X is None:
        raise ValueError("EOS requires a Composition object.")

    # PAPAY
    if eos_lc == "papay":
        return _eos_papay(p_gauge, T_K, X.pc, X.tc, mw)

    # RK
    if eos_lc == "rk":
        return _eos_rk(p_gauge, T_K, X.x, mw)

    # GERG with fallback to RK on failure
    if eos_lc == "gerg":
        if not _HAS_COOLPROP:
            raise ImportError("CoolProp is required for eos='gerg'.")

        # attempt GERG
        try:
            return _eos_gerg(p_gauge, T_K, X.x, mw)
        except Exception:
            raise ValueError("GERG EOS could not evaluate")


# -----------------------------------------------------------------------------
# Individual EOS implementations
# -----------------------------------------------------------------------------


def _eos_papay(
    p_gauge: float, T_K: float, pc: float, tc: float, mw: float
) -> tuple[float, float]:
    """Papay correlation – quick but CH₄-centric, recommend P < 10 MPa."""
    p_abs = p_gauge + ct.one_atm
    P_r = p_abs / pc
    T_r = T_K / tc
    z = 1 - 3.53 * P_r / 10 ** (0.9813 * T_r) + 0.274 * P_r**2 / 10 ** (0.8157 * T_r)
    rho = p_abs / ctu.R_GAS / T_K / z * mw
    return rho, z


def _eos_rk(
    p_gauge: float, T_K: float, X: Union[str, dict[str, float]], mw: float
) -> tuple[float, float]:
    """Redlich–Kwong via Cantera"""
    p_abs = p_gauge + ct.one_atm
    ctu.gas.TPX = T_K, p_abs, X
    rho = ctu.gas.density

    # RK can mis-converge near critical; probe ±5 % P for stability
    if np.isnan(rho) or rho == np.inf:
        ctu.gas.TPX = T_K, p_abs * 1.05, X
        rho_hi = ctu.gas.density
        ctu.gas.TPX = T_K, p_abs * 0.95, X
        rho_lo = ctu.gas.density
        rho = 0.5 * (rho_hi + rho_lo)

    z = p_abs * mw / ctu.R_GAS / T_K / rho
    return rho, z


# -----------------------------------------------------------------------------
# GERG helper
# -----------------------------------------------------------------------------


def _mix_to_cp(comp: dict[str, float]) -> str:
    """Convert {"CH4":0.9,"H2":0.1} → 'Methane[0.9]&Hydrogen[0.1]'."""
    parts = []
    for sp, frac in comp.items():
        cp_name = _COOLPROP_NAMES.get(sp.upper())
        if cp_name is None:
            raise ValueError(f"Species '{sp}' not mapped to CoolProp name.")
        parts.append(f"{cp_name}[{frac}]")
    return "&".join(parts)


def _eos_gerg(
    p_gauge: float, T_K: float, comp: dict[str, float], mw: float
) -> tuple[float, float]:
    """GERG-2008 density & Z via CoolProp (valid for CH₄±H₂ blends)."""
    p_abs = p_gauge + ct.one_atm
    mix_str = _mix_to_cp(comp)
    rho = cp.PropsSI("D", "P", p_abs, "T", T_K, mix_str)
    z = cp.PropsSI("Z", "P", p_abs, "T", T_K, mix_str)
    return rho, z


# -----------------------------------------------------------------------------
# Set state
# -----------------------------------------------------------------------------
def set_state(input_type: str, input_vals: tuple, eos_type: _EOS_OPTIONS) -> float:
    # Check eos type is valid
    if eos_type not in get_args(_EOS_OPTIONS):
        raise ValueError(f"{eos_type} is not a valid EOS option…")

    # Check input type, must be TPX or SPX
    if input_type not in ["TPX", "SPX"]:
        raise ValueError(
            f"{input_type} is not a valid input type option, use TPX or SPX"
        )

    # papay
    if eos_type == "papay":
        return _papay_set_state()

    # RK
    if eos_type == "rk":
        return _rk_set_state(input_type=input_type, input_vals=input_vals)

    # RK
    if eos_type == "gerg":
        return _gerg_set_state(input_vals=input_vals)


def _rk_set_state(input_type: str, input_vals: tuple) -> float:
    p = input_vals[1]
    if input_type == "TPX":
        no_sol = True
        while no_sol:
            try:
                if input_type == "TPX":
                    ctu.gas.TPX = input_vals[0], p, input_vals[2]
                no_sol = False
            except ct.CanteraError:
                p *= 1.0001
    elif input_type == "SPX":
        ctu.gas.SPX = input_vals[0], p, input_vals[2]
    return p


def _gerg_set_state(input_vals: tuple) -> float:
    p = input_vals[1]
    return p


def _papay_set_state(input_vals: tuple) -> float:
    p = input_vals[1]
    return p


# -----------------------------------------------------------------------------
# Get EOS properties
# -----------------------------------------------------------------------------
def get_props(
    input_type: str,
    input_vals: tuple,
    eos_type: _EOS_OPTIONS,
    properties: list,
    set_state_flag: bool = True,
) -> tuple:
    if set_state_flag:
        set_state(input_type=input_type, input_vals=input_vals, eos_type=eos_type)

    # papay
    # if eos_type == "papay":
    #     return _papay_set_state()

    # RK
    if eos_type == "rk":
        return _get_rk(properties=properties)

    # RK
    if eos_type == "gerg":
        return _get_gerg(
            input_type=input_type, input_vals=input_vals, properties=properties
        )


def _flatten(vals):
    if len(vals) == 1:
        return vals[0]
    return vals


def _get_rk(properties) -> Union[tuple, float]:
    return _flatten([getattr(ctu.gas, prop) for prop in properties])


def _get_gerg(
    input_type: str, input_vals: tuple, properties: tuple
) -> Union[tuple, float]:
    state_var = "T"
    if input_type == "TPX":
        state_var = "T"
    elif input_type == "SPX":
        state_var = "S"

    vals = []
    for prop in properties:
        try:
            vals.append(
                cp.PropsSI(
                    _CP_PROP_FILTER[prop],
                    "P",
                    input_vals[1],
                    state_var,
                    input_vals[0],
                    _mix_to_cp(input_vals[2]),
                )
                * _CP_PROP_UNITFIX.get(prop, 1)
            )
        except ValueError as e:
            vals.append(np.nan)

    return _flatten(vals)
