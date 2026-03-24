from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np

import BlendPATH.Global as gl

MPA2PSI = 145.038
PSA2KPSI = 1 / 1000
MPA2KPSI = MPA2PSI * PSA2KPSI
# https://www.neelconsteel.com/api-5l-seamless-pipe.html
PIPE_STR = {
    "A": {"SMYS": 30500, "SMTS": 48600},
    "B": {"SMYS": 35500, "SMTS": 60200},
    "X42": {"SMYS": 42100, "SMTS": 60200},
    "X46": {"SMYS": 46400, "SMTS": 63100},
    "X52": {"SMYS": 52200, "SMTS": 66700},
    "X56": {"SMYS": 56600, "SMTS": 71100},
    "X60": {"SMYS": 60200, "SMTS": 75400},
    "X65": {"SMYS": 65300, "SMTS": 77500},
    "X70": {"SMYS": 70300, "SMTS": 82700},
}
PIPE_STR = {
    x: {"SMYS": v["SMYS"] / MPA2PSI, "SMTS": v["SMTS"] / MPA2PSI}
    for x, v in PIPE_STR.items()
    if x not in ["A"]
}
# To match up with int conversions in V1
# PIPE_STR = {
#     "X42": {"SMYS": 290, "SMTS": 415},
#     "X52": {"SMYS": 360, "SMTS": 460},
#     "X56": {"SMYS": 390, "SMTS": 490},
#     "X60": {"SMYS": 415, "SMTS": 520},
#     "X65": {"SMYS": 450, "SMTS": 535},
#     "X70": {"SMYS": 485, "SMTS": 570},
# }
_STEEL_GRADE_CODES = ["X"]


_DESIGN_OPTIONS = Literal["a", "b", "no fracture criterion", "nfc"]


@dataclass
class ASME_consts:
    location_class: int = None
    T_rating: int = None
    joint_factor: int = None


def get_pipe_grades() -> list:
    """
    Get all available pipe grades
    """
    return [val for val in PIPE_STR.keys() if val not in ["B", "X46"]]


def get_SMYS_SMTS(grade: str) -> tuple:
    """
    Gets specified minimum yield and tensile strength based ion grade
    """
    if grade not in PIPE_STR:
        raise ValueError(f"Grade {grade} is not a valid value")
    return PIPE_STR[grade]["SMYS"], PIPE_STR[grade]["SMTS"]


def get_design_factor(design_option: _DESIGN_OPTIONS, location_class: int) -> float:
    """
    Returns design factor based on location
    """

    if location_class not in [1, 2, 3, 4]:
        raise ValueError(
            f"{location_class} is not a valid location class (must be 1, 2, 3, or 4)"
        )
    # Use ASME if design option is preselected
    if isinstance(design_option, str):
        if design_option == "a":
            design_factor_list = [0.5, 0.5, 0.5, 0.4]
        elif design_option == "b":
            design_factor_list = [0.72, 0.60, 0.50, 0.40]
        elif design_option in [
            "nfc",
            "no fracture criterion",
        ]:
            design_factor_list = [0.4, 0.4, 0.4, 0.4]

        return design_factor_list[location_class - 1]
    # Return %SMYS if design option is a number
    # It should already be case to a float
    elif isinstance(design_option, float):
        return design_option


def get_material_performance_factor(
    design_option: _DESIGN_OPTIONS, design_p_MPa: float, SMYS: float, SMTS: float
) -> float:
    """
    Calculate material performance factor. Based on ASME code case, always will return 1
    """

    # Set material performance factor to 1 based on ASME Code Case
    if design_option != "a":
        return 1

    # if design_option_formatted in ["b", "no fracture criterion", "nfc", "a"]:
    #     return 1  # Set material performance factor to 1

    # # Design pressures in ASME B31.12 Table IX-5A in MPa (converted from psi)
    design_pressures = np.array(
        [6.8948, 13.7895, 15.685, 16.5474, 17.9264, 19.3053, 20.6843]
    )

    # # B31.12 Table IX-5A for stresses and design pressure in MPa
    h_f_array = get_hf_array(SMYS=SMYS, SMTS=SMTS)

    return np.interp(design_p_MPa, design_pressures, h_f_array)


def get_hf_array(SMYS: float, SMTS: float) -> list:
    """
    Create arrays of material performance factors based on SMYS and SMTS for ASME
    B31.12 Table IX-5A for stresses and design pressure in MPa
    """

    SMTS_limits_ksi = np.array([0, 66, 75, 82, 90])
    SMYS_limits_ksi = np.array([52, 60, 70, 80])
    SMTS_limits_MPa = SMTS_limits_ksi / MPA2KPSI
    SMYS_limits_MPa = SMYS_limits_ksi / MPA2KPSI

    h_f_arrays = np.array(
        [
            [1, 1, 0.954, 0.91, 0.88, 0.84, 0.78],
            [0.874, 0.874, 0.834, 0.796, 0.77, 0.734, 0.682],
            [0.776, 0.776, 0.742, 0.706, 0.684, 0.652, 0.606],
            [0.694, 0.694, 0.662, 0.632, 0.61, 0.584, 0.542],
        ]
    )

    for i in range(len(SMYS_limits_MPa)):
        if (
            SMYS <= SMYS_limits_MPa[i]
            and SMTS_limits_MPa[i] < SMTS <= SMTS_limits_MPa[i + 1]
        ):
            return h_f_arrays[i]

    raise ValueError(
        f"SMYS: {SMYS} MPa and SMTS: {SMTS} MPa don't line up with values in ASME B31.12 Table IX-5A"
    )


def get_design_pressure_ASME(
    design_p_MPa: float,
    design_option: _DESIGN_OPTIONS,
    SMYS: float,
    SMTS: float,
    t: float,
    D: float,
    F: float,
    E: float,
    T: float,
) -> float:
    """
    Calculate ASME B31.12 design pressure
    """
    design_pressure = design_p_MPa
    error = 1
    while error > 0.001:
        Hf = get_material_performance_factor(
            design_option=design_option, design_p_MPa=design_p_MPa, SMYS=SMYS, SMTS=SMTS
        )
        design_pressure_revised = design_eqn_asme_b31_12(
            S=SMYS, t=t, D=D, F=F, E=E, T=T, Hf=Hf
        )
        error = (
            abs(design_pressure - design_pressure_revised)
            / design_pressure_revised
            * 100
        )

        design_pressure = design_pressure_revised

    return design_pressure


def get_th_from_ASME(
    design_p_MPa: float,
    design_option: _DESIGN_OPTIONS,
    SMYS: float,
    SMTS: float,
    D: float,
    F: float,
    E: float,
    T: float,
) -> float:
    """
    Calculate thickness based on ASME B31.12 design pressure
    """
    Hf = get_material_performance_factor(
        design_option=design_option, design_p_MPa=design_p_MPa, SMYS=SMYS, SMTS=SMTS
    )
    th = design_p_MPa / design_eqn_asme_b31_12(S=SMYS, t=1, D=D, F=F, E=E, T=T, Hf=Hf)

    return th


def design_eqn_asme_b31_12(
    S: float, t: float, D: float, F: float, E: float, T: float, Hf: float
) -> float:
    """
    ASME B31.12 design eqn
    """
    return 2 * S * t / D * F * E * T * Hf


def get_viable_schedules(
    sch_list: dict,
    design_option: _DESIGN_OPTIONS,
    ASME_params: ASME_consts,
    grade: str,
    p_max_mpa_g: float,
    design_pressure_MPa: float,
    DN: float,
) -> tuple:
    """
    Get schedules and thicknesses that satisfy ASME B31.12
    """
    th_vals = np.array([*sch_list.values()])
    sch_vals = np.array([*sch_list.keys()])

    # Get design factor
    design_factor = get_design_factor(
        design_option=design_option, location_class=ASME_params.location_class
    )

    th_vals_valid = []
    sch_vals_valid = []
    pressure_valid = []
    closest_th_index = -1

    # Get SMYS and SMTS based on grade
    SMYS, SMTS = get_SMYS_SMTS(grade)
    # only get minimum th if there is a design pressure
    if p_max_mpa_g is None:
        th = 0
    else:
        th = get_th_from_ASME(
            design_p_MPa=p_max_mpa_g,
            design_option=design_option,
            SMYS=SMYS,
            SMTS=SMTS,
            D=DN,
            F=design_factor,
            E=ASME_params.joint_factor,
            T=ASME_params.T_rating,
        )
    # Check if availalbe thickness are >= to required thickness
    if max(th_vals) >= th:
        th_vals_valid = th_vals[th_vals >= th]
        sch_vals_valid = sch_vals[th_vals >= th]

        for i in range(len(th_vals_valid)):
            pressure_valid.append(
                get_design_pressure_ASME(
                    design_p_MPa=1 if p_max_mpa_g is None else p_max_mpa_g,
                    design_option=design_option,
                    SMYS=SMYS,
                    SMTS=SMTS,
                    t=th_vals_valid[i],
                    D=DN,
                    F=design_factor,
                    E=ASME_params.joint_factor,
                    T=ASME_params.T_rating,
                )
            )
        # sort according to thickness
        th_vals_valid, sch_vals_valid, pressure_valid = zip(
            *sorted(zip(th_vals_valid, sch_vals_valid, pressure_valid))
        )

        closest_th_index = np.argmin(abs(np.array(th_vals_valid) - th))
    return (th_vals_valid, sch_vals_valid, pressure_valid, closest_th_index)


def get_pipe_mass(volume_m3: float) -> float:
    """
    Calculate pipe mass based on volume and density
    """
    return volume_m3 * gl.STEEL_RHO_KG_M3


def get_pipe_volume(diam_o_m: float, diam_i_m: float, length_m: float) -> float:
    """
    Calculate volume based on inner diameter, outer diameter, and length
    """
    return np.pi / 4 * (diam_o_m**2 - diam_i_m**2) * length_m


def check_design_option(design_option: str | float):
    # Cast to string
    design_option_formatted = str(design_option).lower()
    # If not in preselected options, then filter for numerical value
    if design_option_formatted not in get_args(_DESIGN_OPTIONS):
        try:
            design_option_formatted = float(design_option_formatted)
            if not (0 <= design_option_formatted <= 1):
                raise ValueError()

        except ValueError:
            raise ValueError(f"{design_option} is not a valid design option")

    return design_option_formatted
