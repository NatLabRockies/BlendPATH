import csv
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, get_args

import numpy as np
from importlib_resources import files

import BlendPATH.costing.costing as bp_cost
import BlendPATH.Global as gl
from BlendPATH import BlendPATH_scenario
from BlendPATH.costing.pipe_costs.anl_pipe_correlations import ANL_COEFS
from BlendPATH.emissions.emissions_params import EmissionsParams
from BlendPATH.util.pipe_assessment import _DESIGN_OPTIONS, ASME_consts

logger = logging.getLogger(__name__)


@dataclass
class Design_params:
    final_outlet_pressure_mpa_g: float | None = None
    max_CR: list | None = None
    existing_comp_elec: bool | None = None
    new_comp_elec: bool | None = None
    new_comp_eta_s: float | None = None
    new_comp_eta_s_elec: float | None = None
    new_comp_eta_driver: float | None = None
    new_comp_eta_driver_elec: float | None = None
    asme: ASME_consts = field(default_factory=lambda: ASME_consts())


@dataclass
class Network_params:
    thermo_curvefit: bool
    composition_tracking: bool
    eos: str
    blend: float


@dataclass
class Scenario_type:
    TRANSMISSION = "transmission"
    options = [TRANSMISSION]


# @dataclass
# class Subcategory_type:
#     ASME = "asme"
#     DESIGN = "design"
#     COSTING = "costing"
#     options = [ASME, DESIGN, COSTING]


@dataclass
class ScenarioValues:
    max_pressure_pa: float
    min_pressure_pa: float
    default_design_option: str | float
    c_relax: float


# Transmission is based on ASME B31.12 upper limit of 200bar and Maximum City Gate Pressure of 20 bar
SCENARIO_VALUES = {
    Scenario_type.TRANSMISSION: ScenarioValues(
        max_pressure_pa=200 * gl.BAR2PA,
        min_pressure_pa=20 * gl.BAR2PA,
        default_design_option="b",
        c_relax=1.5,
    )
}


def process_default_inputs_scenario_type(value: Any) -> Any:
    value = value.lower()
    if value not in Scenario_type.options:
        raise ValueError(f"{value} is not a valid scenario type")
    return value


def process_default_inputs_new_comp_eta_driver_elec(value: Any) -> Any:
    if value is None or value == "":
        return np.nan

    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{value} is not a number")
    if value <= 0:
        raise ValueError(f"{value} is a value that is <=0")
    if value > 1:
        raise ValueError(f"{value} is a value that is >1")
    return value


def process_default_inputs_float_list(value: Any) -> Any:
    try:
        value = float(value)
        return [value]
    except (ValueError, TypeError):
        return value


def process_default_inputs_str_list(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        value = value[0].split(",")
    return value


def process_default_inputs_postive_vals_w_list(value: Any) -> Any:
    if isinstance(value, list):
        for val in value:
            if val <= 0:
                raise ValueError(f"{value} is a value that is <=0")
    return value


def process_default_inputs_regions(value: Any) -> Any:
    value = value.upper()
    if value not in ANL_COEFS:
        raise ValueError(f"{value} is not a valid region")
    return value


def process_default_inputs_design_option(value: Any) -> Any:
    # determine if value is numeric or not
    is_float = False
    try:
        value = float(value)
        is_float = True
    except (ValueError, TypeError):
        is_float = False

    if is_float:
        if not (0 < value <= 1):
            raise ValueError(
                f"Numerical design option must be between 0<value<=1. {value} was given"
            )
        return value
    else:
        value = value.lower()
        if value not in get_args(_DESIGN_OPTIONS):
            raise ValueError(
                f"String design option must be in {get_args(_DESIGN_OPTIONS)}. {value} was given"
            )
        return value


def process_default_inputs_raise_outside_bounds(value: float) -> float:
    if not (0 <= value <= 1):
        raise ValueError(f"Value must be between 0 and 1. {value} was given")
    return value


def process_default_inputs_raise_le_zero(value: float) -> float:
    if value <= 0:
        raise ValueError(f"Value must be greater than zero. {value} was given")
    return value


def process_default_inputs_raise_lt_zero(value: float) -> float:
    if value < 0:
        raise ValueError(f"Value must be greater than zero. {value} was given")
    return value


def process_default_inputs_raise_in_location_class_list(value: int) -> int:
    if value not in [1, 2, 3, 4]:
        raise ValueError(f"Value must be 1, 2, 3, or 4. {value} was given")
    return value


def process_default_inputs_cast_int(value: Any) -> Any:
    """Attempt to cast input to int.

    Args:
        value (Any): Input value coming from file.

    Returns:
        Any: Output value that might be an int if possible.
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return value


def process_default_inputs_cast_float(value: Any) -> Any:
    """Attempt to cast input to float.

    Args:
        value (Any): Input value coming from file.

    Returns:
        Any: Output value that might be a float if possible
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def process_default_inputs_check_truth(val: Any) -> bool:
    """Format true values to boolean.

    Args:
        val (Any): String boolean value.

    Returns:
        bool: Boolean value of input string.
    """
    return val in ["TRUE", "True", "true", 1]


def process_default_inputs_check_list(val: Any) -> list[float]:
    """Format CSV list into Python list

    Args:
        val (Any): String input.

    Returns:
        list[float]: String of values.
    """
    if isinstance(val, list):
        return [float(re.sub(r"[\[\]]", "", str(x))) for x in val]
    return [float(x) for x in re.sub(r"[\[\]]", "", ",".join(val)).split(",")]


@dataclass
class Scenario_Input:
    name: str
    default_val: Any
    get_addl_rows: bool = False
    formatting: list[Callable] = field(
        default_factory=lambda: [process_default_inputs_cast_float]
    )
    subcategory: bool = False


_SCENARIO_INPUTS = {
    "blend": Scenario_Input(
        name="_blend",
        default_val=0.1,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_outside_bounds,
        ],
        subcategory=True,
    ),
    "location_class": Scenario_Input(
        name="location_class",
        default_val=1,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_int,
            process_default_inputs_raise_in_location_class_list,
        ],
    ),
    "T_rating": Scenario_Input(
        name="T_rating",
        default_val=1,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_outside_bounds,
            process_default_inputs_raise_le_zero,
        ],
    ),
    "joint_factor": Scenario_Input(
        name="joint_factor",
        default_val=1,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_outside_bounds,
            process_default_inputs_raise_le_zero,
        ],
    ),
    "design_option": Scenario_Input(
        name="_design_option",
        default_val="b",
        formatting=[process_default_inputs_design_option],
    ),
    "ng_price": Scenario_Input(name="ng_price", default_val=7.39, subcategory=True),
    "h2_price": Scenario_Input(name="h2_price", default_val=4.40756, subcategory=True),
    "elec_price": Scenario_Input(name="elec_price", default_val=0.07, subcategory=True),
    "region": Scenario_Input(
        name="region",
        default_val="GP",
        subcategory=True,
        formatting=[process_default_inputs_regions],
    ),
    "design_CR": Scenario_Input(
        name="design_CR",
        default_val=[1.2, 1.4, 1.6, 1.8, 2.0],
        formatting=[
            process_default_inputs_str_list,
            process_default_inputs_float_list,
            process_default_inputs_check_list,
            process_default_inputs_postive_vals_w_list,
        ],
        get_addl_rows=True,
        subcategory=True,
    ),
    "final_outlet_pressure_mpa_g": Scenario_Input(
        name="final_outlet_pressure_mpa_g",
        default_val=2,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_le_zero,
        ],
    ),
    "results_dir": Scenario_Input(name="results_dir", default_val="out"),
    "eos": Scenario_Input(name="eos", default_val="rk", subcategory=True),
    "ili_interval": Scenario_Input(
        name="ili_interval",
        default_val=3,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_le_zero,
        ],
    ),
    "original_pipeline_cost": Scenario_Input(
        name="original_pipeline_cost",
        default_val=0,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_lt_zero,
        ],
    ),
    "new_compressors_electric": Scenario_Input(
        name="new_compressors_electric",
        default_val=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_check_truth,
        ],
        subcategory=True,
    ),
    "existing_compressors_to_electric": Scenario_Input(
        name="existing_compressors_to_electric",
        default_val=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_check_truth,
        ],
        subcategory=True,
    ),
    "new_comp_eta_s": Scenario_Input(
        name="new_comp_eta_s",
        default_val=0.78,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_le_zero,
            process_default_inputs_raise_outside_bounds,
        ],
    ),
    "new_comp_eta_s_elec": Scenario_Input(
        name="new_comp_eta_s_elec",
        default_val=0.88,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_le_zero,
            process_default_inputs_raise_outside_bounds,
        ],
    ),
    "new_comp_eta_driver": Scenario_Input(
        name="new_comp_eta_driver",
        default_val=0.357,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_le_zero,
            process_default_inputs_raise_outside_bounds,
        ],
    ),
    "new_comp_eta_driver_elec": Scenario_Input(
        name="new_comp_eta_driver_elec",
        default_val=np.nan,
        subcategory=True,
        formatting=[
            process_default_inputs_new_comp_eta_driver_elec,
        ],
    ),
    "pipe_markup": Scenario_Input(
        name="pipe_markup",
        default_val=1,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_lt_zero,
        ],
    ),
    "compressor_markup": Scenario_Input(
        name="compressor_markup",
        default_val=1,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_raise_lt_zero,
        ],
    ),
    "financial_overrides": Scenario_Input(
        name="financial_overrides", default_val={}, subcategory=True
    ),
    "filename_suffix": Scenario_Input(name="filename_suffix", default_val=""),
    "thermo_curvefit": Scenario_Input(
        name="thermo_curvefit",
        default_val=True,
        formatting=[
            process_default_inputs_check_truth,
        ],
        subcategory=True,
    ),
    "composition_tracking": Scenario_Input(
        name="composition_tracking",
        default_val=False,
        formatting=[
            process_default_inputs_check_truth,
        ],
        subcategory=True,
    ),
    "scenario_type": Scenario_Input(
        name="_scenario_type",
        default_val=Scenario_type.TRANSMISSION,
        formatting=[
            process_default_inputs_scenario_type,
        ],
    ),
    # --- emissions inputs (new) ---
    "EF_NG_COMB_gCO2_per_MJ": Scenario_Input(
        name="EF_NG_COMB_gCO2_per_MJ",
        default_val=51.0,
        subcategory=True,
        formatting=[process_default_inputs_cast_float],
    ),
    "EF_NG_UPSTREAM_gCO2_per_MJ": Scenario_Input(
        name="EF_NG_UPSTREAM_gCO2_per_MJ",
        default_val=8.45,
        subcategory=True,
        formatting=[process_default_inputs_cast_float],
    ),
    "EF_H2_UPSTREAM_gCO2_per_MJ": Scenario_Input(
        name="EF_H2_UPSTREAM_gCO2_per_MJ",
        default_val=0.0,
        subcategory=True,
        formatting=[process_default_inputs_cast_float],
    ),
    "EF_ELEC_gCO2_per_MJ": Scenario_Input(
        name="EF_ELEC_gCO2_per_MJ",
        default_val=120.0,
        subcategory=True,
        formatting=[process_default_inputs_cast_float],
    ),
    "EF_FUG_NG_gCO2e_per_gCH4": Scenario_Input(
        name="EF_FUG_NG_gCO2e_per_gCH4",
        default_val=30.0,
        subcategory=True,
        formatting=[process_default_inputs_cast_float],
    ),
    "EF_FUG_H2_gCO2e_per_gH2": Scenario_Input(
        name="EF_FUG_H2_gCO2e_per_gH2",
        default_val=8.0,
        subcategory=True,
        formatting=[process_default_inputs_cast_float],
    ),
    "leak_trans_CH4_m3_per_m3_per_km": Scenario_Input(
        name="leak_trans_CH4_m3_per_m3_per_km",
        default_val=3.33e-6,
        subcategory=True,
        formatting=[process_default_inputs_cast_float],
    ),
    "include_combustion": Scenario_Input(
        name="include_combustion",
        default_val=True,
        subcategory=True,
        formatting=[
            process_default_inputs_cast_float,
            process_default_inputs_check_truth,
        ],
    ),
}


def process_default_inputs(
    casestudy_name: str,
    intialization_inputs: dict[str, Any],
    scenario_instance: BlendPATH_scenario,
) -> tuple[bp_cost.Costing_params, Design_params, Network_params]:
    temp_inputs = {name: val.default_val for name, val in _SCENARIO_INPUTS.items()}

    # Incorporate default file inputs
    temp_inputs = get_default_file_values(
        casestudy_name=casestudy_name, temp_inputs=temp_inputs
    )

    # update with local overrides
    temp_inputs = get_local_kwargs_overrides(
        intialization_inputs=intialization_inputs, temp_inputs=temp_inputs
    )

    # assign to scenario instance and separate out grouped params
    costing_params, design_params, network_params = assign_inputs(
        temp_inputs=temp_inputs, scenario_instance=scenario_instance
    )

    # Add cost overrides
    costing_params = financial_overrides_files(
        casestudy_name=casestudy_name, costing_params=costing_params
    )

    return costing_params, design_params, network_params


def assign_inputs(
    temp_inputs: dict[str, Any], scenario_instance: BlendPATH_scenario
) -> tuple[bp_cost.Costing_params, Design_params, Network_params]:
    # Loop through and assign if they do not have a sub category
    for var, value in _SCENARIO_INPUTS.items():
        if value.subcategory:
            continue
        setattr(scenario_instance, value.name, temp_inputs[var])
        logger.debug(
            f"Setting parameter: {var}; using variable: {value.name}; to value: {temp_inputs[var]}"
        )
    # Assign subcategories
    costing = bp_cost.Costing_params(
        h2_price=temp_inputs["h2_price"],
        ng_price=temp_inputs["ng_price"],
        elec_price=temp_inputs["elec_price"],
        region=temp_inputs["region"],
        cf_price=0,
        casestudy_name=scenario_instance.casestudy_name,
        ili_interval=temp_inputs["ili_interval"],
        original_pipeline_cost=temp_inputs["original_pipeline_cost"],
        pipe_markup=temp_inputs["pipe_markup"],
        compressor_markup=temp_inputs["compressor_markup"],
        financial_overrides=temp_inputs["financial_overrides"],
    )
    design = Design_params(
        final_outlet_pressure_mpa_g=temp_inputs["final_outlet_pressure_mpa_g"],
        max_CR=temp_inputs["design_CR"],
        new_comp_elec=temp_inputs["new_compressors_electric"],
        existing_comp_elec=temp_inputs["existing_compressors_to_electric"],
        new_comp_eta_s=temp_inputs["new_comp_eta_s"],
        new_comp_eta_s_elec=temp_inputs["new_comp_eta_s_elec"],
        new_comp_eta_driver=temp_inputs["new_comp_eta_driver"],
        new_comp_eta_driver_elec=temp_inputs["new_comp_eta_driver_elec"],
        asme=ASME_consts(
            location_class=int(temp_inputs["location_class"]),
            T_rating=temp_inputs["T_rating"],
            joint_factor=temp_inputs["joint_factor"],
        ),
    )

    network_params = Network_params(
        thermo_curvefit=temp_inputs["thermo_curvefit"],
        composition_tracking=temp_inputs["composition_tracking"],
        eos=temp_inputs["eos"],
        blend=temp_inputs["blend"],
    )

    # Emissions params are attached to scenario instance (side-effect),
    # so we don't have to change the return signature used elsewhere.
    scenario_instance.emissions_params = EmissionsParams(
        EF_NG_COMB_gCO2_per_MJ=temp_inputs["EF_NG_COMB_gCO2_per_MJ"],
        EF_NG_UPSTREAM_gCO2_per_MJ=temp_inputs["EF_NG_UPSTREAM_gCO2_per_MJ"],
        EF_H2_UPSTREAM_gCO2_per_MJ=temp_inputs["EF_H2_UPSTREAM_gCO2_per_MJ"],
        EF_ELEC_gCO2_per_MJ=temp_inputs["EF_ELEC_gCO2_per_MJ"],
        EF_FUG_NG_gCO2e_per_gCH4=temp_inputs["EF_FUG_NG_gCO2e_per_gCH4"],
        EF_FUG_H2_gCO2e_per_gH2=temp_inputs["EF_FUG_H2_gCO2e_per_gH2"],
        leak_trans_CH4_m3_per_m3_per_km=temp_inputs["leak_trans_CH4_m3_per_m3_per_km"],
        include_combustion=bool(temp_inputs["include_combustion"]),
    )

    return costing, design, network_params


def get_default_file_values(
    casestudy_name: str, temp_inputs: dict[str, Any]
) -> dict[str, Any]:
    # Check if default inputs file exists. If not, return defaults
    default_inputs_filepath = f"{casestudy_name}/default_inputs.csv"
    if not os.path.exists(default_inputs_filepath):
        return temp_inputs
    # Open file
    with open(default_inputs_filepath, newline="") as csvfile:
        # Skip header
        next(csvfile)

        reader = csv.reader(csvfile)
        # Loop through entries
        # First item in list is parameter, then value
        for row in reader:
            # Check if value parameter
            if row[0] not in _SCENARIO_INPUTS:
                if row[0].strip().lower() in ["verbose"]:
                    logger.warning(
                        "Skipping default_inputs.csv parameter 'verbose' (not used)."
                    )
                    continue
                raise ValueError(f"{row[0]} is not a valid input file parameter")
            # Perform any processing
            temp_inputs[row[0]] = process_inputs_value(row=row)
    return temp_inputs


def get_local_kwargs_overrides(
    intialization_inputs: dict[str, Any], temp_inputs: dict[str, Any]
) -> dict[str, Any]:
    for parameter in intialization_inputs:
        if parameter not in _SCENARIO_INPUTS:
            continue
        val = intialization_inputs[parameter]
        for format_fxn in _SCENARIO_INPUTS[parameter].formatting:
            val = format_fxn(val)
        temp_inputs[parameter] = val
    return temp_inputs


def process_inputs_value(row: list[str]) -> Any:
    """Clean up CSV default inputs file

    Args:
        row (list[str]): CSV row from csv_reader

    Returns:
        Any: Formatted value
    """
    parameter = row[0]
    val = row[1]
    if _SCENARIO_INPUTS[parameter].get_addl_rows:
        val = row[1:]

    for format_fxn in _SCENARIO_INPUTS[parameter].formatting:
        val = format_fxn(val)
    return val


@dataclass
class Overrides_files:
    override_file: str
    processing: Callable
    default_file: str


OVERRIDES_FILES = {
    "valve_cost": Overrides_files(
        override_file="valve_costs.csv",
        processing=bp_cost.valve_replacement_cost_file,
        default_file=files("BlendPATH.costing").joinpath("valve_costs.csv"),
    ),
    "gc_cost": Overrides_files(
        override_file="GC_cost.csv",
        processing=bp_cost.GC_cost_file,
        default_file=files("BlendPATH.costing").joinpath("GC_cost.csv"),
    ),
    "ili_cost": Overrides_files(
        override_file="inline_inspection_costs.csv",
        processing=bp_cost.ili_costs_file,
        default_file=files("BlendPATH.costing").joinpath("inline_inspection_costs.csv"),
    ),
    "regulator_cost": Overrides_files(
        override_file="regulator_costs.csv",
        processing=bp_cost.regulator_cost_file,
        default_file=files("BlendPATH.costing").joinpath("regulator_costs.csv"),
    ),
    "meter_cost": Overrides_files(
        override_file="meter_replacement_cost_regression_parameters.csv",
        processing=bp_cost.meter_replacement_cost_file,
        default_file=files("BlendPATH.costing").joinpath(
            "meter_replacement_cost_regression_parameters.csv"
        ),
    ),
    "steel_cost": Overrides_files(
        override_file="steel_costs_per_kg.csv",
        processing=bp_cost.get_steel_cost_file,
        default_file=files("BlendPATH.costing.pipe_costs").joinpath(
            "steel_costs_per_kg.csv"
        ),
    ),
    "pipe_cost_override": Overrides_files(
        override_file="pipe_cost.csv",
        processing=bp_cost.get_pipe_cost_file,
        default_file=None,
    ),
    "comp_cost_override": Overrides_files(
        override_file="compressor_cost.csv",
        processing=bp_cost.get_compressor_cost_file,
        default_file=None,
    ),
}


def financial_overrides_files(
    casestudy_name: str, costing_params: bp_cost.Costing_params
) -> bp_cost.Costing_params:
    """
    Check if override costing values are empolyed
    """
    # Get overrides directory exists
    overrides_dir = f"{casestudy_name}/overrides"

    for override_attr, override in OVERRIDES_FILES.items():
        file = f"{overrides_dir}/{override.override_file}"
        if not os.path.exists(file):
            file = override.default_file
        if file is not None:
            setattr(costing_params, override_attr, override.processing(file))
        elif override.default_file is None:
            setattr(costing_params, override_attr, {})
    return costing_params
