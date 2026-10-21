import copy
import itertools
import math
import os
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import cantera as ct
import numpy as np

import BlendPATH.costing.costing as bp_cost
import BlendPATH.costing.distribution_network_nonpipe_costs as bp_dist_costs
import BlendPATH.costing.pipe_costs.polymer_pipe_costs as bp_p_pipe_cost
import BlendPATH.costing.pipe_costs.steel_pipe_costs as bp_pipe_cost
import BlendPATH.Global as gl
import BlendPATH.network.pipeline_components as bp_plc
import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.costing.profast_util import ProFAST_name_capital
from BlendPATH.modifications import (
    additional_compressors,
    direct_replacement,
    direct_replacement_distribution,
    new_h2_pipeline,
    parallel_loop,
    parallel_loop_distribution,
)
from BlendPATH.network import BlendPATH_network
from BlendPATH.scenario_helper import SCENARIO_VALUES, Design_params, Scenario_type
from BlendPATH.util import plastic_pipe
from BlendPATH.util.pipe_dimensions import _DN_NPS_MAP, _NPS_DN_MAP
from BlendPATH.util.schedules import SCHEDULES_PE_ID, SCHEDULES_STEEL_ID

NPS_DIAM_LIST = np.asarray(list(SCHEDULES_PE_ID.keys()))
DN_DIAM_LIST = np.asarray(list(SCHEDULES_STEEL_ID.keys()))

_MOD_TYPES = Literal[
    "direct_replacement",
    "parallel_loop",
    "additional_compressors",
    "dr",
    "pl",
    "ac",
    "new_h2_pipeline",
    "newh2",
]


@dataclass
class Mod_result:
    new_pipes: dict
    cap_cost: dict
    pipe_cost_types: list
    l_comps: list
    mod_type: str
    original_sm3_s: float

    def __post_init__(self):
        self.n_comps = [len(x) for x in self.l_comps]


@dataclass
class Mod_fixed_costs:
    ili_costs: float = 0
    odorization: float = 0
    regulator_maintenance: float = 0
    leak_survey: float = 0


@dataclass
class Mod_capital:
    pipe_capex: float = 0
    comp_capex: list = field(default_factory=list)
    revamped_comp_capex: list = field(default_factory=list)
    supply_comp_capex: float = 0
    meter_cost: float = 0
    valve_cost: float = 0
    msa_cost: float = 0
    regulator_refurbishment: float = 0
    supply_regulator_refurbishment: float = 0

    def network_costs(self):
        return {
            ProFAST_name_capital.NEW_PIPE: self.pipe_capex,
            ProFAST_name_capital.NEW_COMPR: sum(self.comp_capex),
            ProFAST_name_capital.REFURB_COMPR: sum(self.revamped_comp_capex),
            ProFAST_name_capital.SUPPLY_COMPR: self.supply_comp_capex,
            ProFAST_name_capital.METER_TRANS: self.meter_cost,
            ProFAST_name_capital.VALVE: self.valve_cost,
            ProFAST_name_capital.REG_REFURB: self.regulator_refurbishment,
            ProFAST_name_capital.SUPPLY_REG: self.supply_regulator_refurbishment,
        }

    def user_costs(self):
        return {
            ProFAST_name_capital.MSA_DIST: self.msa_cost,
        }


@dataclass
class Mod_feedstocks:
    all_fuel_usage: float = 0
    all_fuel_elec: float = 0
    supply_comp_fuel: dict[str, float] = field(
        default_factory=lambda: {"gas": 0, "elec": 0}
    )
    all_fuel_MW: float = 0
    all_fuel_elec_kW: float = 0


@dataclass
class Mod_costing_params:
    capacity: float
    capital: Mod_capital
    fixed_costs: Mod_fixed_costs
    feedstocks: Mod_feedstocks


@dataclass
class Add_supply_comp_inputs:
    blend: float
    composition: bp_plc.Composition
    orig_supply_pressure: float
    pressure: float
    thermo_curvefit: bool
    m_dot_seg: float


@dataclass
class New_pipes:
    grade: str = ""
    cost: float = 0.0
    th: float = 0.0
    schedule: str = ""
    pressure: float = 0.0
    inner_diameter: float = 0.0
    dn: float = 0.0
    nps: float = 0.0
    length: float = 0.0
    ps: float = -1.0
    name: str = ""
    from_node: str = ""
    to_node: str = ""
    roughness_mm: float = 0.0

    @property
    def D_S_G(self):
        return f"{self.dn};;{self.schedule};;{self.grade}"


def check_mod_method(mod_type: str) -> _MOD_TYPES:
    """
    Check if the modification method is valid
    """
    mod_type_filter = mod_type.lower().replace(" ", "_")
    if mod_type_filter not in get_args(_MOD_TYPES):
        raise ValueError(
            f"Modification type must be one {get_args(_MOD_TYPES)}, instead {mod_type} was given"
        )
    return mod_method_hash_map(mod_type_filter)


def mod_method_hash_map(mod_type: _MOD_TYPES) -> str:
    """
    Shorten to 2 letter modification type
    """
    mod_type_shorten = {
        "direct_replacement": "dr",
        "dr": "dr",
        "parallel_loop": "pl",
        "pl": "pl",
        "additional_compressors": "ac",
        "ac": "ac",
        "new_h2_pipeline": "newh2",
        "newh2": "newh2",
    }
    return mod_type_shorten[mod_type]


def mod_network_file_name(
    casestudy_name: str,
    results_dir: str,
    mod_type: str,
    blend: float,
    design_option: float,
    filename_suffix: str,
) -> str:
    """
    Check if dir exists. Returns path to network file
    """
    # Check if results file exists
    full_results_dir = f"{casestudy_name}/{results_dir}"
    network_dir = f"{full_results_dir}/NetworkFiles"
    for filepaths in [
        f"{full_results_dir}",
        network_dir,
        f"{full_results_dir}/ResultFiles",
    ]:
        if not os.path.exists(filepaths):
            os.makedirs(filepaths)

    new_filename = f"{mod_type.upper()}_{blend}_{design_option}_"
    return f"{network_dir}/{new_filename}network_design{filename_suffix}.xlsx"


def mod_handler(
    mod_type: _MOD_TYPES,
    network: BlendPATH_network,
    design_option: str | float,
    new_filename_full: str,
    design_params: Design_params,
    costing_params: bp_cost.Costing_params,
    option: Any = None,
    allow_compressor_bypass: bool = False,
    original_sm3_s: float = 0,
) -> Mod_result:
    if network.scenario_type == Scenario_type.TRANSMISSION:
        mod_type_handler = {
            "dr": direct_replacement.direct_replacement,
            "pl": parallel_loop.parallel_loop,
            "ac": additional_compressors.additional_compressors,
            "newh2": new_h2_pipeline.new_h2_pipeline,
        }
    elif network.scenario_type == Scenario_type.DISTRIBUTION:
        mod_type_handler = {
            "dr": direct_replacement_distribution.direct_replacement_distribution,
            "pl": parallel_loop_distribution.parallel_loop_distribution,
        }
        if option in [2, "2"]:
            mod_type_handler = {
                "dr": direct_replacement_distribution.direct_replacement_distribution_alt,
                "pl": parallel_loop_distribution.parallel_loop_distribution_alt,
            }
    if mod_type not in mod_type_handler:
        raise RuntimeError(
            f"Unknown modification method {mod_type} for scenario_type {network.scenario_type}"
        )
    all_new_pipes, cap_cost, l_comps = mod_type_handler[mod_type](
        network=network,
        design_params=design_params,
        design_option=design_option,
        new_filename=new_filename_full,
        costing_params=costing_params,
        allow_compressor_bypass=allow_compressor_bypass,
    )

    # Aggregate pipe capex (some economies of scale)
    pipe_cost_types = get_pipe_cost_types(
        mod_type=mod_type, scenario_type=network.scenario_type
    )
    cap_cost["total cost"] = bp_pipe_cost.get_aggr_pipe_cost(
        cap_cost, pipe_cost_types=pipe_cost_types, costing_params=costing_params
    )

    return Mod_result(
        new_pipes=all_new_pipes,
        cap_cost=cap_cost,
        pipe_cost_types=pipe_cost_types,
        l_comps=l_comps,
        mod_type=mod_type,
        original_sm3_s=original_sm3_s,
    )


def get_pipe_cost_types(mod_type: _MOD_TYPES, scenario_type: str) -> list:
    """
    If adding pipes, what additional costs are needed (excluding material)
    """
    if scenario_type == Scenario_type.TRANSMISSION:
        if mod_type == "dr":
            return ["Labor", "Misc"]
        elif mod_type == "pl":
            return ["Labor", "Misc", "ROW"]
        elif mod_type == "ac":
            return []
        elif mod_type == "newh2":
            return ["Labor", "Misc", "ROW"]
    elif scenario_type == Scenario_type.DISTRIBUTION:
        return []

    raise RuntimeError(
        f"Unknown modification method {mod_type} for scenario type {scenario_type}"
    )


def generate_modified_network(
    new_filename_full: str,
    composition_tracking: bool,
    thermo_curvefit: bool,
    eos: str,
    ff_type: str,
    scenario_type: str,
) -> BlendPATH_network:
    """
    Import modified network and attempt to solve
    """
    new_network = BlendPATH_network.import_from_file(
        new_filename_full,
        composition_tracking=composition_tracking,
        thermo_curvefit=thermo_curvefit,
        eos=eos,
        ff_type=ff_type,
        scenario_type=scenario_type,
    )

    iters = 0
    while iters < 5:
        try:
            new_network.solve(
                c_relax=SCENARIO_VALUES[scenario_type].c_relax * 1.1**iters,
                low_p_buffer=0.01,
            )
            break
        except (ValueError, ct.CanteraError):
            iters += 1
            continue
    else:
        raise ValueError("Could not solve new network")

    return new_network


def get_mod_costing_params_distribution(
    original_network: BlendPATH_network,
    network: BlendPATH_network,
    design_params: Design_params,
    costing_params: bp_cost.Costing_params,
    mod_result: Mod_result | None,
    supply_comp_inputs: None | Add_supply_comp_inputs = None,
) -> Mod_costing_params:

    blend = network.composition.x.get("H2", 0)
    new_pipe_length = sum([p.length for p in mod_result.new_pipes.values()])
    total_pipe_length = sum([p.length_km for p in network.pipes.values()])

    # Get all demands
    demands_cfh = []
    demand_pressures_psi_g = []
    for demand_node in network.demand_nodes.values():
        blend_limit = (
            original_network.demand_nodes[demand_node.name].blend_limit
            if original_network.demand_nodes[demand_node.name].blend_limit is not None
            else costing_params.msa_blend_limit
        )
        if blend < blend_limit:
            continue
        demands_cfh.append(
            demand_node.flowrate_mdot / demand_node.node.rho * gl.M32FT3 * gl.S2HR
        )
        demand_pressures_psi_g.append(demand_node.node.pressure / gl.PSI2PA)
    # Meter set assembly cost
    msa_cost = bp_dist_costs.get_msa_cost(
        flowrate_cfh=np.asarray(demands_cfh),
        pressure_psig=np.full_like(demand_pressures_psi_g, 0.0),
    )

    additional_m3_s = -mod_result.original_sm3_s
    for d in network.demand_nodes.values():
        rho, _ = d.node.composition.get_rho_z(p_gauge_pa=0, x=d.node.x_h2, mw=d.node.mw)
        additional_m3_s += d.flowrate_mdot / rho

    odorization = bp_dist_costs.get_odorization_cost(
        sm3_s=additional_m3_s,
        costing_params=costing_params,
        blend=blend,
    )

    # Regulators
    supply_regulator_refurbishment = bp_dist_costs.get_regulator_refubishment_cost(
        supply_nodes=network.supply_nodes, blend=blend, costing_params=costing_params
    )

    leak_survey = bp_dist_costs.get_leak_survey_cost(
        new_pipe_length=new_pipe_length,
        mod_type=mod_result.mod_type,
    )

    valves = bp_dist_costs.get_valve_cost(
        total_pipe_length=total_pipe_length, costing_params=costing_params, blend=blend
    )
    inline_regulator_refurbishment_cost = (
        bp_dist_costs.get_inline_regulator_refurbishment_cost(
            original_regulators=original_network.regulators,
            regulators=network.regulators,
            blend=blend,
            costing_params=costing_params,
        )
    )
    inline_regulator_maintenance_cost = (
        bp_dist_costs.get_inline_regulator_maintenance_cost(
            regulator_cost=inline_regulator_refurbishment_cost,
            costing_params=costing_params,
        )
    )


    return Mod_costing_params(
        capacity=network.capacity_MMBTU_day,
        feedstocks=Mod_feedstocks(),
        fixed_costs=Mod_fixed_costs(
            odorization=odorization,
            regulator_maintenance=inline_regulator_maintenance_cost,
            leak_survey=leak_survey,
        ),
        capital=Mod_capital(
            pipe_capex=mod_result.cap_cost["total cost"],
            valve_cost=valves,
            msa_cost=msa_cost,
            regulator_refurbishment=inline_regulator_refurbishment_cost,
            supply_regulator_refurbishment=supply_regulator_refurbishment,
        ),
    )


def get_mod_costing_params_transmission(
    original_network: BlendPATH_network,
    network: BlendPATH_network,
    design_params: Design_params,
    costing_params: bp_cost.Costing_params,
    mod_result: Mod_result | None = None,
    supply_comp_inputs: None | Add_supply_comp_inputs = None,
) -> Mod_costing_params:
    capacity = network.capacity_MMBTU_day

    # If need to make supply compressor
    if supply_comp_inputs is not None:
        from_node = bp_plc.Node(
            name="comp_to_node",
            x_h2=supply_comp_inputs.blend,
            composition=supply_comp_inputs.composition,
            pressure=supply_comp_inputs.orig_supply_pressure * gl.MPA2PA,
            is_supply=True,
        )
        supply_comp = bp_plc.Compressor(
            name="Supply compressor",
            from_node=from_node,
            to_node=bp_plc.Node(
                name="comp_to_node",
                composition=supply_comp_inputs.composition,
            ),
            pressure_out_mpa_g=supply_comp_inputs.pressure,
            fuel_extract=not design_params.new_comp_elec,
        )
        comp_h_1, comp_s_1, comp_h_2_s = network.get_h_s_s(
            x=network.composition.x,
            p1=np.array([supply_comp.from_node.pressure]),
            p2=np.array([supply_comp.to_node.pressure]),
        )
        supply_comp.get_fuel_use(
            h_1=comp_h_1[0],
            s_1=comp_s_1[0],
            h_2_s=comp_h_2_s[0],
            m_dot=supply_comp_inputs.m_dot_seg,
        )

        network.compressors["Supply compressor"] = supply_comp

    # Get compressor aggregate values
    (
        all_fuel_MW,
        all_fuel_elec_kW,
        supply_comp_fuel,
        comp_capex,
        revamped_comp_capex,
        supply_comp_capex,
    ) = network.get_compressor_agg_costs(
        design_params=design_params, costing_params=costing_params
    )

    # Organize all pipe for ILI and valve costs. Aggregate on DN
    all_pipes_len = []
    pipe_dns_lens = {}
    for pipe in network.pipes.values():
        if pipe.DN in pipe_dns_lens:
            pipe_dns_lens[pipe.DN] += pipe.length_km
        else:
            pipe_dns_lens[pipe.DN] = pipe.length_km
    all_pipes_len = [(dn, length) for dn, length in pipe_dns_lens.items()]

    # Get all demands
    demands_MW = [demand.flowrate_MW for demand in network.demand_nodes.values()]

    # Get meter,ILI,valve costs
    meter_cost = bp_cost.meter_reg_station_cost(
        cp=costing_params, demands_MW=demands_MW
    )
    ili_costs = bp_cost.ili_cost(cp=costing_params, pipe_added=all_pipes_len)
    valve_cost = bp_cost.valve_replacement_cost(
        costing_params, all_pipes_len, design_params.asme.location_class
    )

    # Normalize by capacity
    all_fuel_usage = all_fuel_MW * gl.MW2MMBTUDAY / capacity  # MMBTU/MMBTU

    # Get fuel usage rate (electric)
    all_fuel_elec = all_fuel_elec_kW / capacity * gl.DAY2HR

    # Get fuel usage rate for supply compressor
    supply_comp_fuel["gas"] = supply_comp_fuel["gas"] * gl.MW2MMBTUDAY / capacity
    supply_comp_fuel["elec"] = supply_comp_fuel["elec"] / capacity * gl.DAY2HR

    return Mod_costing_params(
        capacity=network.capacity_MMBTU_day,
        feedstocks=Mod_feedstocks(
            all_fuel_usage=all_fuel_usage,
            all_fuel_elec=all_fuel_elec,
            supply_comp_fuel=supply_comp_fuel,
            all_fuel_MW=all_fuel_MW,
            all_fuel_elec_kW=all_fuel_elec_kW,
        ),
        fixed_costs=Mod_fixed_costs(
            ili_costs=ili_costs,
        ),
        capital=Mod_capital(
            # pipe_capex=mod_result.cap_cost["total cost"],
            comp_capex=comp_capex,
            revamped_comp_capex=revamped_comp_capex,
            supply_comp_capex=supply_comp_capex,
            meter_cost=meter_cost,
            valve_cost=valve_cost,
        ),
    )


mod_costing_handler = {
    Scenario_type.DISTRIBUTION: get_mod_costing_params_distribution,
    Scenario_type.TRANSMISSION: get_mod_costing_params_transmission,
}


def get_mod_costing_params(
    original_network: BlendPATH_network,
    network: BlendPATH_network,
    design_params: Design_params,
    costing_params: bp_cost.Costing_params,
    mod_result: Mod_result | None = None,
    supply_comp_inputs: None | Add_supply_comp_inputs = None,
):
    if network.scenario_type not in mod_costing_handler:
        raise ValueError(f"Unknown scenario type {network.scenario_type}")
    return mod_costing_handler[network.scenario_type](
        original_network=original_network,
        network=network,
        design_params=design_params,
        costing_params=costing_params,
        supply_comp_inputs=supply_comp_inputs,
        mod_result=mod_result,
    )


def compressor_eta(design_params: Design_params):
    assign_eta_s = (
        design_params.new_comp_eta_s_elec
        if design_params.new_comp_elec
        else design_params.new_comp_eta_s
    )
    assign_eta_driver = (
        design_params.new_comp_eta_driver_elec
        if design_params.new_comp_elec
        else design_params.new_comp_eta_driver
    )
    return assign_eta_s, assign_eta_driver


def check_velocity_violation(pipe: bp_plc.Pipe, limit):
    """
    Function to determine if a pipe is violating any velocity constraints
    Returns False if violation exists. otherwise True
    """
    # This is a calculation, so save the value
    max_pipe_velocity = pipe.v_max

    return not any(
        (
            max_pipe_velocity > limit,
            max_pipe_velocity > pipe.erosional_velocity_ASME,
            pipe.velocity_limit is not None and max_pipe_velocity > pipe.velocity_limit,
        )
    )


def get_polymer_combined_pipe_stats(
    updated_pipes: dict, costing_params: bp_cost.Costing_params
):
    combined_pipe = {
        x: []
        for x in [
            "D_S_G",
            "DN",
            "NPS",
            "sch",
            "grade",
            "length",
            "mat_cost",
            "other_pipe_cost",
            "total cost",
        ]
    }
    for new_pipe in updated_pipes.values():
        new_pipe.cost = calculate_polymer_pipe_cost(
            cost_type=costing_params.polymer_cost_type,
            nps_diam_in=new_pipe.nps,
            length_km=new_pipe.length,
        )
        if new_pipe.D_S_G in combined_pipe["D_S_G"]:
            dsg_ind = combined_pipe["D_S_G"].index(new_pipe.D_S_G)
            combined_pipe["length"][dsg_ind] += new_pipe.length
            combined_pipe["mat_cost"][dsg_ind] += new_pipe.cost
        else:
            combined_pipe["DN"].append(new_pipe.dn)
            combined_pipe["NPS"].append(new_pipe.nps)
            combined_pipe["sch"].append(new_pipe.schedule)
            combined_pipe["grade"].append(new_pipe.grade)
            combined_pipe["D_S_G"].append(new_pipe.D_S_G)
            combined_pipe["length"].append(new_pipe.length)
            combined_pipe["mat_cost"].append(new_pipe.cost)

    return combined_pipe


def calculate_polymer_pipe_cost(
    cost_type: str, nps_diam_in: float, length_km: float
) -> float:
    """Calculate polymer pipe cost based on cost correlation

    Args:
        cost_type (str): Cost correlation to use (e.g., socal)
        nom_diam_in (float): Nominal diameter [in]
        length (float): Length [km]

    Returns:
        float: Total pipe material cost
    """
    return (
        bp_p_pipe_cost.get_polymer_pipe_cost_per_ft(
            cost_type=cost_type, nps_diam_in=nps_diam_in
        )
        * length_km
        * gl.KM2FT
    )


@dataclass
class pipe_diam_params:
    inner_diameter_mm: float
    nps: float
    dn: float
    pressure_rating_MPa: float
    material: Literal["steel", "polymer"]
    design_option: float
    design_params: Design_params
    grade: str = None
    polymer: str = None
    polymer_designation: str = None


def get_next_diameter_polymer(
    pipe_params_in: pipe_diam_params,
    allow_existing_geom: bool = False,
) -> tuple[float, float, float]:
    """
    Function to get the next largest diameter
    """

    if allow_existing_geom and pipe_params_in.nps in NPS_DIAM_LIST:
        return (pipe_params_in.inner_diameter_mm, 0, 0, 0, 0)

    for next_NPS in range(2):
        # Get index of DN greater than or equal.
        # Increase by i in the case that the diameters in the DN are not larger than the current diameter
        nps_diam_index = np.argmax(NPS_DIAM_LIST >= pipe_params_in.nps) + next_NPS
        if nps_diam_index >= len(NPS_DIAM_LIST):
            raise RuntimeError("Did not find next diameter")
        nps_diam = NPS_DIAM_LIST[nps_diam_index]
        for i, inner_diameter in enumerate(
            SCHEDULES_PE_ID[nps_diam]["diameter_mm"][::-1]
        ):
            if inner_diameter > pipe_params_in.inner_diameter_mm:
                th = SCHEDULES_PE_ID[nps_diam]["thickness_mm"][::-1][i]
                pressure_rating = plastic_pipe.design_pressure_thermoplastic(
                    design_factor=pipe_params_in.design_option,
                    sdr=round(inner_diameter / th, 2),
                    polymer=pipe_params_in.polymer,
                    polymer_designation=pipe_params_in.polymer_designation,
                )
                if pressure_rating >= pipe_params_in.pressure_rating_MPa:
                    return (
                        inner_diameter,
                        nps_diam,
                        _NPS_DN_MAP[nps_diam],
                        th,
                        pressure_rating,
                    )
    raise RuntimeError("Did not find next diameter")


def get_next_diameter_steel(
    pipe_params_in: pipe_diam_params,
    allow_existing_geom: bool = False,
):

    if allow_existing_geom and pipe_params_in.dn in DN_DIAM_LIST:
        return (pipe_params_in.inner_diameter_mm, 0, 0, 0, 0)

    SMYS, SMTS = bp_pa.get_SMYS_SMTS(pipe_params_in.grade)
    design_factor = bp_pa.get_design_factor(
        design_option=pipe_params_in.design_option,
        location_class=pipe_params_in.design_params.asme.location_class,
    )

    for next_DN in range(2):
        # Get index of DN greater than or equal.
        # Increase by i in the case that the diameters in the DN are not larger than the current diameter
        dn_diam_index = np.argmax(DN_DIAM_LIST >= pipe_params_in.dn) + next_DN
        if dn_diam_index >= len(DN_DIAM_LIST):
            raise RuntimeError("Did not find next diameter")
        dn_diam = DN_DIAM_LIST[dn_diam_index]
        for i, inner_diameter in enumerate(
            SCHEDULES_STEEL_ID[dn_diam]["diameter_mm"][::-1]
        ):
            if inner_diameter > pipe_params_in.inner_diameter_mm:
                th = SCHEDULES_STEEL_ID[dn_diam]["thickness_mm"][::-1][i]

                pressure_ASME_MPa = bp_pa.get_design_pressure_ASME(
                    design_p_MPa=pipe_params_in.pressure_rating_MPa,
                    design_option=pipe_params_in.design_option,
                    SMYS=SMYS,
                    SMTS=SMTS,
                    t=th,
                    D=dn_diam,
                    F=design_factor,
                    E=pipe_params_in.design_params.asme.joint_factor,
                    T=pipe_params_in.design_params.asme.T_rating,
                )

                if pressure_ASME_MPa >= pipe_params_in.pressure_rating_MPa:
                    return (
                        inner_diameter,
                        _DN_NPS_MAP[dn_diam],
                        dn_diam,
                        th,
                        pressure_ASME_MPa,
                    )
    raise RuntimeError("Did not find next diameter")


_get_next_diameter_by_type = {
    "steel": get_next_diameter_steel,
    "polymer": get_next_diameter_polymer,
}


def get_next_diameter(
    pipe_params_in: pipe_diam_params,
    allow_existing_geom: bool = False,
) -> tuple[float, float, float]:
    return _get_next_diameter_by_type[pipe_params_in.material](
        pipe_params_in=pipe_params_in, allow_existing_geom=allow_existing_geom
    )


def get_comp_bypass(
    allow_compressor_bypass: bool, nw: BlendPATH_network
) -> list[tuple[int]]:
    """Function to get compressor bypass options

    Args:
        allow_compressor_bypass (bool): If compressor btpass is to be considered
        nw (BlendPATH_network): Copy of the network

    Returns:
        list[tuple[int]]: List of tuples that has the compressors to bypass
    """
    if allow_compressor_bypass:
        return [
            x
            for cc in range(len(nw.compressors) + 1)
            for x in itertools.combinations(range(len(nw.compressors)), cc)
        ]
    return [()]


def copy_network(
    network: BlendPATH_network, design_params: Design_params
) -> BlendPATH_network:
    """Copy network for safe editing and update compressors

    Args:
        network (BlendPATH_network): Original network
        design_params (Design_params): Design params for updating compressor fuel extraction

    Returns:
        BlendPATH_network: Copied network
    """
    # Copy the network
    nw = copy.deepcopy(network)
    # Convert compressors. If not converting to electric, see keep the original choice
    for cs in nw.compressors.values():
        cs.fuel_extract = cs.fuel_extract and not design_params.existing_comp_elec
    return nw


def get_sorted_lengths(
    d_main: list[tuple[float, float]],
    l_loop: float,
    all_mdot: list[float],
    offtakes: list[float],
    l_comps: list[float],
    hhv: float,
    cumulative_offtakes: bool = True,
) -> dict[float, dict[str, str]]:
    """Get ordered list of items segment for making simplified network

    Args:
        d_main (list[tuple[float, float]]): Inner diameter of main pipe [mm]
        l_loop (float): Length of loop [km]
        all_mdot (list[float]): List of mass flow rate offtakes
        offtakes (list[float]): List of offtake lengths
        hhv (float): Higher heating values. Set to 1 if all_mdot is MW instead
        cumulative_offtakes (bool): Flag to take the cumsum of the offtakes

    Returns:
        dict[float, dict[str, str]]: _description_
    """

    # Gather assorted lengths needed
    if len(d_main) > 1:
        diam_lengths = {x[1]: {"val_type": "diam", "diam": x[0]} for x in d_main}
    else:
        diam_lengths = {}
    if l_loop > 0:
        loop_loc = {l_loop: {"val_type": "loop"}}
    else:
        loop_loc = {}
    if cumulative_offtakes:
        offtakes_temp = np.cumsum(offtakes)
    else:
        offtakes_temp = offtakes
    offtake_lengths = {
        x: {"val_type": "offtake", "MW": all_mdot[i] * hhv}
        for i, x in enumerate(offtakes_temp)
    }
    comps_lengths = {x: {"val_type": "comp"} for x in l_comps}

    # Then sort
    all_lengths = dict(
        sorted(
            {
                **diam_lengths,
                **loop_loc,
                **comps_lengths,
                **offtake_lengths,
            }.items()
        )
    )

    all_lengths_list = []
    for i in all_lengths:
        if i in diam_lengths:
            all_lengths_list.append(LengthVal(length_km=i, val_type="diam"))
        if i in loop_loc:
            all_lengths_list.append(LengthVal(length_km=i, val_type="loop"))
        if i in offtake_lengths:
            all_lengths_list.append(
                LengthVal(length_km=i, val_type="offtake", mw=offtake_lengths[i]["MW"])
            )
        if i in comps_lengths:
            all_lengths_list.append(LengthVal(length_km=i, val_type="comp"))

    return all_lengths_list


@dataclass
class LengthVal:
    length_km: float
    val_type: str
    mw: float | None = None


def get_supply_pressure_list(nw: BlendPATH_network, ps: bp_plc.PipeSegment, ps_i: int):
    # Setup the list to loop thru supply pressures
    supp_p_list = [ps.design_pressure_MPa]
    if ps.design_pressure_MPa == ps.design_pressure_MPa:
        return supp_p_list
    # Only relevant for first segment
    if ps_i == 0:
        og_pressure = nw.supply_nodes[next(iter(nw.supply_nodes.keys()))].pressure_mpa

        # This takes integer values between the original
        # pressure and the MAOP
        if og_pressure < ps.design_pressure_MPa:
            supp_p_list = (
                [og_pressure]
                + list(
                    range(
                        math.ceil(og_pressure),
                        math.floor(ps.design_pressure_MPa),
                        1,
                    )
                )
                + [ps.design_pressure_MPa]
            )
    return supp_p_list


def update_offtake_mdot(
    offtake_mdots: list[float], m_dot_in_prev: float
) -> list[float]:
    # Calculate all offtakes -- adds the pipe segment outlet as a
    # offtake if it is not already
    all_mdot = offtake_mdots.copy()
    if (
        len(offtake_mdots) == 0
        or abs(offtake_mdots[-1] - m_dot_in_prev) / offtake_mdots[-1] > 0.01
    ):
        all_mdot.append(m_dot_in_prev)
    else:
        all_mdot[-1] = m_dot_in_prev
    return all_mdot


@dataclass
class Segment_params:
    p_out_target: float
    offtakes_mdot: list[float]
    ps: bp_plc.PipeSegment
    nw: BlendPATH_network
    costing_params: bp_cost.Costing_params
    design_params: Design_params
    seg_compressor_pressure_out: float
    CR_ratio: float
    eta_s: float
    eta_driver: float

    def __post_init__(self):
        self.new_comps_elec = self.design_params.new_comp_elec
        self.comps_elec = self.design_params.existing_comp_elec
        self.design_pressure_MPa = self.ps.design_pressure_MPa
        self.design_pressure_Pa = self.design_pressure_MPa * gl.MPA2PA
        self.roughness_mm = self.ps.pipes[0].roughness_mm
        self.thermo_curvefit = self.nw.thermo_curvefit
        self.seg_compressor = self.ps.comps
        self.eos = self.nw.eos
        self.ff_type = self.nw.ff_type
        self.composition_tracking = self.nw.composition_tracking
        self.ps_nodes = [x.name for x in self.ps.nodes]
        self.d_main_inner = self.ps.diameter
        self.composition = self.nw.composition
        self.l_total = self.ps.length_km
        self.hhv = self.ps.HHV

        self.offtakes_length = self.ps.offtake_lengths.copy()
        if self.offtakes_length[-1] == 0:
            self.offtakes_length = self.offtakes_length[:-1]
        self.offtakes_length = [round(v, 12) for v in self.offtakes_length]
        self.offtakes_length = [round(v, 12) for v in self.offtakes_length]
        self.offtakes_length = [round(v, 12) for v in self.offtakes_length]
