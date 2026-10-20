import csv
import json
from dataclasses import dataclass, field
from os.path import exists
from typing import TYPE_CHECKING, Any

import numpy as np
from importlib_resources import files

import BlendPATH.costing.pipe_costs.steel_pipe_costs
import BlendPATH.costing.profast_util
import BlendPATH.Global as gl

if TYPE_CHECKING:
    from BlendPATH.modifications.mod_util import Mod_costing_params
    from BlendPATH.network.pipeline_components import Composition

COMPR_COSTS = {
    "Material": [3175286, 532.7853, 0.0010416],
    "Labor": [1581740, 299.2887, 0.001142],
    "Misc": [1696686, 184.1443, 0.0018417],
    "Land": [66216.72, 0, 0.0001799],
}


@dataclass
class Costing_params:
    """
    Grouping of cost parameters from BlendPATH_scenario
    """

    h2_price: float = None
    ng_price: float = None
    elec_price: float = None
    region: str = None
    cf_price: float = None
    casestudy_name: str = None
    ili_interval: float = None
    original_pipeline_cost: float = None
    pipe_markup: float = None
    compressor_markup: float = None
    financial_overrides: dict = field(default_factory=dict)
    polymer_cost_type: float = None
    steel_cost: dict = None
    odorant_cost: float = None
    supply_regulator_cost: float = None
    regulator_blend_limit: float = None
    msa_blend_limit: float = None
    valve_blend_limit: float = None
    valve_cost_distribution: float = None
    valve_density: float = None
    valve_replacement_fraction: float = None
    regulator_maintenance_fraction: float = None
    regulator_refurbishment_fraction: float = None


def get_steel_cost_file(steel_cost_file: str) -> dict:
    """
    Retrieve steel cost file
    """
    return BlendPATH.costing.pipe_costs.steel_pipe_costs.get_steel_cost_file(
        steel_cost_file=steel_cost_file
    )


def get_pipe_cost_file(pipe_file: str) -> dict:
    """
    Get override pipe cost file
    """
    return BlendPATH.costing.pipe_costs.steel_pipe_costs.get_pipe_cost_file(
        pipe_file=pipe_file
    )


def get_compressor_cost_file(compressor_file: str) -> dict:
    """
    Get override compressor cost file
    """
    with open(compressor_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return {
            row["Parameter"]: float(row["Price [$/hp]"])
            for row in reader
            if row["Price [$/hp]"] != ""
        }


def calc_lcot(
    mod_costing_params: "Mod_costing_params",
    new_pipe_capex: float,
    costing_params: Costing_params,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Calculate the levelized cost of transport with ProFAST
    """

    pf = BlendPATH.costing.profast_util.profast_init(costing_params=costing_params)
    BlendPATH.costing.profast_util.profast_overrides(
        pf=pf, costing_params=costing_params
    )
    gen_inflation = pf.vals["general inflation rate"]
    BlendPATH.costing.profast_util.set_capacity(
        pf=pf, mod_costing_params=mod_costing_params
    )

    # Add feedstocks to ProFAST (gas, elec compressor fuel)
    BlendPATH.costing.profast_util.pf_add_feedstocks(
        pf=pf,
        mod_feedstocks=mod_costing_params.feedstocks,
        inflation=gen_inflation,
        costing_params=costing_params,
    )

    # Add capital items to ProFAST
    BlendPATH.costing.profast_util.pf_add_capital(
        pf=pf,
        mod_capital=mod_costing_params.capital,
        depr_period=pf.vals["operating life"],
        costing_params=costing_params,
        new_pipe_capex=new_pipe_capex,
    )

    # Add fixed costs to ProFAST
    BlendPATH.costing.profast_util.pf_add_fixed_costs(
        pf=pf, mod_fixed_costs=mod_costing_params.fixed_costs, inflation=gen_inflation
    )

    sol = pf.solve_price()

    prices_all = BlendPATH.costing.profast_util.pf_aggregation(pf=pf, lcot=sol["lco"])

    # Collect ProFAST inputs to recreate
    pf_in = BlendPATH.costing.profast_util.pf_get_inputs(pf=pf)

    return prices_all, pf_in


def get_cs_fuel_cost(
    blend: float,
    ng_cost: float,
    h2_cost: float,
    composition: "Composition",
    json_file: str,
    fin_overrides: dict,
) -> float:
    """
    Calculate the compressor fuel cost in $/MMBTU
    """

    # Get pure H2, NG  HHV,GCV
    H2_energy_HHV, GCV_H2_MJpsm3 = composition.pure_h2_hhv_gcv()
    _, GCV_NG_MJpsm3 = composition.pure_ng_hhv_gcv()

    blend_ratio_energy = (blend * GCV_H2_MJpsm3) / (
        (blend * GCV_H2_MJpsm3) + (1 - blend) * GCV_NG_MJpsm3
    )
    # h2_cost_MMBTU = h2_cost / H2_energy_HHV / gl.MJ2MMBTU

    jsonfilename = f"{json_file}/financial_params.json"
    if not exists(jsonfilename):
        jsonfilename = files("BlendPATH.costing").joinpath(
            "default_financial_params.json"
        )
    with open(jsonfilename) as f:
        d = json.load(f)
    var_list = [
        "general inflation rate",
        "analysis start year",
        "installation months",
        "operating life",
    ]
    fin_params = {
        x: d["variables"][x] if x not in fin_overrides else fin_overrides[x]
        for x in var_list
    }
    make_range = range(
        fin_params["analysis start year"],
        fin_params["analysis start year"]
        + fin_params["operating life"]
        + int(fin_params["installation months"] / 12)
        + 10,
    )
    infl = fin_params["general inflation rate"]

    if isinstance(ng_cost, dict):
        ng_years = set(ng_cost.keys())
    else:
        ng_years = set(make_range)
        ng_cost = {year: ng_cost * (1 + infl) ** i for i, year in enumerate(make_range)}
    if isinstance(h2_cost, dict):
        h2_cost_MMBTU = {
            year: val / H2_energy_HHV / gl.MJ2MMBTU for year, val in h2_cost.items()
        }
        h2_years = set(h2_cost_MMBTU.keys())
    else:
        h2_cost_MMBTU = h2_cost / H2_energy_HHV / gl.MJ2MMBTU
        h2_years = set(make_range)
        h2_cost_MMBTU = {
            year: h2_cost_MMBTU * (1 + infl) ** i for i, year in enumerate(make_range)
        }

    key_intersect = ng_years.union(h2_years)

    ng_price_all = {year: (ng_cost.get(year, 0)) for year in key_intersect}
    h2_price_all = {year: (h2_cost_MMBTU.get(year, 0)) for year in key_intersect}

    def cs_func(ng, h2, b):
        return ng * (1 - b) + h2 * b

    cs_cost = {
        key: cs_func(ng_price_all[key], h2_price_all[key], blend_ratio_energy)
        for key in key_intersect
    }

    return cs_cost


def get_compressor_cost(
    cp: Costing_params,
    avg_station_cap: float,
    num_comp_stations: float,
    revamped_ratio: float = 1,
) -> float:
    """
    Compressor cost correlation
    """
    cols = ["Material", "Labor", "Misc", "Land"]
    rui_amsymptote = {"Material": 694.09, "Labor": 400.24, "Misc": 306.65, "Land": 7.88}
    comp_costs = {x: 0 for x in cols}
    comp_overrides = cp.comp_cost_override or {}
    for cost_type in cols:
        if cost_type in comp_overrides:
            comp_costs[cost_type] = avg_station_cap * comp_overrides[cost_type]
        else:
            # Rui ampytotes to 1400 $/hp above 30,000 hp
            if avg_station_cap > 30_000:
                comp_costs[cost_type] = rui_amsymptote[cost_type] * avg_station_cap
            else:
                station_cap_array = [avg_station_cap**x for x in [0, 1, 2]]
                comp_costs[cost_type] = np.dot(
                    COMPR_COSTS[cost_type], station_cap_array
                )
            CPI_ratio = 596.2 / 575.4  # Go from 2008 to 2020
            comp_costs[cost_type] *= CPI_ratio

    if revamped_ratio != 1:
        comp_costs[cols.index("Land")] = 0

    comp_cost_i = sum(comp_costs.values()) * revamped_ratio * num_comp_stations

    return comp_cost_i * cp.compressor_markup


def meter_reg_station_cost(cp: Costing_params, demands_MW: list) -> float:
    """
    Returns meter, regulator, and GC modification costs
    """

    demand_mmbtu = np.array(demands_MW) * gl.MW2MMBTUDAY

    cost_meters = meter_replacement_cost(cp=cp, demand_mmbtu=demand_mmbtu)
    cost_gc = GC_cost(cp=cp, demand_mmbtu=demand_mmbtu)
    cost_reg = regulator_cost(cp=cp, demand_mmbtu=demand_mmbtu)

    return cost_meters + cost_gc + cost_reg


def meter_replacement_cost_file(meter_cost_file: str) -> tuple:
    """
    Retrieve default meter cost file
    """
    with open(meter_cost_file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        for row in reader:
            m = float(row["m [2020$/MMBTU-day]"])
            b = float(row["b [2020$]"])
            return (m, b)


def meter_replacement_cost(cp: Costing_params, demand_mmbtu: list) -> float:
    """
    Get meter replacement costs
    """
    m, b = cp.meter_cost

    return np.sum(m * demand_mmbtu + b)


def GC_cost_file(gc_cost_file) -> float:
    """
    Read in GC cost file
    """
    with open(gc_cost_file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        for row in reader:
            installed_cost = float(row["Installed cost [2020$]"])
            return installed_cost


def GC_cost(cp: Costing_params, demand_mmbtu: list) -> float:
    """
    Get gas chromatography modifications/addition cost

    """
    installed_cost = cp.gc_cost  # [2020$]

    return installed_cost * len(demand_mmbtu)


def regulator_cost_file(regulator_cost_file: str) -> tuple:
    """
    Retrieve default regulator cost file
    """
    with open(regulator_cost_file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        for row in reader:
            capacity_ref = float(row["Capacity [MMBTU/day]"])
            installed_cost_ref = float(row["Installed regulator cost [2020$]"])
            return (capacity_ref, installed_cost_ref)


def regulator_cost(cp: Costing_params, demand_mmbtu: list) -> float:
    """
    Installed pressure regulator costs

    """
    capacity_ref = 311400  # [MMBTU/day]
    installed_cost_ref = 2248722  # [2020$]
    capacity_ref, installed_cost_ref = cp.regulator_cost

    regulator_runs = np.ceil(demand_mmbtu / capacity_ref)
    replacement_cost = regulator_runs * installed_cost_ref

    return sum(replacement_cost)


def ili_cost(cp: Costing_params, pipe_added: list) -> float:
    """
    ILI costs from tabulated values
    """

    ili_cost = sum(
        [ili_check_interp(cp=cp, dn=dn) * length for dn, length in pipe_added]
    )

    return ili_cost / cp.ili_interval


def ili_check_interp(cp: Costing_params, dn: int) -> float:
    if dn not in cp.ili_cost:
        dns = np.array(list(cp.ili_cost.keys()))
        mask = dns > dn
        index = np.argmin(abs(dns[mask] - dn))
        return cp.ili_cost[dns[mask][index]]
    return cp.ili_cost[dn]


def ili_costs_file(ili_cost_file: str) -> dict:
    """
    Import ILI costs file
    """

    with open(ili_cost_file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        ili_dict = {int(x["DN"]): float(x["ILI cost [2020$/mi]"]) for x in reader}
    return ili_dict


def inline_regulator_costs_file(inline_regulator_cost_file: str) -> dict[float, float]:
    with open(inline_regulator_cost_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        inline_reg_dict = {
            float(x["regulator_pressure_limit_bar"]): float(x["regulator_cost"])
            for x in reader
        }
    return inline_reg_dict


def valve_replacement_cost_file(file: str) -> dict:
    """
    Get valve replacement costs:
    """
    with open(file, newline="") as csvfile:
        next(csvfile)
        reader = csv.DictReader(csvfile)
        buried = {}
        above = {}
        for row in reader:
            if row["Install type"] == "Buried":
                buried[int(row["DN"])] = int(row["Installed valve cost [2020$]"])
            elif row["Install type"] == "Above ground, welded":
                above[int(row["DN"])] = int(row["Installed valve cost [2020$]"])
    return {"buried": buried, "above": above}


def valve_replacement_cost(
    cp: Costing_params, pipe_added: list, loc_class: int, v_type: str = "buried"
) -> float:
    ASMEB3112_valve_spacing = {1: 20, 2: 15, 3: 10, 4: 5}
    if loc_class not in ASMEB3112_valve_spacing:
        raise ValueError(
            f"Location class value is not valid. {loc_class} was given. Must be in {ASMEB3112_valve_spacing.keys()}"
        )
    spacing = ASMEB3112_valve_spacing[loc_class]
    if v_type not in ["buried"]:
        raise ValueError(f"Valve type must be 'buried', {v_type} was given")
    ref_data = cp.valve_cost[v_type]

    valve_cost = sum(
        [
            np.floor(length / spacing)
            * valve_cost_check_interp(ref_data=ref_data, dn=dn)
            for dn, length in pipe_added
        ]
    )

    return valve_cost


def valve_cost_check_interp(ref_data: dict[int, float], dn: int) -> float:
    if dn not in ref_data:
        dns = np.array(list(ref_data.keys()))
        mask = dns > dn
        index = np.argmin(abs(dns[mask] - dn))
        return ref_data[dns[mask][index]]
    return ref_data[dn]
