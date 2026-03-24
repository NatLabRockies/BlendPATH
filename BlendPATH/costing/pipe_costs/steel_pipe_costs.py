import csv
from typing import TYPE_CHECKING

import BlendPATH.Global as gl
import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.costing.pipe_costs.anl_pipe_correlations import ANL_COEFS

if TYPE_CHECKING:
    from BlendPATH.costing.costing import Costing_params


def get_steel_cost_file(steel_cost_file: str) -> dict:
    """
    Retrieve steel cost file
    """
    with open(steel_cost_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return {row["Steel grade"]: float(row["Price [$/kg]"]) for row in reader}


def get_steel_cost(cp: "Costing_params", grade: str, mass: float) -> float:
    """
    Retrive steel cost based on grade
    """
    if grade not in cp.steel_cost.keys():
        raise ValueError(f"{grade} is not a valid steel grade")
    unit_price = cp.steel_cost[grade]
    price = unit_price * mass

    return price


def get_ANL_costs_in_mi(
    diameter_mm: float, length_km: float, region: str, c_type=list
) -> float:
    """
    Get cost correlations from Brown, Reddi, Elgowainy, Int J. Hydrogen Energy, 2022
    """
    d_in = diameter_mm * gl.MM2IN
    l_mi = length_km * gl.KM2MI
    if region not in ANL_COEFS.keys():
        raise ValueError(f"{region} is not a valid region")

    anl_reg = ANL_COEFS[region]

    c = anl_reg[c_type]
    cost_res = c[0] * d_in ** c[1] * l_mi ** c[2]
    return cost_res


def get_pipe_cost_file(pipe_file: str) -> dict:
    """
    Get override pipe cost file
    """
    with open(pipe_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return {
            row["Parameter"]: float(row["Price [$/in/mi]"])
            for row in reader
            if row["Price [$/in/mi]"] != ""
        }


def get_pipe_other_cost(
    cp: "Costing_params", d_mm: float, l_km: float, anl_types: list
) -> dict:
    """
    Get material and other costs for pipes
    """
    all_other_cost = {x: 0 for x in ["Mat"] + anl_types}
    if l_km == 0:
        return all_other_cost
    d_in = d_mm * gl.MM2IN
    l_mi = l_km * gl.KM2MI
    per_in_mi_costs = {x: 0 for x in anl_types}
    for x in per_in_mi_costs.keys():
        if x in cp.pipe_cost_override.keys():
            per_in_mi_costs[x] = cp.pipe_cost_override[x]
        else:
            per_in_mi_costs[x] = get_ANL_costs_in_mi(
                diameter_mm=d_mm, length_km=l_km, region=cp.region, c_type=x
            )
        all_other_cost[x] = per_in_mi_costs[x] * d_in * l_mi * cp.pipe_markup

    return all_other_cost


def get_pipe_material_cost(
    cp: "Costing_params", di_mm: float, do_mm: float, l_km: float, grade: str
) -> float:
    """
    Pipe material cost
    """
    pipe_vol_m3 = bp_pa.get_pipe_volume(
        diam_i_m=di_mm * gl.MM2M,
        diam_o_m=do_mm * gl.MM2M,
        length_m=l_km * gl.KM2M,
    )
    pipe_mass_kg = bp_pa.get_pipe_mass(volume_m3=pipe_vol_m3)
    pipe_mat_cost = get_steel_cost(cp=cp, grade=grade, mass=pipe_mass_kg)

    return pipe_mat_cost * cp.pipe_markup


def get_aggr_pipe_cost(
    cap_cost, pipe_cost_types: list, costing_params: "Costing_params"
) -> float:
    new_pipe_cap = 0
    for i in range(len(cap_cost["D_S_G"])):
        cap_cost["other_pipe_cost"].append(
            get_pipe_other_cost(
                cp=costing_params,
                d_mm=cap_cost["DN"][i],
                l_km=cap_cost["length"][i],
                anl_types=pipe_cost_types,
            )
        )
        mat_cost = cap_cost["mat_cost"][i]
        other_pipe_cost = sum(
            [cap_cost["other_pipe_cost"][i][x] for x in pipe_cost_types]
        )
        new_pipe_cap += mat_cost + other_pipe_cost
    return new_pipe_cap
