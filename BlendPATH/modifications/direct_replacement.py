import copy
import itertools
import logging
import math
from collections import namedtuple
from dataclasses import dataclass

import cantera as ct
import numpy as np

import BlendPATH.costing.costing as bp_cost
import BlendPATH.costing.pipe_costs.steel_pipe_costs as bp_pipe_cost
import BlendPATH.file_writing.mod_file_out_util as mod_file_util
import BlendPATH.Global as gl
import BlendPATH.modifications.mod_util as bp_mod_util
import BlendPATH.network.pipeline_components as bp_plc
import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.network.BlendPATH_network import BlendPATH_network, Design_params

logger = logging.getLogger(__name__)


@dataclass
class compressor_bypass:
    comp_name: int
    comp_i: int
    pipe_changes: list[tuple[str, str, str]]
    original_from_node: str
    original_to_node: str
    comp: bp_plc.Compressor
    ps: int


def direct_replacement(
    network: BlendPATH_network,
    design_option: str = "b",
    new_filename: str = "modified",
    design_params: Design_params = None,
    costing_params: bp_cost.Costing_params = None,
    allow_compressor_bypass: bool = False,
) -> tuple:
    """
    Modify network with direct replacement method
    """

    max_CR = design_params.max_CR
    final_outlet_pressure = design_params.final_outlet_pressure_mpa_g
    assign_eta_s, assign_eta_driver = bp_mod_util.compressor_eta(
        design_params=design_params
    )

    # Copy the network
    nw = bp_mod_util.copy_network(network=network, design_params=design_params)

    # Get size pipesegments
    n_ps = len(nw.pipe_segments)

    dr_entry = namedtuple(
        "dr_entry", ["grade", "diam_i", "th", "dn", "pressure", "schedule"]
    )
    lcot_entry = namedtuple(
        "lcot_entry",
        [
            "lcot",
            "geometry",
            "cr",
            "mat_cost",
            "supply_comp",
            "supp_p",
            "combo",
            "comp_bypass",
        ],
    )

    cr_max_total = max(max_CR)

    seg_options = [[] for _ in range(n_ps)]

    # Loop segments
    for ps_i, ps in enumerate(nw.pipe_segments):
        # Fetch diameters 5 and larger
        dn_options, od_options = ps.get_DNs(5)

        # Loop grades
        for grade in bp_pa.get_pipe_grades():
            # Loop diameters
            for dn_i, dn in enumerate(dn_options):
                # Get all schedules for this combination
                (th_valid, schedule_valid, pressure_valid) = ps.get_viable_schedules(
                    design_option=design_option,
                    ASME_params=design_params.asme,
                    grade=grade,
                    ASME_pressure_flag=False,
                    DN=dn,
                    return_all=True,
                )
                # Loop schedules
                for th_i, th in enumerate(th_valid):
                    if (
                        not schedule_valid
                        or schedule_valid[th_i] is np.nan
                        or pressure_valid[th_i] > 20.0
                    ):
                        continue

                    d_inner_mm = od_options[dn_i] - 2 * th

                    seg_options[ps_i].append(
                        dr_entry(
                            grade,
                            d_inner_mm,
                            th,
                            dn,
                            pressure_valid[th_i],
                            schedule_valid[th_i],
                        )
                    )

    # Run LCOT sweep
    lcot_vals = []

    seg_options_slim = [seg_options[i] for i, v in enumerate(seg_options) if v]
    valid_options = []
    if seg_options_slim:
        valid_options = list(set.intersection(*map(set, seg_options_slim)))

    sn_orig = network.supply_nodes[list(network.supply_nodes.keys())[0]]
    orig_supply_pressure = sn_orig.pressure_mpa
    sn_orig_node = nw.supply_nodes[list(nw.supply_nodes.keys())[0]].node

    # First try to solve the original network, maybe no changes needed
    no_change = False
    try:
        nw.solve()
        for pipe in ps.pipes:
            pipe.design_violation = pipe.pressure_MPa > pipe.design_pressure_MPa
            if pipe.design_violation:
                raise ValueError()
        if nw.pipe_segments[-1].end_node.pressure < final_outlet_pressure * gl.MPA2PA:
            raise ValueError("Final outlet pressure")
        no_change = True
    # else reset to original values. Resolve to see if issues hydraulic issues
    except ValueError:
        pass

    # Get combinations for swapping
    combos = []
    for L in range(n_ps + 1):
        for subset in itertools.combinations(range(n_ps), L):
            combos.append(subset)

    # Get original values
    og_pipe_vals = []
    og_comps_out = []
    saved_sol = None
    for ps_i, ps in enumerate(nw.pipe_segments):
        og_pipe_vals.append(
            {
                "diameter_mm": ps.pipes[0].diameter_mm,
                "thickness_mm": ps.pipes[0].thickness_mm,
                "DN": ps.pipes[0].DN,
            }
        )
        og_comps_out.append(ps.comps[0].pressure_out_mpa_g if ps.comps else -1)

    comp_pipe_changes = {}
    if not no_change:
        sn = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]

        # Get compressor bypass options
        comp_bypass = bp_mod_util.get_comp_bypass(
            allow_compressor_bypass=allow_compressor_bypass, nw=nw
        )

        # Setup compressor bypass connections
        comp_pipe_changes = {}
        for comp_i, comp in enumerate(nw.compressors.values()):
            comp_pipe_changes[comp.name] = compressor_bypass(
                comp_name=comp.name,
                comp_i=comp_i,
                pipe_changes=[],
                original_from_node=comp.from_node.name,
                original_to_node=comp.to_node.name,
                comp=copy.deepcopy(comp),
                ps=[i for i, ps in enumerate(nw.pipe_segments) if comp in ps.comps],
            )
            for cxn in comp.from_node.connections["Pipe"]:
                if cxn.to_node == comp.from_node:
                    comp_pipe_changes[comp.name].pipe_changes.append(("to", cxn.name))
                elif cxn.from_node == comp.from_node:
                    comp_pipe_changes[comp.name].pipe_changes.append(("from", cxn.name))

        for bypass_combo in comp_bypass:
            logger.info(f"Running bypass combo: {bypass_combo}")
            for comp_value in comp_pipe_changes.values():
                if comp_value.comp_i in bypass_combo:
                    logger.info(f"Bypassing compressor {comp_value.comp_name}")
                    for pipe_changes in comp_value.pipe_changes:
                        setattr(
                            nw.pipes[pipe_changes[1]],
                            f"{pipe_changes[0]}_node",
                            nw.nodes[comp_value.original_to_node],
                        )
                    # only bypass if needed. Could already be bypassed from pervious iteration
                    if comp_value.comp_name in nw.compressors:
                        del nw.compressors[comp_value.comp_name]
                        del nw.nodes[comp_value.original_from_node]
                        nw.pipe_segments[comp_value.ps[0]].comps = []
                else:
                    nw.compressors[comp_value.comp_name] = comp_value.comp
                    nw.nodes[comp_value.original_from_node] = comp_value.comp.from_node
                    nw.compressors[comp_value.comp_name].to_node = nw.nodes[
                        comp_value.original_to_node
                    ]
                    nw.pipe_segments[comp_value.ps[0]].comps = [comp_value.comp]

                    for pipe_changes in comp_value.pipe_changes:
                        setattr(
                            nw.pipes[pipe_changes[1]],
                            f"{pipe_changes[0]}_node",
                            nw.nodes[comp_value.original_from_node],
                        )

                nw.assign_nodes()
                nw.assign_connections()

            for i, design in enumerate(valid_options):
                for combo in combos:
                    # Update
                    total_length_km = 0
                    for ps_i, ps in enumerate(nw.pipe_segments):
                        if ps_i in combo:
                            # Make changes
                            for pipe in ps.pipes:
                                pipe.diameter_mm = design.diam_i
                                pipe.thickness_mm = design.th
                                pipe.DN = design.dn
                                total_length_km += pipe.length_km
                            if ps_i > 0:
                                if nw.pipe_segments[ps_i - 1].comps:
                                    comp = nw.pipe_segments[ps_i - 1].comps[0]
                                    comp.pressure_out_mpa_g = design.pressure
                        else:
                            for pipe in ps.pipes:
                                pipe.diameter_mm = og_pipe_vals[ps_i]["diameter_mm"]
                                pipe.thickness_mm = og_pipe_vals[ps_i]["thickness_mm"]
                                pipe.DN = og_pipe_vals[ps_i]["DN"]
                            if ps_i > 0:
                                if nw.pipe_segments[ps_i - 1].comps:
                                    comp = nw.pipe_segments[ps_i - 1].comps[0]
                                    comp.pressure_out_mpa_g = pipe.design_pressure_MPa

                    inlet_pressure = (
                        design.pressure
                        if 0 in combo
                        else nw.pipe_segments[0].design_pressure_MPa
                    )
                    supp_p_list = [inlet_pressure]
                    if orig_supply_pressure < inlet_pressure:
                        supp_p_list = (
                            [orig_supply_pressure]
                            + list(
                                range(
                                    math.ceil(orig_supply_pressure),
                                    math.floor(inlet_pressure),
                                    1,
                                )
                            )
                            + [inlet_pressure]
                        )

                    skip_future_p_supply = False
                    for supp_p in supp_p_list:
                        if skip_future_p_supply:
                            continue
                        if supp_p > orig_supply_pressure:
                            comp_to_node = sn.node
                            new_supply_node = bp_plc.Node(
                                name="new_supply_node_comp",
                                x_h2=sn.blend,
                                composition=sn.node.composition,
                                pressure=orig_supply_pressure * gl.MPA2PA,
                                is_supply=True,
                            )
                            supply_comp = bp_plc.Compressor(
                                name="Supply compressor",
                                from_node=new_supply_node,
                                to_node=comp_to_node,
                                pressure_out_mpa_g=supp_p,
                                fuel_extract=not design_params.new_comp_elec,
                            )
                            sn.node = new_supply_node
                            nw.compressors["Supply compressor"] = supply_comp
                            nw.nodes["new_supply_node_comp"] = new_supply_node
                            nw.assign_nodes()
                            nw.assign_connections()
                        sn.pressure_mpa = min(orig_supply_pressure, supp_p)

                        try:
                            for comp_check in nw.compressors.values():
                                if comp_check.pressure_out_mpa_g > 20.0:
                                    skip_future_p_supply = True
                                    raise ValueError(
                                        "Compressor pressure out of bounds"
                                    )
                            nw.solve(initializer=0 if saved_sol is None else saved_sol)
                            max_cr = []
                            for comp in nw.compressors.values():
                                # Check compression ratio
                                this_cr = comp.compression_ratio
                                if (this_cr > cr_max_total) and (
                                    comp.name != "Supply compressor"
                                ):
                                    raise ValueError("Compression ratio")
                                max_cr.append(comp.compression_ratio)

                            if (
                                nw.pipe_segments[-1].end_node.pressure
                                < final_outlet_pressure * gl.MPA2PA
                            ):
                                raise ValueError("Final outlet pressure")

                            for this_ps_i, this_ps in enumerate(nw.pipe_segments):
                                for this_pipe in this_ps.pipes:
                                    check_pressure = this_pipe.design_pressure_MPa
                                    if this_ps_i in combo:
                                        check_pressure = design.pressure
                                    if this_pipe.pressure_MPa > check_pressure:
                                        skip_future_p_supply = True
                                        raise ValueError("Pressure rating")

                            # Get new pipe capex
                            cost = bp_pipe_cost.get_pipe_material_cost(
                                cp=costing_params,
                                di_mm=design.diam_i,
                                do_mm=design.diam_i + 2 * design.th,
                                l_km=total_length_km,
                                grade=design.grade,
                            )

                            anl_cap = bp_pipe_cost.get_pipe_other_cost(
                                cp=costing_params,
                                d_mm=design.dn,
                                l_km=total_length_km,
                                anl_types=bp_mod_util.get_pipe_cost_types(
                                    mod_type="dr", scenario_type=network.scenario_type
                                ),
                            )

                            new_pipe_cap = cost + sum(anl_cap.values())

                            mod_costing_params = bp_mod_util.get_mod_costing_params(
                                network=nw,
                                design_params=design_params,
                                costing_params=costing_params,
                                supply_comp_inputs=None,
                            )

                            price_breakdown, _ = bp_cost.calc_lcot(
                                mod_costing_params=mod_costing_params,
                                new_pipe_capex=new_pipe_cap,
                                costing_params=costing_params,
                            )

                            lcot = price_breakdown["LCOT: Levelized cost of transport"]
                            lcot_vals.append(
                                lcot_entry(
                                    lcot,
                                    design,
                                    max(max_cr) if max_cr else np.nan,
                                    cost,
                                    "Supply compressor" in nw.compressors.keys(),
                                    supp_p=supp_p,
                                    combo=combo,
                                    comp_bypass=bypass_combo,
                                )
                            )

                            if saved_sol is None:
                                saved_sol = nw.pressure_init_out

                        except (ValueError, ct.CanteraError, RuntimeError):
                            pass
                        finally:
                            # reset supply node and compressor if needed
                            if "Supply compressor" in nw.compressors.keys():
                                sn_node = nw.supply_nodes[
                                    list(nw.supply_nodes.keys())[0]
                                ]
                                sn_node.node = sn_orig_node
                                sn_node.pressure = network.supply_nodes[
                                    list(network.supply_nodes.keys())[0]
                                ].pressure_mpa
                                nw.compressors.pop("Supply compressor")
                                nw.nodes.pop("new_supply_node_comp")
                                nw.assign_nodes()
                                nw.assign_connections()

            ######################################

    # Reset compressors
    for comp_value in comp_pipe_changes.values():
        nw.compressors[comp_value.comp_name] = comp_value.comp
        nw.nodes[comp_value.original_from_node] = comp_value.comp.from_node
        nw.compressors[comp_value.comp_name].to_node = nw.nodes[
            comp_value.original_to_node
        ]
        nw.pipe_segments[comp_value.ps[0]].comps = [comp_value.comp]
    nw.assign_nodes()
    nw.assign_connections()

    add_supply_comp = False
    combo_final = None
    min_geom = None
    supp_p = None
    if valid_options and not no_change:
        lcot_list = [x.lcot for x in lcot_vals]
        min_lcot_ind = lcot_list.index(min(lcot_list))
        min_geom = lcot_vals[min_lcot_ind].geometry
        add_supply_comp = lcot_vals[min_lcot_ind].supply_comp
        supp_p = lcot_vals[min_lcot_ind].supp_p
        combo_final = lcot_vals[min_lcot_ind].combo

    res = {
        x: {
            v: []
            for v in [
                "grades",
                "costs",
                "ths",
                "schedules",
                "pressures",
                "inner diameters",
                "DN",
                "lengths",
                "ps",
                "name",
            ]
        }
        for x in nw.pipes.keys()
    }
    if not no_change:
        for ps_i, ps in enumerate(network.pipe_segments):
            if ps_i in combo_final:
                for pipe in ps.pipes:
                    if (
                        (min_geom.dn == pipe.DN)
                        and (min_geom.schedule == pipe.schedule)
                        and (min_geom.grade == pipe.grade)
                    ):
                        continue
                    res[pipe.name]["grades"].append(min_geom.grade)
                    res[pipe.name]["costs"].append(lcot_vals[min_lcot_ind].mat_cost)
                    res[pipe.name]["ths"].append(min_geom.th)
                    res[pipe.name]["schedules"].append(min_geom.schedule)
                    res[pipe.name]["pressures"].append(min_geom.pressure)
                    res[pipe.name]["inner diameters"].append(min_geom.diam_i)
                    res[pipe.name]["DN"].append(min_geom.dn)
                    res[pipe.name]["lengths"].append(pipe.length_km)
                    res[pipe.name]["ps"].append(ps_i)
                    res[pipe.name]["name"].append(pipe.name)

    # Can combine this with previous step
    # Get the minimum cost option per pipe and related geometry
    min_cost = {}
    combined_pipe = {
        x: []
        for x in [
            "D_S_G",
            "DN",
            "sch",
            "grade",
            "length",
            "mat_cost",
            "other_pipe_cost",
            "total cost",
        ]
    }
    for pipe, val in res.items():
        if not val["costs"]:
            continue
        mincost_i = min(val["costs"])
        mincost_ind = val["costs"].index(mincost_i)
        # min_cost[pipe] = {x: v[mincost_ind] for x, v in val.items()}

        min_cost[pipe] = bp_mod_util.New_pipes(
            grade=val["grades"][mincost_ind],
            cost=val["costs"][mincost_ind],
            th=val["ths"][mincost_ind],
            schedule=val["schedules"][mincost_ind],
            pressure=val["pressures"][mincost_ind],
            inner_diameter=val["inner diameters"][mincost_ind],
            dn=val["DN"][mincost_ind],
            length=val["lengths"][mincost_ind],
            ps=val["ps"][mincost_ind],
            name=val["name"][mincost_ind],
        )

        # Combine by DN,sch,grade
        combined_ind = min_cost[pipe].D_S_G
        # If the same DN, sch, and grade, combo exists, then add to the length and cost
        # Otherwise make a new entry
        if combined_ind in combined_pipe["D_S_G"]:
            dsg_ind = combined_pipe["D_S_G"].index(combined_ind)
            combined_pipe["length"][dsg_ind] += min_cost[pipe].length
            combined_pipe["mat_cost"][dsg_ind] += min_cost[pipe].cost
        else:
            combined_pipe["DN"].append(min_cost[pipe].dn)
            combined_pipe["sch"].append(min_cost[pipe].schedule)
            combined_pipe["grade"].append(min_cost[pipe].grade)
            combined_pipe["D_S_G"].append(combined_ind)
            combined_pipe["length"].append(min_cost[pipe].length)
            combined_pipe["mat_cost"].append(min_cost[pipe].cost)
    # Temporary fix for setting total material cost
    if combined_pipe["mat_cost"]:
        combined_pipe["mat_cost"][-1] = lcot_vals[min_lcot_ind].mat_cost

    # Remake file

    make_result_file(
        filename=new_filename,
        network=network,
        nw=nw,
        min_cost=min_cost,
        no_change=no_change,
        combo_final=combo_final,
        min_geom=min_geom,
        supp_p=supp_p,
        add_supply_comp=add_supply_comp,
        design_params=design_params,
        assign_eta_s=assign_eta_s,
        assign_eta_driver=assign_eta_driver,
        compressor_bypass=lcot_vals[min_lcot_ind].comp_bypass if lcot_vals else (),
    )

    return min_cost, combined_pipe, []


def make_result_file(
    filename: str,
    network: BlendPATH_network,
    nw: BlendPATH_network,
    min_cost: dict,
    no_change: bool,
    combo_final: tuple,
    min_geom: tuple,
    supp_p: float,
    add_supply_comp: bool,
    design_params: Design_params,
    assign_eta_s: float,
    assign_eta_driver: float,
    compressor_bypass: tuple,
):

    # Compressors
    # Make new compressors
    bypass_swaps = {}
    new_comps = {x: [] for x in mod_file_util.comps_cols()}
    for comp_i, comp in enumerate(nw.compressors.values()):
        if comp_i in compressor_bypass:
            bypass_swaps[comp.from_node.name] = comp.to_node.name
            continue
        fuel_extract = comp.fuel_extract and not design_params.existing_comp_elec
        new_comps["compressor_name"].append(comp.name)
        new_comps["from_node"].append(comp.from_node.name)
        new_comps["to_node"].append(comp.to_node.name)
        new_comps["rating_MW"].append(comp.original_rating_MW)
        new_comps["extract_fuel"].append(fuel_extract)
        new_comps["eta_s"].append(
            comp.eta_comp_s if fuel_extract else comp.eta_comp_s_elec
        )
        new_comps["eta_driver"].append(comp.eta_driver if fuel_extract else "")

        # Assume to node has to be the outlet pressure
        for pipe in comp.to_node.connections["Pipe"]:
            for ps_i, ps in enumerate(nw.pipe_segments):
                if pipe in ps.pipes:
                    if not no_change and ps_i in combo_final:
                        if ps_i > 0 and nw.pipe_segments[ps_i - 1].comps:
                            p_max = min_geom.pressure
                    else:
                        p_max = ps.design_pressure_MPa

        new_comps["pressure_out_mpa_g"].append(p_max)

    # Pipes
    new_pipes = {x: [] for x in mod_file_util.pipes_cols()}

    for pipe in network.parent_pipes.values():
        # These values don't change from DR method
        new_pipes["pipe_name"].append(pipe.name)
        new_pipes["from_node"].append(
            pipe.from_node.name
            if pipe.from_node.name not in bypass_swaps
            else bypass_swaps[pipe.from_node.name]
        )

        new_pipes["to_node"].append(
            pipe.to_node.name
            if pipe.to_node.name not in bypass_swaps
            else bypass_swaps[pipe.to_node.name]
        )
        new_pipes["length_km"].append(pipe.length_km)
        new_pipes["roughness_mm"].append(pipe.roughness_mm)
        # These values update
        repd = pipe.name in min_cost.keys()
        inner_diam = min_cost[pipe.name].inner_diameter if repd else pipe.diameter_mm
        thickness = min_cost[pipe.name].th if repd else pipe.thickness_mm
        grade = min_cost[pipe.name].grade if repd else pipe.grade
        new_pipes["diameter_mm"].append(inner_diam)
        new_pipes["thickness_mm"].append(thickness)
        new_pipes["rating_code"].append(grade)

    # Nodes
    # Make new nodes sheet
    new_nodes = {x: [] for x in mod_file_util.nodes_cols()}
    for node in network.nodes.values():
        if not node._report_out or node.name in bypass_swaps:
            continue
        new_nodes["node_name"].append(node.name)
        p_max = np.inf
        for pipe in node.connections["Pipe"]:
            min_cost_val = (
                min_cost[pipe.name].pressure
                if pipe.name in min_cost.keys()
                else pipe.design_pressure_MPa
            )
            p_max = min(p_max, min_cost_val)
        new_nodes["p_max_mpa_g"].append(p_max)

    # Make new supply
    new_supply = {x: [] for x in mod_file_util.supply_cols()}
    for supply in network.supply_nodes.values():
        new_supply["supply_name"].append(supply.name)
        new_supply["node_name"].append(supply.node.name)
        new_supply["pressure_mpa_g"].append(
            supp_p if not no_change else supply.pressure_mpa
        )
        new_supply["flowrate_MW"].append("")
        new_supply["blend"].append(nw.composition.x["H2"])

    if add_supply_comp:
        sn = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]

        # Add new compressor
        new_comps["compressor_name"].insert(0, "Supply compressor")
        new_comps["from_node"].insert(0, "Supply compressor from_node")
        new_comps["to_node"].insert(0, sn.node.name)
        new_comps["rating_MW"].insert(0, 0)
        new_comps["extract_fuel"].insert(0, not design_params.new_comp_elec)
        new_comps["eta_s"].insert(0, assign_eta_s)
        new_comps["eta_driver"].insert(
            0, "" if np.isnan(assign_eta_driver) else assign_eta_driver
        )
        p_max = np.inf
        for pipe in sn.node.connections["Pipe"]:
            min_cost_val = (
                min_cost[pipe.name].pressure
                if pipe.name in min_cost.keys()
                else pipe.design_pressure_MPa
            )
            p_max = min(p_max, min_cost_val)
        new_comps["pressure_out_mpa_g"].insert(0, supp_p)

        # Add new node
        new_nodes["node_name"].insert(0, "Supply compressor from_node")
        new_nodes["p_max_mpa_g"].insert(0, p_max)

        # Update supply node
        new_supply["node_name"][-1] = "Supply compressor from_node"
        new_supply["pressure_mpa_g"][-1] = supply.pressure_mpa
        new_supply["blend"][-1] = nw.composition.x["H2"]

    # Make new demands
    new_demand = {x: [] for x in mod_file_util.demand_cols()}
    for demand in network.demand_nodes.values():
        new_demand["demand_name"].append(demand.name)
        new_demand["node_name"].append(demand.node.name)
        new_demand["flowrate_MW"].append(demand.flowrate_MW)

    # Make new composition
    new_composition = {x: [] for x in mod_file_util.composition_cols()}
    for species, x in network.composition.x.items():
        new_composition["SPECIES"].append(species)
        new_composition["X"].append(x)

    mod_file_util.write_to_network_file(
        filename=filename,
        new_pipes=new_pipes,
        new_nodes=new_nodes,
        new_comps=new_comps,
        new_supply=new_supply,
        new_demand=new_demand,
        new_composition=new_composition,
    )
