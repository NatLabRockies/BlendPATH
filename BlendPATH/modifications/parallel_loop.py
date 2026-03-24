import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

import BlendPATH.costing.costing as bp_cost
import BlendPATH.costing.pipe_costs.steel_pipe_costs as bp_pipe_cost
import BlendPATH.file_writing.mod_file_out_util as mod_file_util
import BlendPATH.Global as gl
import BlendPATH.modifications.mod_util as bp_mod_util
import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.network import pipeline_components as bp_plc
from BlendPATH.network.BlendPATH_network import BlendPATH_network
from BlendPATH.scenario_helper import Design_params

logger = logging.getLogger(__name__)


@dataclass
class PL_res:
    min_sol_per_supp_p: list
    grade: str
    th: float
    schedule: str
    pressure: float
    diam_inner: float
    dn: float

    def __post_init__(self):
        self.cost = self.min_sol_per_supp_p["lcot"]
        self.mat_cost = self.min_sol_per_supp_p["mat_cost"]
        self.loop_length = self.min_sol_per_supp_p["loop_length"]
        self.supply_comp = self.min_sol_per_supp_p["add_supply_comp"]
        self.inlet_p = self.min_sol_per_supp_p["inlet_p"]
        self.m_dot_in = self.min_sol_per_supp_p["m_dot_in"]
        self.p_out = self.min_sol_per_supp_p["p_out"]


def parallel_loop(
    network: BlendPATH_network,
    design_params: Design_params = None,
    design_option: str = "b",
    new_filename: str = "modified",
    costing_params: bp_cost.Costing_params = None,
    allow_compressor_bypass: bool = False,
) -> tuple:
    """
    Modify the pipeline with parallel looping
    """

    # Copy the network
    nw = bp_mod_util.copy_network(network=network, design_params=design_params)

    # Set compression ratio variables
    max_CR = design_params.max_CR
    n_cr = len(max_CR)
    # Set final outlet pressure
    final_outlet_pressure = design_params.final_outlet_pressure_mpa_g

    # For new compressors (supply compressor in PL case)
    assign_eta_s, assign_eta_driver = bp_mod_util.compressor_eta(
        design_params=design_params
    )

    # Get compressor bypass options
    comp_bypass = bp_mod_util.get_comp_bypass(
        allow_compressor_bypass=allow_compressor_bypass, nw=nw
    )

    # Initialize results. This is per bypass option per CR and per pipe segment
    # number of pipe segments change with bypass option
    res = [
        [
            {i: [] for i in range(len(nw.pipe_segments) - len(comp_bypass[a]))}
            for _ in range(n_cr)
        ]
        for a in range(len(comp_bypass))
    ]
    res_per_bc = []

    # Loop through bypass options
    for bc_i, bypass_combo in enumerate(comp_bypass):
        logger.info(f"Running bypass combo: {bypass_combo}")

        # Remake psuedo pipe segments based on compressor bypasses (combining pipe segments)
        # This assumes that only compressors are separating pipe segments
        pipe_segments = generateNewPipeSegments(
            pipe_segments=nw.pipe_segments, bypass_combo=bypass_combo
        )
        # Save number of pipe segments
        n_ps = len(pipe_segments)

        # Loop through compressor ratios
        skip_cr: bool = False
        for cr_i, CR_ratio in enumerate(max_CR):
            if skip_cr:
                skip_cr = False
                continue

            # Seed initial values. These update per segment
            prev_ASME_pressure = -1
            m_dot_in_prev = pipe_segments[-1].mdot_out

            # Loop through pipe segments in reverse.
            # Maintain reverse, since compressor use can effect upstream mass flow rate
            for ps_i, ps in reversed(list(enumerate(pipe_segments))):
                # Take the next 15 greater than or equal DNs
                dn_options, od_options = ps.get_DNs(15)

                # setup list of supply pressure to investigate with a supply compressor
                supp_p_list = bp_mod_util.get_supply_pressure_list(
                    nw=nw, ps=ps, ps_i=ps_i
                )

                # Set up pressure bounds for segment analysis
                pressure_in_Pa = ps.design_pressure_MPa * gl.MPA2PA
                if ps_i == n_ps - 1:
                    pressure_out_Pa = final_outlet_pressure * gl.MPA2PA
                else:
                    pressure_out_Pa = prev_ASME_pressure / CR_ratio
                # Saving length so it isn't calculated each time
                l_total = ps.length_km

                # Add total offtake mass flow, if not already included
                all_mdot = bp_mod_util.update_offtake_mdot(
                    offtake_mdots=ps.offtake_mdots, m_dot_in_prev=m_dot_in_prev
                )

                # Aggregate static parameters per pipe segment
                seg_params = bp_mod_util.Segment_params(
                    p_out_target=pressure_out_Pa,
                    offtakes_mdot=all_mdot,
                    ps=ps,
                    nw=nw,
                    costing_params=costing_params,
                    design_params=design_params,
                    seg_compressor_pressure_out=prev_ASME_pressure,
                    CR_ratio=CR_ratio,
                    eta_s=assign_eta_s,
                    eta_driver=assign_eta_driver,
                )

                # Loop through pipe steel grades
                for grade in bp_pa.get_pipe_grades():
                    # Speed up calculations: lowest DN is cheapest so skip higher DNs
                    dn_satisfied = False

                    # Loop thru diameters >= current DN
                    for dn_i, dn in enumerate(dn_options):
                        # Skip larger DN's if smaller solution found
                        if dn_satisfied:
                            break

                        # Get closest schedule that satifies design pressure
                        (th, schedule, pressure) = ps.get_viable_schedules(
                            design_option=design_option,
                            ASME_params=design_params.asme,
                            grade=grade,
                            ASME_pressure_flag=True,
                            DN=dn,
                        )
                        # Skip if no viable schedules with this grade
                        if schedule is np.nan:
                            continue

                        # Get inner and outer diameters
                        d_outer_mm = od_options[dn_i]
                        d_inner_mm = d_outer_mm - 2 * th

                        # Loop through supply pressures
                        supp_p_min_res = []
                        for sup_p in supp_p_list:
                            loop_network, loop_length, m_dot_seg, comp_out, p_out = (
                                get_loop_length2(
                                    d_loop=d_inner_mm,
                                    th=th,
                                    l_total=l_total,
                                    p_in=sup_p,
                                    rating_code=grade,
                                    seg_params=seg_params,
                                )
                            )
                            if np.isnan(loop_length):
                                continue
                            dn_satisfied = True

                            # Add as an entry to determine best supply pressure
                            supp_p_min_res.append(
                                get_segment_results(
                                    seg_params=seg_params,
                                    d_inner_mm=d_inner_mm,
                                    d_outer_mm=d_outer_mm,
                                    dn=dn,
                                    loop_length=loop_length,
                                    grade=grade,
                                    nw=nw,
                                    loop_network=loop_network,
                                    sup_p=sup_p,
                                    m_dot_seg=m_dot_seg,
                                    p_out=p_out,
                                )
                            )
                        # After looping throug pressure supply choices, choose best supply pressure
                        # based on LCOT
                        if not supp_p_min_res:
                            continue

                        lcot_p_spply = [x["lcot"] for x in supp_p_min_res]
                        idxmin_lcot_p_supp = lcot_p_spply.index(min(lcot_p_spply))

                        # Add to results for this cr and segment
                        res[bc_i][cr_i][ps_i].append(
                            PL_res(
                                min_sol_per_supp_p=supp_p_min_res[idxmin_lcot_p_supp],
                                grade=grade,
                                th=th,
                                schedule=schedule,
                                pressure=pressure,
                                diam_inner=d_inner_mm,
                                dn=dn,
                            )
                        )

                        logger.info(
                            f"Solution found for dn={dn}, grade={grade}, th={th}, Loop length = {supp_p_min_res[idxmin_lcot_p_supp]['loop_length']}, segment={ps_i}, Inlet pressure = {supp_p_min_res[idxmin_lcot_p_supp]['inlet_p']}, LCOT = {supp_p_min_res[idxmin_lcot_p_supp]['lcot']}"
                        )

                # Update the previous segment pressure for use in fuel extraction calc
                # m_dot_seg isnt changing with grade and diameter, since pipe is
                # rated to original pipe MAOP
                prev_ASME_pressure = pressure_in_Pa

                # If a single segment does have any results for this cr and bypass, then no reason to continue
                if not res[bc_i][cr_i][ps_i]:
                    skip_cr = True
                    break

                # Get min cost for this segment to get the m dot for the upstream segment
                min_cost_index = res[bc_i][cr_i][ps_i].index(
                    min(res[bc_i][cr_i][ps_i], key=lambda x: x.cost)
                )
                m_dot_in_prev = res[bc_i][cr_i][ps_i][min_cost_index].m_dot_in
        cr_lcot_sum = [0] * n_cr
        for cr_i in range(n_cr):
            # Get the lowest cost LCOT for each CR and PS. Add up the PS costs to get the lowest network cost for each CR.
            cr_lcot_sum[cr_i] = sum(
                [
                    min([z.cost for z in res[bc_i][cr_i][x]])
                    if res[bc_i][cr_i][x]
                    else np.inf
                    for x in range(n_ps)
                ]
            )
        cr_min_index = cr_lcot_sum.index(min(cr_lcot_sum))
        # if cr_lcot_sum[cr_min_index] == np.inf:
        #     raise ValueError("No solutions found")
        res_per_bc.append(
            [
                min(res[bc_i][cr_min_index][ppss], key=lambda x: x.cost)
                if res[bc_i][cr_min_index][ppss]
                else np.inf
                for ppss in res[bc_i][cr_min_index]
            ]
        )

    # Sum up across segments
    summations = [
        sum(
            seg_entry if seg_entry is np.inf else seg_entry.cost
            for seg_entry in bypass_entry
        )
        for bypass_entry in res_per_bc
    ]
    min_bypass_combo_index = summations.index(min(summations))
    min_vals = res_per_bc[min_bypass_combo_index]
    add_supply_comp_min = min_vals[0].supply_comp
    min_bypass_combo = comp_bypass[min_bypass_combo_index]

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
    for i in min_vals:
        # Ignore 0 loop length loops
        if i.loop_length == 0:
            continue
        # Combine by DN,sch,grade
        combined_ind = (i.dn, i.schedule, i.grade)
        combined_pipe["DN"].append(i.dn)
        combined_pipe["sch"].append(i.schedule)
        combined_pipe["grade"].append(i.grade)
        combined_pipe["D_S_G"].append(combined_ind)
        combined_pipe["length"].append(i.loop_length)
        combined_pipe["mat_cost"].append(i.mat_cost)

    added_pipe_names = make_result_file(
        filename=new_filename,
        min_vals=min_vals,
        nw=nw,
        add_supply_comp=add_supply_comp_min,
        design_params=design_params,
        assign_eta_s=assign_eta_s,
        assign_eta_driver=assign_eta_driver,
        min_bypass_combo=min_bypass_combo,
    )

    # Format pipes for result file
    new_pipes_f = {}
    for p_i, min_val in enumerate(min_vals):
        new_pipes_f[f"pipe_segment_{p_i}"] = bp_mod_util.New_pipes(
            grade=min_val.grade,
            cost=min_val.mat_cost,
            th=min_val.th,
            schedule=min_val.schedule,
            pressure=min_val.pressure,
            inner_diameter=min_val.diam_inner,
            dn=min_val.dn,
            length=min_val.loop_length,
            ps=p_i,
            name=added_pipe_names[p_i],
        )

    return new_pipes_f, combined_pipe, []


def add_pipe_segments(
    p_name: str,
    length_km: float,
    diameter_mm: float,
    from_node: bp_plc.Node,
    to_node: bp_plc.Node,
    composition: bp_plc.Composition,
    ro: float,
    rating_code: str,
    th: float,
) -> tuple:
    nodes = {}
    pipes = {}

    pipes[p_name] = bp_plc.Pipe(
        name=p_name,
        from_node=from_node,
        to_node=to_node,
        diameter_mm=diameter_mm,
        length_km=length_km,
        roughness_mm=ro,
        rating_code=rating_code,
        thickness_mm=th,
    )
    node_base_name = f"{from_node.name}_{p_name}"

    if (length_km * gl.KM2M) / (diameter_mm * gl.MM2M) > gl.SEG_MAX:
        length_sub_segment = gl.SEG_MAX * (diameter_mm * gl.MM2M) / gl.KM2M
        n_nodes = int(np.floor(length_km / length_sub_segment))
        length_sub_segment = length_km / (n_nodes + 1)
        for subseg in range(n_nodes):
            new_node_name = f"{node_base_name}_subseg_{subseg}"
            nodes[new_node_name] = bp_plc.Node(
                name=new_node_name,
                composition=composition,
                _report_out=False,
            )
            to_node = nodes[new_node_name]
            new_pipe_name = f"{p_name}_subseg_{subseg}"
            pipes[new_pipe_name] = bp_plc.Pipe(
                name=new_pipe_name,
                from_node=from_node,
                to_node=to_node,
                diameter_mm=diameter_mm,
                length_km=length_sub_segment,
                roughness_mm=ro,
                rating_code=rating_code,
                thickness_mm=th,
            )
            pipes[new_pipe_name]._parent_pipe = p_name
            from_node = to_node
        pipes[p_name].from_node = from_node
        pipes[p_name].length_km = length_sub_segment
    return nodes, pipes


def make_loop_network(
    l_loop: float,
    composition: bp_plc.Composition,
    p_in: float,
    offtakes: list,
    all_mdot: list,
    d_main: float,
    d_loop: float,
    th: float,
    roughness_mm: float,
    seg_compressor: list,
    prev_ASME_pressure: float,
    comps_elec: bool,
    rating_code: str = "X70",
    eos: bp_plc.eos._EOS_OPTIONS = "rk",
    thermo_curvefit: bool = True,
    composition_tracking: bool = False,
    ff_type: bp_plc.friction_factor.FF_TYPES = "hofer",
) -> tuple:
    """
    Create new network to simulate the parallel looped segment
    """
    # Make inlet node
    n_ds_in = bp_plc.Node(
        name="in",
        composition=composition,
    )
    nodes = {n_ds_in.name: n_ds_in}
    # Initialize pipes, demands, supplies
    pipes = {}
    demands = {}
    supplys = {
        "supply": bp_plc.Supply_node(
            node=n_ds_in, pressure_mpa=p_in, blend=composition.x["H2"]
        )
    }
    compressors = {}

    if thermo_curvefit:
        HHV = composition.get_curvefit_hhv(x=composition.x["H2"])
    else:
        HHV = composition.calc_HHV(composition.x)

    # Loop through offtakes
    cumsum_offtakes = np.insert(np.cumsum(offtakes), 0, 0)
    l_done = l_loop == 0
    prev_node = n_ds_in
    node_index = 1
    for ot_i, ot_mdot in enumerate(all_mdot):
        name = f"ot_{ot_i}"
        nodes[name] = bp_plc.Node(
            name=name,
            composition=composition,
        )
        node_index += 1
        d_name = f"demand_{ot_i}"
        demands[d_name] = bp_plc.Demand_node(
            node=nodes[name],
            flowrate_MW=ot_mdot * HHV,
        )
        # If the loop is within the bounds
        if not l_done and abs(cumsum_offtakes[ot_i + 1] - l_loop) < 0.01:
            # Make loop cxn node
            l_cxn_name = name

            if l_loop > 0:
                nodes_tmp, pipes_tmp = add_pipe_segments(
                    p_name="loop_2_l_cxn",
                    length_km=l_loop,
                    diameter_mm=d_loop,
                    from_node=n_ds_in,
                    to_node=nodes[l_cxn_name],
                    composition=composition,
                    ro=roughness_mm,
                    rating_code=rating_code,
                    th=th,
                )
                nodes.update(nodes_tmp)
                pipes.update(pipes_tmp)

            l_done = True
        if not l_done and cumsum_offtakes[ot_i] < l_loop < cumsum_offtakes[ot_i + 1]:
            # Make loop cxn node
            l_cxn_name = "loop_cxn"
            nodes[l_cxn_name] = bp_plc.Node(
                name=l_cxn_name,
                composition=composition,
            )
            node_index += 1

            nodes_tmp, pipes_tmp = add_pipe_segments(
                p_name="main_2_l_cxn",
                length_km=l_loop - cumsum_offtakes[ot_i],
                diameter_mm=d_main,
                from_node=prev_node,
                to_node=nodes[l_cxn_name],
                composition=composition,
                ro=roughness_mm,
                rating_code=rating_code,
                th=th,
            )
            nodes.update(nodes_tmp)
            pipes.update(pipes_tmp)

            if l_loop > 0:
                nodes_tmp, pipes_tmp = add_pipe_segments(
                    p_name="loop_2_l_cxn",
                    length_km=l_loop,
                    diameter_mm=d_loop,
                    from_node=n_ds_in,
                    to_node=nodes[l_cxn_name],
                    composition=composition,
                    ro=roughness_mm,
                    rating_code=rating_code,
                    th=th,
                )
                nodes.update(nodes_tmp)
                pipes.update(pipes_tmp)

            nodes_tmp, pipes_tmp = add_pipe_segments(
                p_name="l_cxn_2_main",
                length_km=cumsum_offtakes[ot_i + 1] - l_loop,
                diameter_mm=d_main,
                from_node=nodes[l_cxn_name],
                to_node=nodes[name],
                composition=composition,
                ro=roughness_mm,
                rating_code=rating_code,
                th=th,
            )
            nodes.update(nodes_tmp)
            pipes.update(pipes_tmp)

            prev_node = nodes[name]

            l_done = True
        else:
            nodes_tmp, pipes_tmp = add_pipe_segments(
                p_name=f"main_{ot_i}",
                length_km=offtakes[ot_i],
                diameter_mm=d_main,
                from_node=prev_node,
                to_node=nodes[name],
                composition=composition,
                ro=roughness_mm,
                rating_code=rating_code,
                th=th,
            )
            nodes.update(nodes_tmp)
            pipes.update(pipes_tmp)

            prev_node = nodes[name]

    end_node = nodes[name]

    # If a compressor exists in the segment
    if seg_compressor:
        comp_orig = seg_compressor[0]
        comp_name = "segment_compressor"

        # Keep the final node as is for pipe connections as is, but then add another node after.
        # Compressor will compress to this node. Demand node will be updated

        # Add node after compressor
        final_node_name = "final_node"
        nodes[final_node_name] = bp_plc.Node(
            name=final_node_name,
            composition=composition,
        )
        # Update demand to be the final node

        final_demand_node_name = f"demand_{len(all_mdot) - 1}"
        prev_final_node = demands[final_demand_node_name].node
        demands[final_demand_node_name].node = nodes[final_node_name]
        # Add the compressor
        compressors[comp_name] = bp_plc.Compressor(
            name=comp_name,
            from_node=prev_final_node,
            to_node=nodes[final_node_name],
            pressure_out_mpa_g=prev_ASME_pressure / gl.MPA2PA,
            original_rating_MW=comp_orig.original_rating_MW,
            fuel_extract=not comps_elec,
        )
        compressors[comp_name].eta_comp_s = comp_orig.eta_comp_s
        compressors[comp_name].eta_comp_s_elec = comp_orig.eta_comp_s_elec
        compressors[comp_name].eta_driver = comp_orig.eta_driver
        compressors[comp_name].eta_driver_elec = comp_orig.eta_driver_elec

    looping = BlendPATH_network(
        name="looping",
        pipes=pipes,
        nodes=nodes,
        demand_nodes=demands,
        supply_nodes=supplys,
        compressors=compressors,
        composition=composition,
        thermo_curvefit=thermo_curvefit,
        eos=eos,
        ff_type=ff_type,
    )
    looping.set_thermo_curvefit(thermo_curvefit)
    looping.set_composition_tracking(composition_tracking)
    return looping, end_node, n_ds_in.connections["Pipe"], compressors


def return_nan_loop_length():
    return None, np.nan, None, None, None


def returning_loop_length(
    loop_network: BlendPATH_network,
    length: float,
    end_pipes: dict,
    compressors: dict,
    seg_params: "bp_mod_util.Segment_params",
    end_node,
) -> tuple[BlendPATH_network, float, float, dict[str, float], float]:
    """
    Return the length and mdot of the segment
    """

    comp_out = {"fuel": 0, "cost": 0, "revamp": 0, "elec": 0}

    if np.isnan(length):
        return loop_network, length, 0, comp_out, end_node.pressure
    # Added this function to sum up the m_dot in to the segment in one place

    mdot = sum([pipe.m_dot for pipe in end_pipes])

    if compressors:
        comp = compressors["segment_compressor"]
        fuel_use = comp.fuel_mdot
        comp_cost = comp.get_cap_cost(
            cp=seg_params.costing_params,
            to_electric=seg_params.design_params.existing_comp_elec,
        )

        revamped_comp_capex = comp.get_cap_cost(
            cp=seg_params.costing_params,
            revamp=True,
            to_electric=seg_params.design_params.existing_comp_elec,
        )
        fuel_use_elec = comp.fuel_electric_W
        comp_out = {
            "fuel": fuel_use,
            "cost": comp_cost,
            "revamp": revamped_comp_capex,
            "elec": fuel_use_elec,
        }

    return loop_network, length, mdot, comp_out, end_node.pressure


def make_result_file(
    filename: str,
    min_vals: list,
    nw: BlendPATH_network,
    add_supply_comp: bool,
    design_params: Design_params,
    assign_eta_s: float,
    assign_eta_driver: float,
    min_bypass_combo=tuple,
) -> list:
    new_nodes = {x: [] for x in mod_file_util.nodes_cols()}

    new_pipes = {x: [] for x in mod_file_util.pipes_cols()}

    pipe_segments = generateNewPipeSegments(
        pipe_segments=nw.pipe_segments, bypass_combo=min_bypass_combo
    )
    added_pipe_names = [""] * len(pipe_segments)

    # Make new compressors
    bypass_swaps = {}
    new_comps = {x: [] for x in mod_file_util.comps_cols()}
    for comp_i, comp in enumerate(nw.compressors.values()):
        if comp_i in min_bypass_combo:
            bypass_swaps[comp.from_node.name] = comp.to_node.name
            continue
        new_comps["compressor_name"].append(comp.name)
        new_comps["from_node"].append(comp.from_node.name)
        new_comps["to_node"].append(comp.to_node.name)
        new_comps["rating_MW"].append(comp.original_rating_MW)
        new_comps["extract_fuel"].append(comp.fuel_extract)
        new_comps["eta_s"].append(
            comp.eta_comp_s if comp.fuel_extract else comp.eta_comp_s_elec
        )
        new_comps["eta_driver"].append(
            comp.eta_driver if comp.fuel_extract else ""  # comp.eta_driver_elec_used
        )

        # Assume to node has to be the outlet pressure
        p_max = min(
            pipe.design_pressure_MPa for pipe in comp.to_node.connections["Pipe"]
        )
        new_comps["pressure_out_mpa_g"].append(p_max)

    for ps_i, pl_sol in enumerate(min_vals):
        loop_grade = pl_sol.grade
        loop_th = pl_sol.th
        loop_diam = pl_sol.diam_inner
        loop_length = pl_sol.loop_length

        # get pipe segment object
        ps = pipe_segments[ps_i]
        p_max_seg = ps.design_pressure_MPa

        # Get total loop length
        l_total = pipe_segments[ps_i].length_km

        # loop thru all nodes in pipe_segment, update pressure
        for node in pipe_segments[ps_i].nodes:
            if not node._report_out or node.name in bypass_swaps:
                continue
            new_nodes["node_name"].append(node.name)
            new_nodes["p_max_mpa_g"].append(p_max_seg)

        # Add pipes until past the offtake length
        l_added = 0
        loop_done = False
        if loop_length == 0 or loop_length == l_total:
            loop_done = True

        for pipe in ps.parent_pipes.values():
            pipe_l = pipe.length_km
            if not loop_done and l_added + pipe_l > loop_length:
                # add segmented pipe x2, intermediate node, loop pipe
                loop_cxn_name = f"loop_cxn_node_ps_{ps_i}"

                # Get index of node before loop cxn
                insert_ind = new_nodes["node_name"].index(
                    pipe.to_node.name
                    if pipe.to_node.name not in bypass_swaps
                    else bypass_swaps[pipe.to_node.name]
                )
                # Add new node for loop connection
                new_nodes["node_name"].insert(insert_ind, loop_cxn_name)
                new_nodes["p_max_mpa_g"].insert(insert_ind, p_max_seg)

                # Add segmented pipe prior to loop cxn
                pre_pipe_len = loop_length - l_added
                new_pipes["pipe_name"].append(f"{pipe.name}_pre_loop_cxn")
                new_pipes["from_node"].append(
                    pipe.from_node.name
                    if pipe.from_node.name not in bypass_swaps
                    else bypass_swaps[pipe.from_node.name]
                )
                new_pipes["to_node"].append(loop_cxn_name)
                new_pipes["length_km"].append(pre_pipe_len)
                new_pipes["roughness_mm"].append(pipe.roughness_mm)
                new_pipes["diameter_mm"].append(pipe.diameter_mm)
                new_pipes["thickness_mm"].append(pipe.thickness_mm)
                new_pipes["rating_code"].append(pipe.grade)

                # Add looped pipe to loop cxn
                added_pipe_names[ps_i] = f"PS_{ps_i}_loop"
                new_pipes["pipe_name"].append(added_pipe_names[ps_i])
                new_pipes["from_node"].append(ps.start_node.name)
                new_pipes["to_node"].append(loop_cxn_name)
                new_pipes["length_km"].append(loop_length)
                new_pipes["roughness_mm"].append(pipe.roughness_mm)
                new_pipes["diameter_mm"].append(loop_diam)
                new_pipes["thickness_mm"].append(loop_th)
                new_pipes["rating_code"].append(loop_grade)

                # Add segmented pipe after loop cxn
                new_pipes["pipe_name"].append(f"{pipe.name}_post_loop_cxn")
                new_pipes["from_node"].append(loop_cxn_name)
                new_pipes["to_node"].append(
                    pipe.to_node.name
                    if pipe.to_node.name not in bypass_swaps
                    else bypass_swaps[pipe.to_node.name]
                )
                new_pipes["length_km"].append(pipe_l - pre_pipe_len)
                new_pipes["roughness_mm"].append(pipe.roughness_mm)
                new_pipes["diameter_mm"].append(pipe.diameter_mm)
                new_pipes["thickness_mm"].append(pipe.thickness_mm)
                new_pipes["rating_code"].append(pipe.grade)

                loop_done = True
            else:
                # add pipe as normal
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
                new_pipes["diameter_mm"].append(pipe.diameter_mm)
                new_pipes["thickness_mm"].append(pipe.thickness_mm)
                new_pipes["rating_code"].append(pipe.grade)

            # update cumulative length off pipe added
            l_added += pipe.length_km

        # If pipeline has no looping, then nothing needs to be done
        if loop_length == 0:
            pass
        # If 100% looping, add in the loop
        if loop_length == l_total:
            added_pipe_names[ps_i] = f"PS_{ps_i}_loop"
            new_pipes["pipe_name"].append(added_pipe_names[ps_i])
            new_pipes["from_node"].append(ps.start_node.name)
            new_pipes["to_node"].append(ps.end_node.name)
            new_pipes["length_km"].append(l_total)
            new_pipes["roughness_mm"].append(pipe.roughness_mm)
            new_pipes["diameter_mm"].append(loop_diam)
            new_pipes["thickness_mm"].append(loop_th)
            new_pipes["rating_code"].append(loop_grade)

    # Add demand, supply, compressors, as usual

    # Add supply
    new_supply = {x: [] for x in mod_file_util.supply_cols()}
    for supply in nw.supply_nodes.values():
        new_supply["supply_name"].append(supply.name)
        new_supply["node_name"].append(supply.node.name)

        p_max = np.inf
        p_max = min_vals[0].inlet_p
        # for pipe in supply.node.connections["Pipe"]:
        #     p_max = min(pipe.design_pressure_MPa, p_max)

        new_supply["pressure_mpa_g"].append(p_max)
        new_supply["flowrate_MW"].append("")
        new_supply["blend"].append(nw.composition.x["H2"])

    # Make new demands
    new_demand = {x: [] for x in mod_file_util.demand_cols()}
    for demand in nw.demand_nodes.values():
        new_demand["demand_name"].append(demand.name)
        new_demand["node_name"].append(demand.node.name)
        new_demand["flowrate_MW"].append(demand.flowrate_MW)

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
        # p_max = np.inf
        # for pipe in supply.node.connections["Pipe"]:
        #     p_max = min(pipe.design_pressure_MPa, p_max)
        new_comps["pressure_out_mpa_g"].insert(0, min_vals[0].inlet_p)

        # Add new node
        new_nodes["node_name"].insert(0, "Supply compressor from_node")
        new_nodes["p_max_mpa_g"].insert(0, p_max)

        # Update supply node
        new_supply["node_name"][-1] = "Supply compressor from_node"
        new_supply["pressure_mpa_g"][-1] = sn.pressure_mpa
        new_supply["blend"][-1] = nw.composition.x["H2"]

    # Make new composition
    new_composition = {x: [] for x in mod_file_util.composition_cols()}
    for species, x in nw.composition.x.items():
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

    return added_pipe_names


def generateNewPipeSegments(
    pipe_segments: list[bp_plc.PipeSegment], bypass_combo: tuple[int]
) -> list[bp_plc.PipeSegment]:
    """Generate list of pipe segments, but combine segments when bypassing compressor that splits these segments

    Args:
        pipe_segments (list[bp_plc.PipeSegment]): List of original pipe segments
        bypass_combo (tuple[int]): Iterable of compressor being bypassed

    Returns:
        list[bp_plc.PipeSegment]: List of modified pipe segments
    """
    pps = []
    for ps_i, ps in enumerate(pipe_segments):
        if ps_i - 1 in bypass_combo:
            pps[-1].design_pressure_MPa = min(
                ps.design_pressure_MPa,
                pps[-1].design_pressure_MPa,
            )
            pps[-1].nodes.extend(ps.nodes)
            pps[-1].demand_nodes.extend(ps.demand_nodes)
            pps[-1].offtake_mdots.extend(ps.offtake_mdots)
            # Need to pop since the last item is also the total length
            pps[-1].offtake_lengths.pop()
            # Add the existing length to the new length to get total length from start
            # Make sure this happens before updating the length
            # Only add to first element
            pps[-1].offtake_lengths.extend(
                [
                    otl + (pps[-1].length_km if otl_i == 0 else 0)
                    for otl_i, otl in enumerate(ps.offtake_lengths)
                    if otl != 0
                ]
            )

            pps[-1].DN = min(ps.DN, pps[-1].DN)
            pps[-1].comps = copy.copy(ps.comps)

            if pps[-1].diameter[-1][0] == ps.diameter:
                pps[-1].diameter[-1] = (
                    ps.diameter,
                    pps[-1].diameter[-1][1] + ps.length_km,
                )
            else:
                pps[-1].diameter.append((ps.diameter, ps.length_km))

            # Keep last for any length calculations need prior
            # pps[-1].length_km += ps.length_km
            pps[-1].pipes.extend(ps.pipes)
            pass

        else:
            pps.append(copy.deepcopy(ps))
            if pps[-1].offtake_lengths[-1] == 0:
                pps[-1].offtake_lengths.pop()
            pps[-1].diameter = [(ps.diameter, ps.length_km)]
    return pps


def get_loop_length2(
    d_loop: float,
    th: float,
    l_total: float,
    p_in: float,
    rating_code: str,
    seg_params: "bp_mod_util.Segment_params",
) -> tuple[BlendPATH_network, float, float, dict[str, float], float]:
    """
    Determine the loop length to satisfy constraints
    """
    l_bounds = [0, l_total]

    l_vals = []
    p_vals = []

    comps_elec = seg_params.design_params.existing_comp_elec

    # Test boundary conditions first
    for l_loop in l_bounds:
        looping, end_node, end_pipes, compressors = make_loop_network2(
            l_loop=l_loop,
            d_loop=d_loop,
            th=th,
            p_in=p_in,
            comps_elec=comps_elec,
            rating_code=rating_code,
            seg_params=seg_params,
        )
        try:
            looping.solve()
        except (ValueError, RuntimeError) as err_val:
            p_vals.append(0)
            l_vals.append(l_loop)
            if err_val.args[0] in ["Negative pressure", "Pressure below threshold"]:
                # If negative or low pressure is achieved with 100% looping then
                # less looping will only make it more negative (higher pressure drop)
                # Thus this diameter/grade is not valid
                if l_loop == l_total:
                    return return_nan_loop_length()
            continue
        p_vals.append(end_node.pressure)
        l_vals.append(l_loop)
        # If 0 looping, and outlet pressure is greater than target, then no looping is the cheapest option
        if end_node.pressure > seg_params.p_out_target and l_loop == 0:
            return returning_loop_length(
                loop_network=looping,
                length=0,
                end_pipes=end_pipes,
                compressors=compressors,
                seg_params=seg_params,
                end_node=end_node,
            )
        # If end pressure is less than target at 100% looping, then this combo is not a solution, since even with
        # 100% looping, there is too much pressure drop
        if end_node.pressure < seg_params.p_out_target and l_loop == l_total:
            return return_nan_loop_length()

    iter = 0
    err = np.inf
    while err > gl.PL_LEN_TOL:
        l_loop = get_next_l_loop(
            p_vals=p_vals,
            l_vals=l_vals,
            p_target=seg_params.p_out_target,
            l_bounds=l_bounds,
        )
        if l_loop > l_total:
            raise RuntimeError("Predicted loop is greater than total length")

        # If it is close to the full loop just return full loop
        if abs(l_loop - l_total) < gl.PL_LEN_TOL:
            return returning_loop_length(
                looping,
                l_total,
                end_pipes,
                compressors,
                seg_params,
                end_node,
            )
        # If it is close to zero loop and 0 wasn't a solution, then not a solution
        if abs(l_loop) < gl.PL_LEN_TOL:
            return return_nan_loop_length()

        # If the bounds are too close then break out
        if len(l_vals) > 1 and abs(l_vals[-1] - l_vals[-2]) < gl.PL_LEN_TOL:
            return returning_loop_length(
                looping,
                l_loop,
                end_pipes,
                compressors,
                seg_params,
                end_node,
            )

        looping, end_node, end_pipes, compressors = make_loop_network2(
            l_loop=l_loop,
            d_loop=d_loop,
            th=th,
            p_in=p_in,
            comps_elec=comps_elec,
            rating_code=rating_code,
            seg_params=seg_params,
        )

        # If negative pressure, solve will provide a Value error
        # If so, then the loop needs to be longer
        # So the lower bound is increased to the current loop length
        try:
            looping.solve()
        except ValueError as err_val:
            if err_val.args[0] in [
                "Negative pressure",
                "Pressure below threshold",
                f"Could not converge in {gl.MAX_ITER} iterations",
            ]:
                l_vals.append(l_loop)
                p_vals.append(0)
            else:
                pass
        else:
            outlet_p = end_node.pressure

            err = abs(
                (seg_params.p_out_target - outlet_p) / seg_params.p_out_target
            )  # In Pa

            l_vals.append(l_loop)
            p_vals.append(end_node.pressure)

        if iter > gl.MAX_ITER:
            raise ValueError(f"Solver could not solve in {gl.MAX_ITER} iterations")
        iter += 1

    return returning_loop_length(
        looping,
        l_loop,
        end_pipes,
        compressors,
        seg_params,
        end_node,
    )


def make_loop_network2(
    l_loop: float,
    p_in: float,
    d_loop: float,
    th: float,
    comps_elec: bool,
    rating_code: str,
    seg_params: "bp_mod_util.Segment_params",
) -> tuple:
    """
    Create new network to simulate the parallel looped segment
    """
    # Make inlet node
    n_ds_in = bp_plc.Node(
        name="in",
        composition=seg_params.composition,
    )
    nodes = {n_ds_in.name: n_ds_in}
    # Initialize pipes, demands, supplies
    pipes = {}
    demands = {}
    supplys = {
        "supply": bp_plc.Supply_node(
            node=n_ds_in, pressure_mpa=p_in, blend=seg_params.composition.x["H2"]
        )
    }
    prev_node = n_ds_in
    compressors = {}

    all_lengths = bp_mod_util.get_sorted_lengths(
        d_main=seg_params.d_main_inner,
        l_loop=l_loop,
        all_mdot=seg_params.offtakes_mdot,
        offtakes=seg_params.offtakes_length,
        l_comps=[],
        hhv=seg_params.hhv,
    )

    diam_index = 0
    ot_i = 0
    prev_length = 0
    for val in all_lengths:
        length = val.length_km
        same_node = length - prev_length == 0
        # If current item is a offtake, connect the previous node to this offtake
        if val.val_type == "offtake":
            if same_node:
                d_name = f"demand_{ot_i}"
                demands[d_name] = bp_plc.Demand_node(
                    node=prev_node,
                    flowrate_MW=val.mw,
                )
            else:
                node_name = f"ot_node_{ot_i}"
                nodes[node_name] = bp_plc.Node(
                    name=node_name, composition=seg_params.composition
                )
                d_name = f"demand_{ot_i}"
                demands[d_name] = bp_plc.Demand_node(
                    node=nodes[node_name],
                    flowrate_MW=val.mw,
                )
                nodes_tmp, pipes_tmp = add_pipe_segments(
                    p_name=f"pipe_to_ot_{ot_i}",
                    length_km=length - prev_length,
                    diameter_mm=seg_params.d_main_inner[diam_index][0],
                    from_node=prev_node,
                    to_node=nodes[node_name],
                    composition=seg_params.composition,
                    ro=seg_params.roughness_mm,
                    rating_code=rating_code,
                    th=th,
                )
                nodes.update(nodes_tmp)
                pipes.update(pipes_tmp)
                prev_node = nodes[node_name]
                prev_length = length
            ot_i += 1
        # If the current item indicates a change in diameter, connect the previous node to this diameter change nodev
        elif val.val_type == "diam":
            if not same_node:
                node_name = f"diam_change_node_{diam_index}"
                nodes[node_name] = bp_plc.Node(
                    name=node_name, composition=seg_params.composition
                )
                nodes_tmp, pipes_tmp = add_pipe_segments(
                    p_name=f"pipe_to_diam_change_{diam_index}",
                    length_km=length - prev_length,
                    diameter_mm=seg_params.d_main_inner[diam_index][0],
                    from_node=prev_node,
                    to_node=nodes[node_name],
                    composition=seg_params.composition,
                    ro=seg_params.roughness_mm,
                    rating_code=rating_code,
                    th=th,
                )
                nodes.update(nodes_tmp)
                pipes.update(pipes_tmp)
                prev_node = nodes[node_name]
                prev_length = length
            diam_index += 1
        # If the current item indicates a loop connection. Then connect the previous
        # node to the loop cxn, and also add in the loop
        elif val.val_type == "loop":
            if same_node:
                nodes_tmp, pipes_tmp = add_pipe_segments(
                    p_name="loop_2_loop_cxn",
                    length_km=length,
                    diameter_mm=d_loop,
                    from_node=n_ds_in,
                    to_node=prev_node,
                    composition=seg_params.composition,
                    ro=seg_params.roughness_mm,
                    rating_code=rating_code,
                    th=th,
                )
                nodes.update(nodes_tmp)
                pipes.update(pipes_tmp)
            else:
                node_name = "loop_cxn"
                nodes[node_name] = bp_plc.Node(
                    name=node_name, composition=seg_params.composition
                )
                nodes_tmp, pipes_tmp = add_pipe_segments(
                    p_name="main_2_loop_cxn",
                    length_km=length - prev_length,
                    diameter_mm=seg_params.d_main_inner[diam_index][0],
                    from_node=prev_node,
                    to_node=nodes[node_name],
                    composition=seg_params.composition,
                    ro=seg_params.roughness_mm,
                    rating_code=rating_code,
                    th=th,
                )
                nodes.update(nodes_tmp)
                pipes.update(pipes_tmp)
                nodes_tmp, pipes_tmp = add_pipe_segments(
                    p_name="loop_2_loop_cxn",
                    length_km=length,
                    diameter_mm=d_loop,
                    from_node=n_ds_in,
                    to_node=nodes[node_name],
                    composition=seg_params.composition,
                    ro=seg_params.roughness_mm,
                    rating_code=rating_code,
                    th=th,
                )
                nodes.update(nodes_tmp)
                pipes.update(pipes_tmp)
                prev_node = nodes[node_name]
                prev_length = length

    end_node = prev_node

    # If a compressor exists in the segment
    if seg_params.seg_compressor:
        comp_orig = seg_params.seg_compressor[0]
        comp_name = "segment_compressor"

        # Keep the final node as is for pipe connections as is, but then add another node after.
        # Compressor will compress to this node. Demand node will be updated

        # Add node after compressor
        final_node_name = "final_node"
        nodes[final_node_name] = bp_plc.Node(
            name=final_node_name,
            composition=seg_params.composition,
        )
        # Update demand to be the final node

        final_demand_node_name = f"demand_{len(seg_params.offtakes_mdot) - 1}"
        prev_final_node = demands[final_demand_node_name].node
        demands[final_demand_node_name].node = nodes[final_node_name]
        # Add the compressor
        compressors[comp_name] = bp_plc.Compressor(
            name=comp_name,
            from_node=prev_final_node,
            to_node=nodes[final_node_name],
            pressure_out_mpa_g=seg_params.seg_compressor_pressure_out / gl.MPA2PA,
            original_rating_MW=comp_orig.original_rating_MW,
            fuel_extract=not comps_elec,
        )
        compressors[comp_name].eta_comp_s = comp_orig.eta_comp_s
        compressors[comp_name].eta_comp_s_elec = comp_orig.eta_comp_s_elec
        compressors[comp_name].eta_driver = comp_orig.eta_driver
        compressors[comp_name].eta_driver_elec = comp_orig.eta_driver_elec

    looping = BlendPATH_network(
        name="looping",
        pipes=pipes,
        nodes=nodes,
        demand_nodes=demands,
        supply_nodes=supplys,
        compressors=compressors,
        composition=seg_params.composition,
        thermo_curvefit=seg_params.thermo_curvefit,
        eos=seg_params.eos,
        ff_type=seg_params.ff_type,
        composition_tracking=seg_params.composition_tracking,
    )
    looping.blendH2(seg_params.composition.x["H2"])
    return looping, end_node, n_ds_in.connections["Pipe"], compressors


def get_segment_results(
    d_inner_mm: float,
    d_outer_mm: float,
    dn: float,
    loop_length: float,
    grade: str,
    nw: BlendPATH_network,
    loop_network: BlendPATH_network,
    sup_p: float,
    m_dot_seg: float,
    p_out: float,
    seg_params: "bp_mod_util.Segment_params",
) -> dict[str, Any]:
    """Calculate the LCOT and associated metrics after finding a solution

    Args:
        d_inner_mm (float): inner diameter [mm]
        d_outer_mm (float): outer diameter [mm]
        dn (float): nominal diameter [mm]
        loop_length (float): Length of parallel loop [km]
        grade (str): Steel grade
        nw (BlendPATH_network): Copied network
        loop_network (BlendPATH_network): Network of just looped segment
        sup_p (float): supply pressure
        m_dot_seg (float): Mass flow rate into segment
        p_out (float): Pressure at end of segment
        seg_params (Segment_params): Segment parameters

    Returns:
        dict[str, Any]: Collection of segment results
    """
    # Get material cost
    loop_cost = bp_pipe_cost.get_pipe_material_cost(
        cp=seg_params.costing_params,
        di_mm=d_inner_mm,
        do_mm=d_outer_mm,
        l_km=loop_length,
        grade=grade,
    )

    # Get other pipe costs
    new_pipe_cap = 0
    if loop_length > 0:
        anl_cap = bp_pipe_cost.get_pipe_other_cost(
            cp=seg_params.costing_params,
            d_mm=dn,
            l_km=loop_length,
            anl_types=bp_mod_util.get_pipe_cost_types(
                mod_type="pl",
                scenario_type=nw.scenario_type,
            ),
        )
        anl_cap_sum = sum(anl_cap.values())
        new_pipe_cap = anl_cap_sum + loop_cost

    # Check if supply compressor needed
    sn = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]
    orig_supply_pressure = min(sn.pressure_mpa, seg_params.design_pressure_MPa)

    add_supply_comp = False
    supply_comp_inputs = None
    if sn.node.name in seg_params.ps_nodes and orig_supply_pressure < sup_p:
        add_supply_comp = True
        supply_comp_inputs = bp_mod_util.Add_supply_comp_inputs(
            blend=sn.blend,
            composition=sn.node.composition,
            orig_supply_pressure=orig_supply_pressure,
            pressure=sup_p,
            thermo_curvefit=nw.thermo_curvefit,
            m_dot_seg=m_dot_seg,
        )

    mod_costing_params = bp_mod_util.get_mod_costing_params(
        network=loop_network,
        design_params=seg_params.design_params,
        costing_params=seg_params.costing_params,
        supply_comp_inputs=supply_comp_inputs,
    )

    price_breakdown, _ = bp_cost.calc_lcot(
        mod_costing_params=mod_costing_params,
        new_pipe_capex=new_pipe_cap,
        costing_params=seg_params.costing_params,
    )

    segment_lcot = price_breakdown["LCOT: Levelized cost of transport"]

    return {
        "lcot": segment_lcot,
        "mat_cost": loop_cost,
        "loop_length": loop_length,
        "add_supply_comp": add_supply_comp,
        "inlet_p": sup_p,
        "m_dot_in": m_dot_seg,
        "p_out": p_out,
    }


def get_next_l_loop(
    p_vals: list, l_vals: list, p_target: float, l_bounds: list
) -> float:
    if len(p_vals) > 1:
        if l_vals[-1] - l_vals[-2] == np.inf or p_vals[-1] == 0:
            l_loop = np.mean(l_vals[-2:])
            l_vals.pop()
            p_vals.pop()
        else:
            slope = (p_vals[-1] - p_vals[-2]) / (l_vals[-1] - l_vals[-2])
            intercept = p_vals[-1] - slope * l_vals[-1]
            l_loop = (p_target - intercept) / slope
            if l_loop <= 0:
                slope = (p_vals[-1] - p_vals[0]) / (l_vals[-1] - l_vals[0])
                intercept = p_vals[-1] - slope * l_vals[-1]
                l_loop = (p_target - intercept) / slope
        return l_loop
    else:
        return np.mean(l_bounds)
