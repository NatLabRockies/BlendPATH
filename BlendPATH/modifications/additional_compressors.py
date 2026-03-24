import logging
from dataclasses import dataclass

import cantera as ct
import numpy as np

import BlendPATH.costing.costing as bp_cost
import BlendPATH.file_writing.mod_file_out_util as mod_file_util
import BlendPATH.Global as gl
import BlendPATH.modifications.mod_util as bp_mod_util
import BlendPATH.network.pipeline_components.cantera_util as ctu
from BlendPATH.network import pipeline_components as bp_plc
from BlendPATH.network.BlendPATH_network import BlendPATH_network
from BlendPATH.scenario_helper import Design_params

logger = logging.getLogger(__name__)


def additional_compressors(
    network: BlendPATH_network,
    design_params: Design_params = None,
    design_option: str = "b",
    new_filename: str = "modified",
    costing_params: bp_cost.Costing_params = None,
    allow_compressor_bypass: bool = False,
) -> tuple:
    """
    Modify network with additional compressors method
    """

    # Copy the network
    nw = bp_mod_util.copy_network(network=network, design_params=design_params)

    # Set compression ratio variables
    max_CR = design_params.max_CR
    n_cr = len(max_CR)
    # Set final outlet pressure
    final_outlet_pressure = design_params.final_outlet_pressure_mpa_g

    # For new compressors (adl compressors + supply compressor)
    assign_eta_s, assign_eta_driver = bp_mod_util.compressor_eta(
        design_params=design_params
    )

    # Save number of pipe segments
    n_ps = len(nw.pipe_segments)

    # Initialize lists
    res = [{i: [] for i in range(len(nw.pipe_segments))} for _ in range(n_cr)]

    # Loop thru compression ratios
    for cr_i, CR_ratio in enumerate(max_CR):
        # Loop thru segments (in reverse)
        prev_ASME_pressure = -1
        m_dot_in_prev = nw.pipe_segments[-1].mdot_out
        for ps_i, ps in reversed(list(enumerate(nw.pipe_segments))):
            # setup list of supply pressure to investigate with a supply compressor
            supp_p_list = bp_mod_util.get_supply_pressure_list(nw=nw, ps=ps, ps_i=ps_i)

            # Set up pressure bounds for segment analysis
            pressure_in_Pa = ps.design_pressure_MPa * gl.MPA2PA
            if ps_i == n_ps - 1:
                pressure_out_Pa = final_outlet_pressure * gl.MPA2PA
            else:
                pressure_out_Pa = prev_ASME_pressure / CR_ratio

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

            # Loop through supply pressures
            supp_p_min_res = []
            for sup_p in supp_p_list:
                # Get number of compressors and the lengths
                (
                    n_comps,
                    l_comps,
                    m_dot_seg,
                    addl_comps,
                ) = get_num_compressors(
                    p_in=sup_p * gl.MPA2PA,
                    seg_params=seg_params,
                )
                if n_comps == np.inf:
                    continue

                supp_p_min_res.append(
                    get_segment_results(
                        nw=nw,
                        addl_comps=addl_comps,
                        sup_p=sup_p,
                        m_dot_seg=m_dot_seg,
                        seg_params=seg_params,
                        l_comps=l_comps,
                    )
                )

            # Get minimum solution per supply pressure
            if not supp_p_min_res:
                break
            lcot_p_spply = [x["lcot"] for x in supp_p_min_res]
            idxmin_lcot_p_supp = lcot_p_spply.index(min(lcot_p_spply))
            # Asssign values for lowest LCOT per supply pressure
            res[cr_i][ps_i] = AC_res(
                min_sol_per_supp_p=supp_p_min_res[idxmin_lcot_p_supp]
            )

            logger.info(
                f"Solution found for CR_ratio={CR_ratio}, segment={ps_i}, N_comps = {len(supp_p_min_res[idxmin_lcot_p_supp]['l_comps'])}, Inlet pressure = {supp_p_min_res[idxmin_lcot_p_supp]['inlet_p']}, LCOT = {supp_p_min_res[idxmin_lcot_p_supp]['lcot']}"
            )

            # Assign flow rate and pressure to next segment
            m_dot_in_prev = m_dot_seg
            prev_ASME_pressure = pressure_in_Pa

    # Get the results for the CR with the lowest LCOT across segments
    cr_lcot_sums = [
        sum(y.cost if y else np.inf for y in x.values()) for x in res
    ]  # [sum(x) for x in cr_lcot]
    cr_min_index = cr_lcot_sums.index(min(cr_lcot_sums))
    l_comps_ps = [x.l_comps for x in res[cr_min_index].values()]
    n_comps_ps = [len(x) for x in l_comps_ps]
    add_supply_comp = res[cr_min_index][0].add_supply_comp
    supply_p = res[cr_min_index][0].inlet_p

    logger.info(
        f"Solution taken: CR_ratio={max_CR[cr_min_index]}, N_comps={n_comps_ps}, Segment LCOT sum={cr_lcot_sums[cr_min_index]}, supply compressor={add_supply_comp}, supply pressure={supply_p}"
    )

    # REMAKE file
    make_result_file(
        filename=new_filename,
        nw=nw,
        n_comps_ps=n_comps_ps,
        l_comps_ps=l_comps_ps,
        supply_p=supply_p,
        add_supply_comp=add_supply_comp,
        design_params=design_params,
        assign_eta_s=assign_eta_s,
        assign_eta_driver=assign_eta_driver,
    )

    return {}, {"D_S_G": [], "length": [], "total cost": 0}, l_comps_ps


def make_compressor_network(
    n_comps: int, p_in: float, seg_params: bp_mod_util.Segment_params
) -> tuple:
    """
    Make a new network with compressors added to segment
    """
    # Make inlet node
    n_ds_in = bp_plc.Node(name="in", composition=seg_params.composition, index=0)
    nodes = {n_ds_in.name: n_ds_in}
    prev_node = n_ds_in
    prev_length = 0

    pipes = {}
    compressors = {}
    demands = {}
    supplys = {
        "supply": bp_plc.Supply_node(
            node=n_ds_in,
            pressure_mpa=p_in / gl.MPA2PA,
            blend=seg_params.composition.x["H2"],
        )
    }

    l_between = seg_params.l_total / (n_comps + 1)
    l_comps = [l_between * (x + 1) for x in range(n_comps)]

    all_lengths = bp_mod_util.get_sorted_lengths(
        d_main=[(seg_params.d_main_inner, seg_params.l_total)],
        l_loop=0,
        all_mdot=seg_params.offtakes_mdot,
        offtakes=seg_params.offtakes_length,
        l_comps=l_comps,
        hhv=seg_params.hhv,
    )

    node_index = 1  # Since supply node was alread added
    pipe_index = 0
    demand_index = 0
    comp_index = 0
    for val in all_lengths:
        length = val.length_km
        same_node = length - prev_length == 0
        if val.val_type == "offtake":
            if same_node:
                d_name = f"demand_{demand_index}"
                demands[d_name] = bp_plc.Demand_node(
                    node=prev_node,
                    flowrate_MW=val.mw,
                )
            else:
                name = f"ot_{node_index - 1}"
                nodes[name] = bp_plc.Node(name=name, composition=seg_params.composition)
                node_index += 1
                d_name = f"demand_{demand_index}"
                demands[d_name] = bp_plc.Demand_node(
                    node=nodes[name],
                    flowrate_MW=val.mw,
                )

                # MAKE PIPE
                p_name = f"pipe_{pipe_index}"
                pipes[p_name] = bp_plc.Pipe(
                    name=p_name,
                    from_node=prev_node,
                    to_node=nodes[name],
                    diameter_mm=seg_params.d_main_inner,
                    length_km=length - prev_length,
                    roughness_mm=seg_params.roughness_mm,
                )
                prev_length = length
                pipe_index += 1
                prev_node = nodes[name]
            demand_index += 1

        elif val.val_type == "comp":
            if same_node:
                # # Make compressor to node
                name_from = f"c_{node_index - 1}"
                nodes[name_from] = bp_plc.Node(
                    name=name_from, composition=seg_params.composition
                )
                node_index += 1
                # Make comp
                comp_name = f"c_{comp_index}"
                compressors[comp_name] = bp_plc.Compressor(
                    name=comp_name,
                    from_node=nodes[name_from],
                    to_node=prev_node,
                    pressure_out_mpa_g=seg_params.design_pressure_MPa,
                    original_rating_MW=0,
                    fuel_extract=not seg_params.new_comps_elec,
                )
                pipes[p_name].to_node = nodes[name_from]
                if not seg_params.new_comps_elec:
                    compressors[comp_name].eta_comp_s = seg_params.eta_s
                    compressors[comp_name].eta_driver = seg_params.eta_driver
                else:
                    compressors[comp_name].eta_comp_s_elec = seg_params.eta_s
                    compressors[comp_name].eta_driver_elec = seg_params.eta_driver

                # No update to prev node
                # prev_node = prev_node
            else:
                # Make compressor from node
                name_from = f"c_{node_index - 1}"
                nodes[name_from] = bp_plc.Node(
                    name=name_from, composition=seg_params.composition
                )
                node_index += 1

                # Make pipe to compressor
                p_name = f"pipe_{pipe_index}"
                pipes[p_name] = bp_plc.Pipe(
                    name=p_name,
                    from_node=prev_node,
                    to_node=nodes[name_from],
                    diameter_mm=seg_params.d_main_inner,
                    length_km=length - prev_length,
                    roughness_mm=seg_params.roughness_mm,
                )
                prev_length = length
                pipe_index += 1

                # # Make compressor to node
                name_to = f"c_{node_index - 1}"
                nodes[name_to] = bp_plc.Node(
                    name=name_to, composition=seg_params.composition
                )
                node_index += 1

                # MAKE COMPRESSOR
                comp_name = f"c_{comp_index}"
                compressors[comp_name] = bp_plc.Compressor(
                    name=comp_name,
                    from_node=nodes[name_from],
                    to_node=nodes[name_to],
                    pressure_out_mpa_g=seg_params.design_pressure_MPa,
                    original_rating_MW=0,
                    fuel_extract=not seg_params.new_comps_elec,
                )
                if not seg_params.new_comps_elec:
                    compressors[comp_name].eta_comp_s = seg_params.eta_s
                    compressors[comp_name].eta_driver = seg_params.eta_driver
                else:
                    compressors[comp_name].eta_comp_s_elec = seg_params.eta_s
                    compressors[comp_name].eta_driver_elec = seg_params.eta_driver

                prev_node = nodes[name_to]
            comp_index += 1

    # If a compressor exists in the segment
    if seg_params.seg_compressor:
        comp_orig = seg_params.seg_compressor[0]
        comp_name = "segment_compressor"

        # Add node after compressor
        final_node_name = "final_node"
        nodes[final_node_name] = bp_plc.Node(
            name=final_node_name,
            composition=seg_params.composition,
        )

        # Update demand to be the final node
        final_demand_node_name = f"demand_{demand_index - 1}"
        prev_final_node = demands[final_demand_node_name].node
        demands[final_demand_node_name].node = nodes[final_node_name]

        # Add the compressor
        compressors[comp_name] = bp_plc.Compressor(
            name=comp_name,
            from_node=prev_final_node,
            to_node=nodes[final_node_name],
            pressure_out_mpa_g=seg_params.seg_compressor_pressure_out / gl.MPA2PA,
            original_rating_MW=comp_orig.original_rating_MW,
            fuel_extract=not seg_params.comps_elec,
        )
        compressors[comp_name].eta_comp_s = comp_orig.eta_comp_s
        compressors[comp_name].eta_comp_s_elec = comp_orig.eta_comp_s_elec
        compressors[comp_name].eta_driver = comp_orig.eta_driver
        compressors[comp_name].eta_driver_elec = comp_orig.eta_driver_elec

    addl_comp_network = BlendPATH_network(
        name="addl_comps",
        pipes=pipes,
        nodes=nodes,
        demand_nodes=demands,
        supply_nodes=supplys,
        compressors=compressors,
        composition=seg_params.composition,
        thermo_curvefit=seg_params.thermo_curvefit,
        composition_tracking=seg_params.composition_tracking,
        eos=seg_params.eos,
        ff_type=seg_params.ff_type,
    )

    end_node = nodes[f"ot_{node_index - 2}"]
    return (
        addl_comp_network,
        end_node,
        l_comps,
        addl_comp_network.pipes[list(pipes.keys())[0]],
    )


def get_num_compressors(
    p_in: float, seg_params: bp_mod_util.Segment_params
) -> tuple[list[int], list[float], float, BlendPATH_network]:
    """Determine the number of compressors needed

    Args:
        p_in (float): Inlet pressure [Pa]
        seg_params (bp_mod_util.Segment_params): Segment static parameters

    Returns:
        tuple[list[int], list[float], float, BlendPATH_network]: n_comps, l_comps, m_dot, new_segment_network
    """

    # Quick algebraic solve for guessing number of compressors
    n_comps = guess_n_comps(p_in=p_in, seg_params=seg_params)
    max_n_comps = 10 if seg_params.comps_elec else 50

    sols = []
    n_comps_tried = []
    relax = 1.5

    # Loop through chaning compressors amounts until least amount of compressors
    # that satisfies constraints is found
    while (
        0 <= n_comps < 200
        and n_comps not in n_comps_tried
        and len(n_comps_tried) < max_n_comps
    ):
        addl_comps, end_node, l_comps, pipe_in = make_compressor_network(
            n_comps=n_comps, p_in=p_in, seg_params=seg_params
        )
        try:
            addl_comps.solve(relax, cr_max=seg_params.CR_ratio)
        except ValueError as err_val:
            if err_val.args[0] in [
                "Negative pressure",
                "Pressure below threshold",
                f"Could not converge in {gl.MAX_ITER} iterations",
            ]:
                # if negative pressure than increase compressors
                n_comps = update_n_comps(n_list=n_comps_tried, n_comps=n_comps, incr=1)
            if err_val.args[0] == "GERG EOS could not evaluate":
                raise RuntimeError("GERG EOS could not evaluate")
        except ct.CanteraError:
            relax *= 1.05
            continue
        else:
            # Check if outlet pressure satisfies target pressure
            if end_node.pressure >= seg_params.p_out_target:
                # Check if compressor ratios are too high
                if any(
                    [
                        cs.compression_ratio > seg_params.CR_ratio
                        for cs in addl_comps.compressors.values()
                    ]
                ):
                    n_comps = update_n_comps(
                        n_list=n_comps_tried, n_comps=n_comps, incr=1
                    )
                    continue

                # Else, solution is found. Note it, and check if less compressors is possible
                sols.append(
                    (
                        n_comps,
                        l_comps,
                        pipe_in.m_dot,
                        addl_comps,
                    )
                )
                n_comps = update_n_comps(n_list=n_comps_tried, n_comps=n_comps, incr=-1)
            else:
                n_comps = update_n_comps(n_list=n_comps_tried, n_comps=n_comps, incr=1)
    if not sols:
        return (np.inf, None, None, None)
    return sols[-1]


def update_n_comps(n_list: list[int], n_comps: int, incr: int) -> int:
    """Simplified location to update n_comps

    Args:
        n_list (list[int]): n_comps_tried list
        n_comps (int): current number of compressors
        incr (int): +1 or -1

    Returns:
        int: New n_comps
    """
    n_list.append(n_comps)
    n_comps += incr
    return n_comps


def make_result_file(
    filename: str,
    nw: BlendPATH_network,
    n_comps_ps: list,
    l_comps_ps: list,
    supply_p: float,
    add_supply_comp: bool,
    design_params: Design_params,
    assign_eta_s: float,
    assign_eta_driver: float,
) -> None:
    new_pipes = {x: [] for x in mod_file_util.pipes_cols()}
    new_comps = {x: [] for x in mod_file_util.comps_cols()}
    new_nodes = {x: [] for x in mod_file_util.nodes_cols()}

    for ps_i, ps_comp in enumerate(n_comps_ps):
        # Get segment
        ps = nw.pipe_segments[ps_i]
        p_max_seg = ps.design_pressure_MPa

        # Get lengths of segment
        l_comps = l_comps_ps[ps_i]

        # Nodes
        len_added = 0
        comp_len_i = 0
        pipe_len_cum = 0
        for pipe in ps.pipes:
            pipe_len_cum += pipe.length_km
            pipe_len_remaining = pipe_len_cum

            pipe_segmented = False

            pipe_from_node = pipe.from_node.name
            new_nodes["node_name"].append(pipe_from_node)
            new_nodes["p_max_mpa_g"].append(p_max_seg)

            # If length passes where the compresser, add the compressor
            while comp_len_i < ps_comp and l_comps[comp_len_i] < pipe_len_remaining:
                pipe_segmented = True

                # Check if it overlaps with already existing node:
                if abs(l_comps[comp_len_i] - len_added) < 0.01:
                    from_comp_name = f"N_pre_C_{ps_i}_{comp_len_i}"

                    # Make existing node after compressor
                    to_comp_name = new_nodes["node_name"][-1]
                    new_pipes["to_node"][-1] = from_comp_name

                    new_nodes["node_name"].append(from_comp_name)
                    new_nodes["p_max_mpa_g"].append(p_max_seg)

                else:
                    # Add node before compressor
                    from_comp_name = f"N_pre_C_{ps_i}_{comp_len_i}"
                    new_nodes["node_name"].append(from_comp_name)
                    new_nodes["p_max_mpa_g"].append(p_max_seg)

                    # Add pipe to compressor
                    pipe_name = f"{pipe.name}_pre_C_{ps_i}_{comp_len_i}"
                    new_pipes["pipe_name"].append(pipe_name)
                    new_pipes["from_node"].append(pipe_from_node)
                    new_pipes["to_node"].append(from_comp_name)  #
                    new_pipes["length_km"].append(l_comps[comp_len_i] - len_added)
                    new_pipes["roughness_mm"].append(pipe.roughness_mm)
                    new_pipes["diameter_mm"].append(pipe.diameter_mm)
                    new_pipes["thickness_mm"].append(pipe.thickness_mm)
                    new_pipes["rating_code"].append(pipe.grade)

                    # Add node after compressor
                    to_comp_name = f"N_post_C_{ps_i}_{comp_len_i}"
                    new_nodes["node_name"].append(to_comp_name)
                    new_nodes["p_max_mpa_g"].append(p_max_seg)

                # Add compressor
                comp_name = f"C_{ps_i}_{comp_len_i}"
                new_comps["compressor_name"].append(comp_name)
                new_comps["from_node"].append(from_comp_name)
                new_comps["to_node"].append(to_comp_name)
                new_comps["pressure_out_mpa_g"].append(p_max_seg)
                new_comps["rating_MW"].append(0)
                new_comps["extract_fuel"].append(not design_params.new_comp_elec)
                new_comps["eta_s"].append(assign_eta_s)
                new_comps["eta_driver"].append(
                    "" if np.isnan(assign_eta_driver) else assign_eta_driver
                )

                # Update latest node to the outlet of new compressor
                pipe_from_node = to_comp_name

                #
                len_added = l_comps[comp_len_i]
                # increase comp number
                comp_len_i += 1

            # Else add the pipe as usual
            else:
                pipe_name = pipe.name
                from_node = pipe.from_node.name
                pipe_len_final = pipe.length_km
                if pipe_segmented:
                    pipe_name = f"{pipe.name}_remaining"
                    from_node = pipe_from_node
                    pipe_len_final = pipe_len_remaining - len_added
                if pipe_len_final > 0.01:
                    # Add pipes as normal
                    new_pipes["pipe_name"].append(pipe_name)
                    new_pipes["from_node"].append(from_node)
                    new_pipes["to_node"].append(pipe.to_node.name)
                    new_pipes["length_km"].append(pipe_len_final)
                    new_pipes["roughness_mm"].append(pipe.roughness_mm)
                    new_pipes["diameter_mm"].append(pipe.diameter_mm)
                    new_pipes["thickness_mm"].append(pipe.thickness_mm)
                    new_pipes["rating_code"].append(pipe.grade)
                    # Update total pipe length
                    len_added += pipe_len_final
                else:
                    new_nodes["node_name"].pop()
                    new_nodes["p_max_mpa_g"].pop()
                    new_comps["to_node"][-1] = pipe.to_node.name
                    pass

        new_nodes["node_name"].append(pipe.to_node.name)
        new_nodes["p_max_mpa_g"].append(p_max_seg)

    # Add existing compressors
    for comp in nw.compressors.values():
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

        # Assume to_node has to be the outlet pressure
        p_max = np.inf
        for pipe in comp.to_node.connections["Pipe"]:
            for key, ps in enumerate(nw.pipe_segments):
                if pipe in ps.pipes:
                    p_max = min(p_max, ps.design_pressure_MPa)

        new_comps["pressure_out_mpa_g"].append(p_max)

    # Add supply
    new_supply = {x: [] for x in mod_file_util.supply_cols()}
    for supply in nw.supply_nodes.values():
        new_supply["supply_name"].append(supply.name)
        new_supply["node_name"].append(supply.node.name)
        new_supply["pressure_mpa_g"].append(supply_p)
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
        for pipe in supply.node.connections["Pipe"]:
            p_max = min(pipe.design_pressure_MPa, p_max)
        new_comps["pressure_out_mpa_g"].insert(0, supply_p)

        # Add new node
        new_nodes["node_name"].insert(0, "Supply compressor from_node")
        new_nodes["p_max_mpa_g"].insert(0, p_max)

        # Update supply node
        new_supply["node_name"][-1] = "Supply compressor from_node"
        new_supply["pressure_mpa_g"][-1] = sn.pressure_mpa
        new_supply["blend"][-1] = nw.composition.x["H2"]

    # Make new demands
    new_demand = {x: [] for x in mod_file_util.demand_cols()}
    for demand in nw.demand_nodes.values():
        new_demand["demand_name"].append(demand.name)
        new_demand["node_name"].append(demand.node.name)
        new_demand["flowrate_MW"].append(demand.flowrate_MW)

    # Composition
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
        sort_comps=True,
    )


def guess_n_comps(p_in: float, seg_params: bp_mod_util.Segment_params):

    # Guess n comps
    ctu.gas.TPX = gl.T_FIXED, ct.one_atm, seg_params.composition.x_str
    mw = ctu.gas.mean_molecular_weight
    zrt = 1 * ctu.gas_constant * gl.T_FIXED
    p_val = (p_in**2 - (p_in / seg_params.CR_ratio) ** 2) ** 0.5
    f = 0.01
    d_m = seg_params.d_main_inner * gl.MM2M
    area = np.pi * d_m**2 / 4
    l_seg = mw / zrt * d_m / f * (sum(seg_params.offtakes_mdot) / area / p_val) ** -2

    max_comps = 200
    return min(max([int(seg_params.l_total / (l_seg / gl.KM2M)), 0]), max_comps - 1)


def get_segment_results(
    nw: BlendPATH_network,
    addl_comps: BlendPATH_network,
    sup_p: float,
    m_dot_seg: float,
    seg_params: bp_mod_util.Segment_params,
    l_comps: list[float],
):
    supply_comp_inputs = None
    add_supply_comp = False
    sn = nw.supply_nodes[list(nw.supply_nodes.keys())[0]]
    orig_supply_pressure = min(sn.pressure_mpa, seg_params.design_pressure_MPa)
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
        network=addl_comps,
        design_params=seg_params.design_params,
        costing_params=seg_params.costing_params,
        supply_comp_inputs=supply_comp_inputs,
    )

    price_breakdown, _ = bp_cost.calc_lcot(
        mod_costing_params=mod_costing_params,
        new_pipe_capex=0,
        costing_params=seg_params.costing_params,
    )

    segment_lcot = price_breakdown["LCOT: Levelized cost of transport"]

    return {
        "lcot": segment_lcot,
        "add_supply_comp": add_supply_comp,
        "inlet_p": sup_p,
        "m_dot_in": m_dot_seg,
        "l_comps": l_comps,
    }


@dataclass
class AC_res:
    min_sol_per_supp_p: list

    def __post_init__(self):
        self.l_comps = self.min_sol_per_supp_p["l_comps"]
        self.cost = self.min_sol_per_supp_p["lcot"]
        self.add_supply_comp = self.min_sol_per_supp_p["add_supply_comp"]
        self.inlet_p = self.min_sol_per_supp_p["inlet_p"]
