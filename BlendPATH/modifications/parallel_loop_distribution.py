from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

import BlendPATH.costing.costing as bp_cost
import BlendPATH.file_writing.mod_file_out_util as mod_file_util
import BlendPATH.Global as gl
import BlendPATH.modifications.mod_util as bp_mod_util
import BlendPATH.network.pipeline_components as bp_plc
from BlendPATH.modifications.parallel_loop import make_loop_network
from BlendPATH.network.BlendPATH_network import BlendPATH_network
from BlendPATH.scenario_helper import Design_params
from BlendPATH.util.pipe_assessment import Design_options

if TYPE_CHECKING:
    from BlendPATH.network.pipeline_components import Pipe

SUFFIX = "_mod_looped"


def parallel_loop_distribution(
    network: BlendPATH_network,
    design_option: Design_options,
    new_filename: str = "modified",
    design_params: Design_params = None,
    costing_params: bp_cost.Costing_params = None,
    allow_compressor_bypass: bool = False,
) -> tuple:

    updated_pipes, updated_pipe_instances, looped_parent_pipes = check_high_dpdx(
        network=network,
        design_option=design_option,
        design_params=design_params,
    )

    original_diameters = {}

    # Check for pressure violations
    for pipe_segment in network.pipe_segments:
        pipe_names_to_swap, seg_max_pressure_MPa = pipe_segment.violating_pipes()
        for pipe_name in pipe_names_to_swap:
            pipe_params_in = build_pipe_params(
                pipe=network.pipes[pipe_name],
                looped_parent_pipes=looped_parent_pipes,
                network=network,
                design_option=design_option,
                design_params=design_params,
                pressure_rating_MPa_override=seg_max_pressure_MPa,
            )
            original_diameters[pipe_name] = (
                network.pipes[pipe_name].diameter_mm,
                network.pipes[pipe_name].NPS,
                network.pipes[pipe_name].DN,
                network.pipes[pipe_name].thickness_mm,
                network.pipes[pipe_name].design_pressure_MPa,
            )
            (
                network.pipes[pipe_name].diameter_mm,
                network.pipes[pipe_name].NPS,
                network.pipes[pipe_name].DN,
                network.pipes[pipe_name].thickness_mm,
                network.pipes[pipe_name].design_pressure_MPa,
            ) = bp_mod_util.get_next_diameter(pipe_params_in=pipe_params_in)

            updated_pipes[network.pipes[pipe_name].name] = build_new_pipes(
                pipe=network.pipes[pipe_name],
                p_rating=network.pipes[pipe_name].design_pressure_MPa,
            )

            # TODO: check if it is looped already, then the loop may need updating also
            if (
                f"{pipe_name}{SUFFIX}" in updated_pipes
                and updated_pipe_instances[f"{pipe_name}{SUFFIX}"].design_pressure_MPa
                < seg_max_pressure_MPa
            ):
                # Check MAOP
                # Get next diameter
                (
                    updated_pipe_instances[[f"{pipe_name}{SUFFIX}"]].diameter_mm,
                    updated_pipe_instances[[f"{pipe_name}{SUFFIX}"]].NPS,
                    updated_pipe_instances[[f"{pipe_name}{SUFFIX}"]].DN,
                    updated_pipe_instances[[f"{pipe_name}{SUFFIX}"]].thickness_mm,
                    updated_pipe_instances[
                        [f"{pipe_name}{SUFFIX}"]
                    ].design_pressure_MPa,
                ) = bp_mod_util.get_next_diameter(
                    pipe_params_in=build_pipe_params(
                        pipe=updated_pipe_instances[[f"{pipe_name}{SUFFIX}"]],
                        looped_parent_pipes=looped_parent_pipes,
                        network=network,
                        design_option=design_option,
                        design_params=design_params,
                        pressure_rating_MPa_override=seg_max_pressure_MPa,
                    )
                )

    velocity_violtations = True
    iters = 0
    while velocity_violtations:
        # Reset velocity violation flag
        velocity_violtations = False
        # Solve network
        network.solve()

        # Keep track of mod_looped pipes already updated.
        # it is possible that both the main and the loop are violating
        # then the loop diameter would increase twice
        already_updated_loops = []

        # Loop over all pipes to check violations, and modify if needed
        for pipe in network.pipes.values():
            # Skip over pipe if no violations
            if bp_mod_util.check_velocity_violation(
                pipe, design_params.max_dist_velocity
            ):
                continue
            # Violation must be present if this point is reached
            velocity_violtations = True

            pipe_params = build_pipe_params(
                pipe=pipe,
                looped_parent_pipes=looped_parent_pipes,
                network=network,
                design_params=design_params,
                design_option=design_option,
            )

            # Get the name of the looped pipe
            # If it already exists (already looped), then the same name
            looped_name = get_looped_name(
                pipe=pipe,
                updated_pipe_instances=updated_pipe_instances,
                looped_parent_pipes=looped_parent_pipes,
            )
            # If the loop has already been updated, then skip
            if looped_name in already_updated_loops:
                continue

            # Create new pipe instance. Cannot add the network.pipes since we
            # are currently looping through it.
            # DN is updated separately
            update_pipes(
                pipe=pipe,
                looped_name=looped_name,
                updated_pipe_instances=updated_pipe_instances,
                pipe_params=pipe_params,
            )

            # Add to list of updated loops
            already_updated_loops.append(looped_name)

        # After adding all pipes, then add to network
        # Update pipe designations and connections
        for pipe_name, pipe_i in updated_pipe_instances.items():
            network.pipes[pipe_name] = pipe_i

            updated_pipes[pipe_name] = bp_mod_util.New_pipes(
                inner_diameter=pipe_i.diameter_mm,
                dn=pipe_i.DN,
                nps=pipe_i.NPS,
                th=pipe_i.thickness_mm,
                length=pipe_i.length_km,
                grade=pipe_i.rating_code,
                name=pipe_name,
                from_node=pipe_i.from_node.name,
                to_node=pipe_i.to_node.name,
                roughness_mm=pipe_i.roughness_mm,
                pressure=pipe_i.design_pressure_MPa,
            )

        update_network(network=network)

        # Prevent run away while loop
        iters += 1
        if iters > 10:
            raise RuntimeError("Solution not found after 10 iterations")

    # reset_network to original values
    reset_network(
        updated_pipe_instances=updated_pipe_instances,
        network=network,
        original_diameters=original_diameters,
    )

    combined_pipe = bp_mod_util.get_polymer_combined_pipe_stats(
        updated_pipes=updated_pipes, costing_params=costing_params
    )

    make_result_file(filename=new_filename, network=network, update_pipes=updated_pipes)

    return updated_pipes, combined_pipe, []


def parallel_loop_distribution_alt(
    network: BlendPATH_network,
    design_option: Design_options,
    new_filename: str = "modified",
    design_params: Design_params = None,
    costing_params: bp_cost.Costing_params = None,
) -> tuple:
    updated_pipes = {}
    # updated_pipe_instances = {}

    # looped_parent_pipes = {}
    network.solve()

    # Loop over all pipes to check violations, and modify if needed
    for pipe in network.pipes.values():
        # Skip over pipe if no violations
        if bp_mod_util.check_velocity_violation(pipe, design_params.max_dist_velocity):
            continue

        # Create mini network
        pipe_params = bp_mod_util.pipe_diam_params(
            inner_diameter_mm=pipe.diameter_mm,
            nps=pipe.NPS,
            dn=pipe.DN,
            pressure_rating_MPa=pipe.design_pressure_MPa,
            material=pipe.material,
            grade=getattr(pipe, "grade", None),
            polymer=getattr(pipe, "polymer", None),
            polymer_designation=getattr(pipe, "polymer_designation", None),
            design_option=design_option,
            design_params=design_params,
        )
        initial_diam, _, _, _, _ = bp_mod_util.get_next_diameter(
            pipe_params_in=pipe_params,
            allow_existing_geom=True,
        )
        p_max = max(pipe.to_node.pressure, pipe.from_node.pressure)
        loop_network, _, _, _ = make_loop_network(
            l_loop=pipe.length_km,
            composition=network.composition,
            p_in=p_max / gl.MPA2PA,
            offtakes=[pipe.length_km],
            all_mdot=[abs(pipe.m_dot)],
            d_main=pipe.diameter_mm,
            d_loop=initial_diam,
            roughness_mm=pipe.roughness_mm,
            seg_compressor=[],
            prev_ASME_pressure=np.inf,
            comps_elec=False,
            eos=network.eos,
            thermo_curvefit=network.thermo_curvefit,
            composition_tracking=network.composition_tracking,
            ff_type=network.ff_type,
            rating_code=pipe.rating_code,
        )
        loop_network.pipes["loop_2_l_cxn"].NPS = pipe.NPS
        loop_network.pipes["loop_2_l_cxn"].DN = pipe.DN
        loop_network.pipes["loop_2_l_cxn"].thickness_mm = pipe.thickness_mm
        loop_network.pipes[
            "loop_2_l_cxn"
        ].design_pressure_MPa = pipe.design_pressure_MPa
        loop_network.scenario_type = network.scenario_type
        # Get additional names if subsegments
        loop_pipe_names = []
        for looping_pipe in loop_network.pipes.values():
            if looping_pipe.name.startswith("loop_2_l_cxn"):
                loop_pipe_names.append(looping_pipe.name)

        # Loop until pipe velocity is no longer violating
        velocity_violtations = True
        iters = 0
        while velocity_violtations:
            loop_network.solve()

            velocity_violtations = False
            for looping_pipe in loop_network.pipes.values():
                # Skip over pipe if no violations
                if bp_mod_util.check_velocity_violation(
                    looping_pipe, design_params.max_dist_velocity
                ):
                    continue
                # Violation must be present if this point is reached
                velocity_violtations = True
                break

            if not velocity_violtations:
                break
            # Get next diameter
            pipe_params.inner_diameter_mm = loop_network.pipes[
                "loop_2_l_cxn"
            ].diameter_mm
            pipe_params.nps = loop_network.pipes["loop_2_l_cxn"].NPS
            pipe_params.dn = loop_network.pipes["loop_2_l_cxn"].DN
            diam, nps, dn, th, p_rating = bp_mod_util.get_next_diameter(
                pipe_params_in=pipe_params
            )
            # Increase diameter of loop
            for pipe_name in loop_pipe_names:
                loop_network.pipes[pipe_name].diameter_mm = diam
                loop_network.pipes[pipe_name].NPS = nps
                loop_network.pipes[pipe_name].DN = dn
                loop_network.pipes[pipe_name].thickness_mm = th
                loop_network.pipes[pipe_name].design_pressure_MPa = p_rating

            # Limit iterations in while loop
            iters += 1
            if iters > 20:
                break

        updated_pipes[f"{pipe.name}{SUFFIX}"] = bp_mod_util.New_pipes(
            inner_diameter=loop_network.pipes["loop_2_l_cxn"].diameter_mm,
            dn=loop_network.pipes["loop_2_l_cxn"].DN,
            nps=loop_network.pipes["loop_2_l_cxn"].NPS,
            th=loop_network.pipes["loop_2_l_cxn"].thickness_mm,
            length=pipe.length_km,
            grade=pipe.rating_code,
            name=f"{pipe.name}{SUFFIX}",
            from_node=pipe.from_node.name,
            to_node=pipe.to_node.name,
            roughness_mm=pipe.roughness_mm,
            pressure=loop_network.pipes["loop_2_l_cxn"].design_pressure_MPa,
        )
    combined_pipe = bp_mod_util.get_polymer_combined_pipe_stats(
        updated_pipes=updated_pipes, costing_params=costing_params
    )

    make_result_file(filename=new_filename, network=network, update_pipes=updated_pipes)

    return updated_pipes, combined_pipe, []


def reset_network(
    updated_pipe_instances: dict,
    network: BlendPATH_network,
    original_diameters: dict[str, tuple[float, float, float, float]],
):
    for pipe_name in updated_pipe_instances:
        if pipe_name in network.pipes:
            del network.pipes[pipe_name]
    for pipe_name, pipe_vals in original_diameters.items():
        (
            network.pipes[pipe_name].diameter_mm,
            network.pipes[pipe_name].NPS,
            network.pipes[pipe_name].DN,
            network.pipes[pipe_name].thickness_mm,
            network.pipes[pipe_name].design_pressure_MPa,
        ) = pipe_vals

    update_network(network=network)


def update_network(network: BlendPATH_network) -> None:
    """
    Update pipe designations and connections
    """
    network.update_pipe_designation()
    network.assign_nodes()
    network.assign_connections()


def make_result_file(filename: str, network: BlendPATH_network, update_pipes: dict):
    new_pipes = defaultdict(list)
    for pipe in network.parent_pipes.values():
        # These values don't change from the DR method
        new_pipes["pipe_name"].append(pipe.name)
        new_pipes["from_node"].append(pipe.from_node.name)
        new_pipes["to_node"].append(pipe.to_node.name)
        new_pipes["diameter_mm"].append(pipe.diameter_mm)
        new_pipes["length_km"].append(pipe.length_km)
        new_pipes["roughness_mm"].append(pipe.roughness_mm)
        new_pipes["thickness_mm"].append(pipe.thickness_mm)
        new_pipes["rating_code"].append(pipe.rating_code)
    for pipe in update_pipes.values():
        new_pipes["pipe_name"].append(pipe.name)
        new_pipes["from_node"].append(pipe.from_node)
        new_pipes["to_node"].append(pipe.to_node)
        new_pipes["diameter_mm"].append(pipe.inner_diameter)
        new_pipes["length_km"].append(pipe.length)
        new_pipes["roughness_mm"].append(pipe.roughness_mm)
        new_pipes["thickness_mm"].append(pipe.th)
        new_pipes["rating_code"].append(pipe.grade)

    new_nodes = defaultdict(list)
    for node in network.nodes.values():
        if not node._report_out:
            continue
        new_nodes["node_name"].append(node.name)
        # TODO: calculate node pressure max based on pipe rating
        new_nodes["p_max_mpa_g"].append(
            min(
                [
                    update_pipes[pipe.name].pressure
                    if pipe.name in update_pipes
                    else pipe.design_pressure_MPa
                    for pipe in node.connections["Pipe"]
                ]
            )
        )
        x, y = node.plot_xy
        new_nodes["plot_x"].append(x)
        new_nodes["plot_y"].append(y)

    # Make new supply
    new_supply = defaultdict(list)
    for supply in network.supply_nodes.values():
        new_supply["supply_name"].append(supply.name)
        new_supply["node_name"].append(supply.node.name)
        new_supply["pressure_mpa_g"].append(supply.pressure_mpa)
        new_supply["flowrate_MW"].append("")
        new_supply["blend"].append(network.composition.x["H2"])
    # Make new demands
    new_demand = defaultdict(list)
    for demand in network.demand_nodes.values():
        new_demand["demand_name"].append(demand.name)
        new_demand["node_name"].append(demand.node.name)
        new_demand["flowrate_MW"].append(demand.flowrate_MW)

    # Make new composition
    new_composition = defaultdict(list)
    for species, x in network.composition.x.items():
        new_composition["SPECIES"].append(species)
        new_composition["X"].append(x)

    new_comps = {x: [] for x in mod_file_util.comps_cols()}
    new_regs = {x: [] for x in mod_file_util.regs_cols()}
    for reg in network.regulators.values():
        new_regs["regulator_name"].append(reg.name)
        new_regs["from_node"].append(reg.from_node.name)
        new_regs["to_node"].append(reg.to_node.name)
        new_regs["pressure_out_mpa_g"].append(reg.pressure_out_mpa_g)

    mod_file_util.write_to_network_file(
        filename=filename,
        new_pipes=new_pipes,
        new_nodes=new_nodes,
        new_comps=new_comps,
        new_supply=new_supply,
        new_demand=new_demand,
        new_composition=new_composition,
        new_regs=new_regs,
    )


def remedy_high_dpdx(
    network: BlendPATH_network,
    design_option: Design_options,
    design_params: Design_params,
) -> tuple[
    dict[str, "bp_mod_util.New_pipes"], dict[str, "bp_plc.Pipe"], dict[str, str]
]:
    updated_pipes = {}
    looped_parent_pipes = {}
    updated_pipe_instances = {}
    n_iter = 0

    # While network does not solve, continue increasing diameters
    network_solves = False
    while not network_solves:
        # Keep track of mod_looped pipes already updated.
        # it is possible that both the main and the loop are violating
        # then the loop diameter would increase twice
        already_updated_loops = []

        # First reduce flow to identify problematic pipes
        d_ratio = 1 / 10
        for dn in network.demand_nodes.values():
            dn.flowrate_MW *= d_ratio
        network.solve()

        # Get dpdx list
        dpdx_values = {
            p.name: abs(p.from_node.pressure - p.to_node.pressure) / p.length_km
            for p in network.pipes.values()
        }

        # Sort list to get highest dpdx
        sorted_dpdx = {
            k: v
            for k, v in sorted(
                dpdx_values.items(), key=lambda item: item[1], reverse=True
            )
        }

        # Get top 1% of pipes that violate
        pipe_filter_amt = 0.01
        upper_index = max(int(len(sorted_dpdx) * pipe_filter_amt), 1)
        top_vals = list(sorted_dpdx.keys())[:upper_index]

        for pipe_name in top_vals:
            if pipe_name in looped_parent_pipes:
                parent_pipe = network.pipes[looped_parent_pipes[pipe_name]]
                pressure_rating_MPa = parent_pipe.design_pressure_MPa
                material = parent_pipe.material
                grade = getattr(parent_pipe, "grade", None)
                polymer = getattr(parent_pipe, "polymer", None)
                polymer_designation = getattr(parent_pipe, "polymer_designation", None)
            else:
                pressure_rating_MPa = network.pipes[pipe_name].design_pressure_MPa
                material = network.pipes[pipe_name].material
                grade = getattr(network.pipes[pipe_name], "grade", None)
                polymer = getattr(network.pipes[pipe_name], "polymer", None)
                polymer_designation = getattr(
                    network.pipes[pipe_name], "polymer_designation", None
                )
            pipe_params = bp_mod_util.pipe_diam_params(
                inner_diameter_mm=network.pipes[pipe_name].diameter_mm,
                nps=network.pipes[pipe_name].NPS,
                dn=network.pipes[pipe_name].DN,
                pressure_rating_MPa=pressure_rating_MPa,
                material=material,
                grade=grade,
                polymer=polymer,
                polymer_designation=polymer_designation,
                design_option=design_option,
                design_params=design_params,
            )

            # Get the name of the looped pipe
            # If it already exists (already looped), then the same name
            if pipe_name in updated_pipe_instances:
                looped_name = pipe_name
            else:
                looped_name = f"{pipe_name}{SUFFIX}"
                looped_parent_pipes[looped_name] = pipe_name
            # If the loop has already been updated, then skip
            if looped_name in already_updated_loops:
                continue

            # Get the loop diameter. If already exists, get the next largest diameter
            # If not yet looped, add loop of same size
            if looped_name in updated_pipe_instances:
                pipe_params.inner_diameter_mm = updated_pipe_instances[
                    looped_name
                ].diameter_mm
                pipe_params.nps = updated_pipe_instances[looped_name].NPS
                pipe_params.dn = updated_pipe_instances[looped_name].DN

                diam, nps, dn, th, p_rating = bp_mod_util.get_next_diameter(
                    pipe_params_in=pipe_params, allow_existing_geom=False
                )
            else:
                diam, _, _, _, _ = bp_mod_util.get_next_diameter(
                    pipe_params_in=pipe_params,
                    allow_existing_geom=True,
                )
                dn = network.pipes[pipe_name].DN
                nps = network.pipes[pipe_name].NPS
                th = network.pipes[pipe_name].thickness_mm
                p_rating = network.pipes[pipe_name].design_pressure_MPa

            # Create new pipe instance. Cannot add the network.pipes since we
            # are currently looping through it.
            # DN is updated separately
            updated_pipe_instances[looped_name] = bp_plc.Pipe(
                name=looped_name,
                from_node=network.pipes[pipe_name].from_node,
                to_node=network.pipes[pipe_name].to_node,
                diameter_mm=diam,
                length_km=network.pipes[pipe_name].length_km,
                roughness_mm=network.pipes[pipe_name].roughness_mm,
                thickness_mm=th,
                rating_code=network.pipes[pipe_name].rating_code,
                p_max_mpa_g=network.pipes[pipe_name].p_max_mpa_g,
            )
            updated_pipe_instances[looped_name].NPS = nps
            updated_pipe_instances[looped_name].DN = dn
            updated_pipe_instances[looped_name].design_pressure_MPa = p_rating

            # Add to list of updated loops
            already_updated_loops.append(looped_name)

        # After adding all pipes, then add to network
        # Update pipe designations and connections
        for pipe_name, pipe_i in updated_pipe_instances.items():
            network.pipes[pipe_name] = pipe_i

            updated_pipes[pipe_name] = bp_mod_util.New_pipes(
                inner_diameter=pipe_i.diameter_mm,
                dn=pipe_i.DN,
                nps=pipe_i.NPS,
                th=pipe_i.thickness_mm,
                length=pipe_i.length_km,
                grade=pipe_i.rating_code,
                name=pipe_name,
                from_node=pipe_i.from_node.name,
                to_node=pipe_i.to_node.name,
                roughness_mm=pipe_i.roughness_mm,
                pressure=pipe_i.design_pressure_MPa,
            )

        update_network(network=network)

        # Reset demands to try solve again
        for dn in network.demand_nodes.values():
            dn.flowrate_MW /= d_ratio
        try:
            network.solve()
            network_solves = True
        except ValueError as _:
            network_solves = False

        # Prevent run away while loop
        n_iter += 1
        if n_iter > 20:
            raise RuntimeError("Solution not found after 10 iterations")

    return updated_pipes, updated_pipe_instances, looped_parent_pipes


def check_high_dpdx(
    network: BlendPATH_network,
    design_option: Design_options,
    design_params: Design_params,
) -> tuple[
    dict[str, "bp_mod_util.New_pipes"], dict[str, "bp_plc.Pipe"], dict[str, str]
]:

    try:
        network.solve()
        return {}, {}, {}
    except ValueError as _:
        # Currently we only treat low pressure
        # if not (e.args[0] == "Negative pressure"):
        #     raise RuntimeError("Distribution network is not solving")
        # Network doesn't solve. Need to increase diameter of affected nodes
        pass

    return remedy_high_dpdx(
        network=network,
        design_option=design_option,
        design_params=design_params,
    )


def build_pipe_params(
    pipe: "bp_plc.Pipe",
    looped_parent_pipes: dict[str, str],
    network: "BlendPATH_network",
    design_params: Design_params,
    design_option: Design_options,
    pressure_rating_MPa_override: float | None = None,
):
    if pipe.name in looped_parent_pipes:
        parent_pipe = network.pipes[looped_parent_pipes[pipe.name]]
        pressure_rating_MPa = parent_pipe.design_pressure_MPa
        material = parent_pipe.material
        grade = getattr(parent_pipe, "grade", None)
        polymer = getattr(parent_pipe, "polymer", None)
        polymer_designation = getattr(parent_pipe, "polymer_designation", None)
    else:
        pressure_rating_MPa = pipe.design_pressure_MPa
        material = pipe.material
        grade = getattr(pipe, "grade", None)
        polymer = getattr(pipe, "polymer", None)
        polymer_designation = getattr(pipe, "polymer_designation", None)
    return bp_mod_util.pipe_diam_params(
        inner_diameter_mm=pipe.diameter_mm,
        nps=pipe.NPS,
        dn=pipe.DN,
        pressure_rating_MPa=pressure_rating_MPa
        if pressure_rating_MPa_override is None
        else pressure_rating_MPa_override,
        material=material,
        grade=grade,
        polymer=polymer,
        polymer_designation=polymer_designation,
        design_option=design_option,
        design_params=design_params,
    )


def get_looped_name(
    pipe: "bp_plc.Pipe",
    updated_pipe_instances: dict[str, "bp_plc.Pipe"],
    looped_parent_pipes: dict[str, str],
) -> str:
    if pipe.name in updated_pipe_instances:
        return pipe.name
    else:
        looped_name = f"{pipe.name}{SUFFIX}"
        looped_parent_pipes[looped_name] = pipe.name
        return looped_name


def update_diameter(
    pipe: "bp_plc.Pipe",
    looped_name: str,
    updated_pipe_instances: dict[str, "bp_plc.Pipe"],
    pipe_params: "bp_mod_util.pipe_diam_params",
):
    if looped_name in updated_pipe_instances:
        pipe_params.inner_diameter_mm = updated_pipe_instances[looped_name].diameter_mm
        pipe_params.nps = updated_pipe_instances[looped_name].NPS
        pipe_params.dn = updated_pipe_instances[looped_name].DN

        return bp_mod_util.get_next_diameter(
            pipe_params_in=pipe_params, allow_existing_geom=False
        )
    else:
        diam, _, _, _, _ = bp_mod_util.get_next_diameter(
            pipe_params_in=pipe_params,
            allow_existing_geom=True,
        )
        return diam, pipe.NPS, pipe.DN, pipe.thickness_mm, pipe.design_pressure_MPa


def update_pipes(
    updated_pipe_instances: dict[str, "bp_plc.Pipe"],
    looped_name: str,
    pipe: "bp_plc.Pipe",
    pipe_params: "bp_mod_util.pipe_diam_params",
) -> None:

    # Get the loop diameter. If already exists, get the next largest diameter
    # If not yet looped, add loop of same size
    diam, nps, dn, th, p_rating = update_diameter(
        pipe=pipe,
        looped_name=looped_name,
        updated_pipe_instances=updated_pipe_instances,
        pipe_params=pipe_params,
    )

    updated_pipe_instances[looped_name] = bp_plc.Pipe(
        name=looped_name,
        from_node=pipe.from_node,
        to_node=pipe.to_node,
        diameter_mm=diam,
        length_km=pipe.length_km,
        roughness_mm=pipe.roughness_mm,
        thickness_mm=th,
        rating_code=pipe.rating_code,
        p_max_mpa_g=pipe.p_max_mpa_g,
        velocity_limit=pipe.velocity_limit,
    )
    updated_pipe_instances[looped_name].NPS = nps
    updated_pipe_instances[looped_name].DN = dn
    updated_pipe_instances[looped_name].design_pressure_MPa = p_rating


def build_new_pipes(pipe: "Pipe", p_rating):
    return bp_mod_util.New_pipes(
        inner_diameter=pipe.diameter_mm,
        dn=pipe.DN,
        nps=pipe.NPS,
        th=pipe.thickness_mm,
        length=pipe.length_km,
        grade=pipe.rating_code,
        name=pipe.name,
        from_node=pipe.from_node.name,
        to_node=pipe.to_node.name,
        roughness_mm=pipe.roughness_mm,
        pressure=p_rating,
    )
