from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

import BlendPATH.costing.costing as bp_cost
import BlendPATH.file_writing.mod_file_out_util as mod_file_util
import BlendPATH.Global as gl
import BlendPATH.modifications.mod_util as bp_mod_util
from BlendPATH.modifications.parallel_loop import make_loop_network
from BlendPATH.network.BlendPATH_network import BlendPATH_network
from BlendPATH.scenario_helper import Design_params

if TYPE_CHECKING:
    from BlendPATH.network.pipeline_components import Pipe


def direct_replacement_distribution(
    network: BlendPATH_network,
    design_option: str = "b",
    new_filename: str = "modified",
    design_params: Design_params = None,
    costing_params: bp_cost.Costing_params = None,
    allow_compressor_bypass: bool = False,
) -> tuple:
    original_diameters = get_original_diameters(network)

    updated_pipes = check_high_dpdx(
        network=network,
        design_option=design_option,
        design_params=design_params,
        original_diameters=original_diameters,
    )

    # Check for pressure violations
    for pipe_segment in network.pipe_segments:
        pipe_names_to_swap, seg_max_pressure_MPa = pipe_segment.violating_pipes()
        for pipe_name in pipe_names_to_swap:
            (
                network.pipes[pipe_name].diameter_mm,
                network.pipes[pipe_name].NPS,
                network.pipes[pipe_name].DN,
                network.pipes[pipe_name].thickness_mm,
                network.pipes[pipe_name].design_pressure_MPa,
            ) = bp_mod_util.get_next_diameter(
                pipe_params_in=build_pipe_params(
                    pipe=network.pipes[pipe_name],
                    design_option=design_option,
                    design_params=design_params,
                    pressure_rating_MPa_override=seg_max_pressure_MPa,
                )
            )

            updated_pipes[network.pipes[pipe_name].name] = build_new_pipes(
                pipe=network.pipes[pipe_name],
                p_rating=network.pipes[pipe_name].design_pressure_MPa,
            )

    # Update network until velocity violations are eliminated
    velocity_violtations = True
    iters = 0
    while velocity_violtations:
        # Solve network
        network.solve()
        # Reset violation flag
        velocity_violtations = False
        for pipe in network.pipes.values():
            # Check if any violations exist
            if bp_mod_util.check_velocity_violation(
                pipe, design_params.max_dist_velocity
            ):
                continue

            # Get and assign next larger diameter for pipes that need updating
            (
                pipe.diameter_mm,
                pipe.NPS,
                pipe.DN,
                pipe.thickness_mm,
                pipe.design_pressure_MPa,
            ) = bp_mod_util.get_next_diameter(
                pipe_params_in=build_pipe_params(
                    pipe=pipe,
                    design_option=design_option,
                    design_params=design_params,
                )
            )

            # Save in list of updated pipes
            updated_pipes[pipe.name] = build_new_pipes(
                pipe=pipe, p_rating=pipe.design_pressure_MPa
            )

            # Flag violation to trigger network solve
            velocity_violtations = True

        # Limit iterations in while loop
        iters += 1
        if iters > 20:
            raise RuntimeError("Solution was not found in 20 iterations")

    # reset_network to original values
    for pipe in network.pipes.values():
        (
            pipe.diameter_mm,
            pipe.NPS,
            pipe.DN,
            pipe.thickness_mm,
            pipe.design_pressure_MPa,
        ) = original_diameters[pipe.name]

    # Organize modified pipes by length
    combined_pipe = bp_mod_util.get_polymer_combined_pipe_stats(
        updated_pipes=updated_pipes, costing_params=costing_params
    )

    make_result_file(filename=new_filename, network=network, update_pipes=updated_pipes)

    return updated_pipes, combined_pipe, []


def direct_replacement_distribution_alt(
    network: BlendPATH_network,
    design_option: str = "b",
    new_filename: str = "modified",
    design_params: Design_params = None,
    costing_params: bp_cost.Costing_params = None,
) -> tuple:
    updated_pipes = {}

    network.solve()

    # Loop over all pipes to check violations, and modify if needed
    for pipe in network.pipes.values():
        # Skip over pipe if no violations
        if bp_mod_util.check_velocity_violation(pipe, design_params.max_dist_velocity):
            continue

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

        # Create mini network
        p_max = max(pipe.to_node.pressure, pipe.from_node.pressure)
        loop_network, _, _, _ = make_loop_network(
            l_loop=0,
            composition=network.composition,
            p_in=p_max / gl.MPA2PA,
            offtakes=[pipe.length_km],
            all_mdot=[abs(pipe.m_dot)],
            d_main=pipe.diameter_mm,
            d_loop=pipe.diameter_mm,
            th=pipe.thickness_mm,
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
        loop_network.pipes["main_0"].NPS = pipe.NPS
        loop_network.pipes["main_0"].DN = pipe.DN
        loop_network.pipes["main_0"].thickness_mm = pipe.thickness_mm
        # Dont need to update pressure rating
        loop_network.scenario_type = network.scenario_type

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
            pipe_params.inner_diameter_mm = loop_network.pipes["main_0"].diameter_mm
            pipe_params.nps = loop_network.pipes["main_0"].NPS
            pipe_params.dn = loop_network.pipes["main_0"].DN
            diam, nps, dn, th, p_rating = bp_mod_util.get_next_diameter(
                pipe_params_in=pipe_params
            )
            # Increase diameter of loop
            for pipe_name in loop_network.pipes.values():
                pipe_name.diameter_mm = diam
                pipe_name.NPS = nps
                pipe_name.DN = dn
                pipe_name.thickness_mm = th

            # Limit iterations in while loop
            iters += 1
            if iters > 20:
                break

        updated_pipes[f"{pipe.name}"] = bp_mod_util.New_pipes(
            inner_diameter=diam,
            dn=dn,
            nps=nps,
            th=th,
            length=pipe.length_km,
            grade=pipe.rating_code,
            name=f"{pipe.name}",
            from_node=pipe.from_node.name,
            to_node=pipe.to_node.name,
            roughness_mm=pipe.roughness_mm,
            pressure=p_rating,
        )

    # Organize modified pipes by length
    combined_pipe = bp_mod_util.get_polymer_combined_pipe_stats(
        updated_pipes=updated_pipes, costing_params=costing_params
    )

    make_result_file(filename=new_filename, network=network, update_pipes=updated_pipes)

    return updated_pipes, combined_pipe, []


def get_original_diameters(network):
    return {
        p.name: (p.diameter_mm, p.NPS, p.DN, p.thickness_mm, p.design_pressure_MPa)
        for p in network.pipes.values()
    }


def make_result_file(filename: str, network: BlendPATH_network, update_pipes: dict):
    new_pipes = defaultdict(list)
    for pipe in network.parent_pipes.values():
        # These values don't change from the DR method
        new_pipes["pipe_name"].append(pipe.name)
        new_pipes["from_node"].append(pipe.from_node.name)
        new_pipes["to_node"].append(pipe.to_node.name)
        new_pipes["length_km"].append(pipe.length_km)
        new_pipes["roughness_mm"].append(pipe.roughness_mm)
        # These values may update
        inner_diam = pipe.diameter_mm
        rating_code = pipe.rating_code
        thickness = pipe.thickness_mm
        if pipe.name in update_pipes:
            inner_diam = update_pipes[pipe.name].inner_diameter
            rating_code = update_pipes[pipe.name].grade
            thickness = update_pipes[pipe.name].th
        new_pipes["diameter_mm"].append(inner_diam)
        new_pipes["rating_code"].append(rating_code)
        new_pipes["thickness_mm"].append(thickness)

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
    design_option: str,
    design_params: Design_params,
    original_diameters: dict[str, tuple[float, float, float, float, float]],
) -> dict[str, "bp_mod_util.New_pipes"]:
    """Increase diameter of pipes with high pressure drop in order to get a convergent simulation

    Args:
        network (BlendPATH_network): Network
        design_option (str): Design option
        design_params (Design_params): Design parameters

    Raises:
        RuntimeError: Too many iterations

    Returns:
        dict[str, bp_mod_util.New_pipes]: List of updated pipes
    """

    updated_pipes = {}
    n_iter = 0

    # While network does not solve, continue increasing diameters
    network_solves = False
    while not network_solves:
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

        # Increase diameters of top dpdx pipes and log
        for pipe_name in top_vals:
            pipe = network.pipes[pipe_name]
            _, _, _, _, og_pipe_rating = original_diameters[pipe_name]
            pipe_params = bp_mod_util.pipe_diam_params(
                inner_diameter_mm=pipe.diameter_mm,
                nps=pipe.NPS,
                dn=pipe.DN,
                pressure_rating_MPa=og_pipe_rating,
                material=pipe.material,
                grade=getattr(pipe, "grade", None),
                polymer=getattr(pipe, "polymer", None),
                polymer_designation=getattr(pipe, "polymer_designation", None),
                design_option=design_option,
                design_params=design_params,
            )

            # Get and assign next larger diameter
            (
                pipe.diameter_mm,
                pipe.NPS,
                pipe.DN,
                pipe.thickness_mm,
                pipe.design_pressure_MPa,
            ) = bp_mod_util.get_next_diameter(pipe_params_in=pipe_params)

            # Save in list of updated pipes
            updated_pipes[pipe.name] = bp_mod_util.New_pipes(
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
                pressure=pipe.design_pressure_MPa,
            )

        # Reset demands to try solve again
        for dn in network.demand_nodes.values():
            dn.flowrate_MW /= d_ratio
        try:
            network.solve()
            network_solves = True
        except ValueError as _:
            network_solves = False

        # Increase iterations. Stop after 10
        n_iter += 1
        if n_iter > 20:
            raise RuntimeError("Too many iterations")
    return updated_pipes


def check_high_dpdx(
    network: BlendPATH_network,
    design_option: str,
    design_params: Design_params,
    original_diameters: dict[str, tuple[float, float, float, float, float]],
) -> dict[str, "bp_mod_util.New_pipes"]:

    try:
        network.solve()
        return {}
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
        original_diameters=original_diameters,
    )


def build_pipe_params(
    pipe: "Pipe",
    design_params: Design_params,
    design_option: str,
    pressure_rating_MPa_override: float | None = None,
):
    return bp_mod_util.pipe_diam_params(
        inner_diameter_mm=pipe.diameter_mm,
        nps=pipe.NPS,
        dn=pipe.DN,
        pressure_rating_MPa=pipe.design_pressure_MPa
        if pressure_rating_MPa_override is None
        else pressure_rating_MPa_override,
        material=pipe.material,
        grade=getattr(pipe, "grade", None),
        polymer=getattr(pipe, "polymer", None),
        polymer_designation=getattr(pipe, "polymer_designation", None),
        design_option=design_option,
        design_params=design_params,
    )


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
