"""
Modification method to represent adding a new H2 pipeline instead of blending into an existing pipeline
"""

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

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
from BlendPATH.util.schedules import SCHEDULES_STEEL_DN, SCHEDULES_STEEL_SCH

logger = logging.getLogger(__name__)

MAX_COMPRESSORS = 5


@dataclass
class new_design_params:
    """Dataclass to hold constants for setting up a new network"""

    new_comps_elec: bool
    assign_eta_s: float
    assign_eta_driver: float
    pipe_roughness_mm: float
    eos: str
    ff_type: str
    composition: bp_plc.Composition
    demands_original: dict
    offtake_distance: list
    supply_pressure_MPa: float


@dataclass
class h2_pipe_solution:
    """Dataclass to hold entries for a solution found"""

    network: BlendPATH_network = None
    lcot: float = np.inf
    dn: float | None = None
    sch: str | None = None
    material_cost: float | None = None
    grade: str | None = None
    l_compressors: list = field(default_factory=list)


def new_h2_pipeline(
    network: BlendPATH_network,
    design_option: str,
    new_filename: str,
    design_params: Design_params,
    costing_params: bp_cost.Costing_params,
    allow_compressor_bypass: bool = False,
) -> tuple[dict[str, "bp_mod_util.New_pipes"], dict, list, list]:
    """Modification method to create new H2 pipeline

    Args:
        network (BlendPATH_network): Original network
        design_option (str, optional): Design option for new pipes
        new_filename (str, optional): Filename for new network file
        design_params (Design_params, optional): Scenario design parameters
        costing_params (bp_cost.Costing_params, optional): Scenario costing parameters

    Returns:
        tuple[dict[str, bp_mod_util.New_pipes], dict, list, list]: new_pipes, combined pipes, n_compressors, l_compressors
    """

    # Find furthest demand node
    offtake_distance, total_length_km = find_furthest_demand(network=network)

    # Miniumu outlet pressure of network
    outlet_pressure_MPa = design_params.final_outlet_pressure_mpa_g
    # Maximum compression ratio considered
    max_CR = max(design_params.max_CR)

    # Efficiencies of new compressors
    assign_eta_s, assign_eta_driver = bp_mod_util.compressor_eta(
        design_params=design_params
    )
    # Constants for setting up a new network
    h2_design_params = new_design_params(
        assign_eta_s=assign_eta_s,
        assign_eta_driver=assign_eta_driver,
        new_comps_elec=design_params.new_comp_elec,
        pipe_roughness_mm=0.012,  # TODO: Pipe roughness as input
        eos=network.eos,
        ff_type=network.ff_type,
        composition=bp_plc.Composition(
            composition_tracking=False, thermo_curvefit=True, eos_type="rk", blend=1.0
        ),
        demands_original=get_stacked_demand_mdot(
            network=network, blend_ratio_energy=network.blend_ratio_energy
        ),
        offtake_distance=offtake_distance,
        supply_pressure_MPa=get_supply_pressure(network=network),
    )

    # Get pipe DN greater than or equal to the smallest DN in the original network
    DN_vals = get_DN_values(
        network=network,
        design_params=design_params,
        total_length_km=total_length_km,
        m_dot=sum(h2_design_params.demands_original.values()),
    )
    # Get list of steel grades available
    all_grades = bp_pa.get_pipe_grades()
    # Initialize solution
    solution = h2_pipe_solution()

    # Loop through DN
    dn_stop = False
    for od, dn in DN_vals.items():
        if dn_stop:
            break
        # Only need one solution per DN. Increasing grade has no beneficial effect without a supply compressor.
        solution_found = False
        # Loop through grades
        for grade in all_grades:
            # We do not need to check better grade steels once a solution is found
            if solution_found:
                break
            # Get viable thicknesses and schedules
            th, schedule = get_thickness(
                design_option=design_option,
                ASME_params=design_params.asme,
                grade=grade,
                supply_pressure_MPa=h2_design_params.supply_pressure_MPa,
                dn=dn,
            )
            # Skip if no valid thicknesses
            if np.isnan(th):
                continue

            # Calculate ID from OD and TH
            diam_inner_mm = od - 2 * th

            # Initialize increasing compressors until valid solution found
            n_compressors = 0
            # While no valid solution, increase compressors. Set max of 10
            while not solution_found and n_compressors < MAX_COMPRESSORS:
                # Uniformly split compressors along the total length
                compressor_lengths = [
                    total_length_km / (n_compressors + 1) * (x + 1)
                    for x in range(n_compressors)
                ]

                # Generate a new network based on chosen DN and grade
                h2_pipeline_network, final_node = make_new_network(
                    diam_inner_mm=diam_inner_mm,
                    thickness_mm=th,
                    compressor_lengths=compressor_lengths,
                    h2_design_params=h2_design_params,
                    grade=grade,
                )
                # Attempty to solve network. Raised value errors will retry with +1 compressors
                try:
                    # Solve hydraulics
                    h2_pipeline_network.solve()

                    # Valid solution checks. Fails if any return true
                    check_solution = [
                        lambda: check_CR(network=h2_pipeline_network, max_CR=max_CR),
                        lambda: check_outlet_pressure(
                            final_node=final_node,
                            outlet_pressure_MPa=outlet_pressure_MPa,
                        ),
                    ]
                    if any(check() for check in check_solution):
                        logger.info(
                            f"Solution found for dn={dn}, grade={grade}, th={th}, # Compressors = {n_compressors}; but failed validity checks {[check() for check in check_solution]}"
                        )
                        raise ValueError("Solution does not pass")

                    # Update solution found for this DN
                    solution_found = True

                    # Calculate material cost
                    mat_cost = bp_pipe_cost.get_pipe_material_cost(
                        cp=costing_params,
                        di_mm=diam_inner_mm,
                        do_mm=od,
                        l_km=total_length_km,
                        grade=grade,
                    )
                    # Calculate other possible pipe costs (e.g., labor, ROW, misc)
                    other_pipe_cap = bp_pipe_cost.get_pipe_other_cost(
                        cp=costing_params,
                        d_mm=dn,
                        l_km=total_length_km,
                        anl_types=bp_mod_util.get_pipe_cost_types(
                            mod_type="newh2", scenario_type=network.scenario_type
                        ),
                    )

                    # Calculate LCOT for this combination
                    price_breakdown, _ = bp_cost.calc_lcot(
                        mod_costing_params=bp_mod_util.get_mod_costing_params(
                            network=h2_pipeline_network,
                            design_params=design_params,
                            costing_params=costing_params,
                        ),
                        new_pipe_capex=sum(other_pipe_cap.values()) + mat_cost,
                        costing_params=costing_params,
                    )

                    logger.info(
                        f"Solution found for dn={dn}, grade={grade}, th={th}, # Compressors = {n_compressors}, LCOT = {price_breakdown['LCOT: Levelized cost of transport']}"
                    )

                    # Inflection points do not always indicate the global min. Keep commented out
                    # if (
                    #     price_breakdown["LCOT: Levelized cost of transport"]
                    #     > solution.lcot
                    # ):
                    #     logger.info(
                    #         "Inflection point found - stopping iterations through DN"
                    #     )
                    #     dn_stop = True
                    #     break

                    if (
                        price_breakdown["LCOT: Levelized cost of transport"]
                        < solution.lcot
                    ):
                        solution = h2_pipe_solution(
                            network=h2_pipeline_network,
                            lcot=price_breakdown["LCOT: Levelized cost of transport"],
                            dn=dn,
                            sch=schedule,
                            grade=grade,
                            material_cost=mat_cost,
                            l_compressors=compressor_lengths,
                        )

                # Increase compressors if failing
                except ValueError as _:
                    n_compressors += 1

    if np.isnan(solution.lcot):
        raise RuntimeError("No solution found")

    # Make network file
    make_network_file(filename=new_filename, network=solution.network)

    # Format combined pipe
    combined_pipe = {
        "DN": [solution.dn],
        "sch": [solution.sch],
        "grade": [solution.grade],
        "length": [total_length_km],
        "mat_cost": [solution.material_cost],
        "other_pipe_cost": [],
        "D_S_G": [f"{solution.dn};;{solution.sch};;{solution.grade}"],
    }

    # Format new pipes
    new_pipes = {}
    for pipe in solution.network.parent_pipes.values():
        new_pipes[pipe.name] = bp_mod_util.New_pipes(
            grade=pipe.grade,
            cost=solution.material_cost,
            th=pipe.thickness_mm,
            schedule=pipe.schedule,
            pressure=pipe.p_max_mpa_g,
            inner_diameter=pipe.diameter_mm,
            dn=pipe.DN,
            length=pipe.length_km,
            ps=0,
            name=pipe.name,
        )

    # Return values
    # TODO: improve formatting
    return (
        new_pipes,
        combined_pipe,
        [solution.l_compressors],
    )


def find_furthest_demand(network: BlendPATH_network) -> tuple[dict[float, str], float]:
    """Find cumulative lengths of offtakes in network. Currently only works for linear networks

    Args:
        network (BlendPATH_network): Original network to analyze

    Returns:
        tuple[dict[float, str], float]: Tuple of offtake distances with keys as the cumulative offtake lengths and values as the node name. The second value of tuple is the max length
    """
    # Choose a supply node to start at
    p_supply_node_name = next(iter(network.supply_nodes.values())).node.name
    # Keep track of nodes visited
    visited = {n: None for n in network.nodes}
    visited[p_supply_node_name] = 0
    # Initialize distances dict and total length counter
    distances = {}
    net_length_km = 0
    # Initialize deque for traversing network
    queue = deque([p_supply_node_name])

    # Continue traversing while there are nodes in the deque
    while queue:
        # Get the current node
        current_name = queue.popleft()
        current_node = network.nodes[current_name]
        # If current node is a demand node, then save its distance
        if current_node.is_demand:
            distances[net_length_km] = current_name

        # Check the pipes that are connected to current node
        for pipe_conn in current_node.connections["Pipe"]:
            neighbor = (
                pipe_conn.to_node.name
                if pipe_conn.from_node.name == current_name
                else pipe_conn.from_node.name
            )
            if visited[neighbor] is None:
                queue.append(neighbor)
                net_length_km += pipe_conn.length_km
                visited[neighbor] = net_length_km

        # Check the compressors that are connected to current node
        for comp_conn in current_node.connections["Comp"]:
            neighbor = (
                comp_conn.to_node.name
                if comp_conn.from_node.name == current_name
                else comp_conn.from_node.name
            )
            if visited[neighbor] is None:
                queue.append(neighbor)
                net_length_km += 0
                visited[neighbor] = net_length_km
    return distances, net_length_km


def get_supply_pressure(network: BlendPATH_network) -> float:
    """Get maximum supply pressure in network

    Args:
        network (BlendPATH_network): Original network

    Returns:
        float: Max supply pressure
    """
    return max([sn.pressure_mpa for sn in network.supply_nodes.values()])


def get_thickness(
    design_option: str,
    ASME_params: bp_pa.ASME_consts,
    grade: str,
    supply_pressure_MPa: float,
    dn: float,
) -> tuple[float, str]:
    """Get thickness and schedule based on pressure rating

    Args:
        design_option (str): Design option for new pipe
        ASME_params (bp_pa.ASME_consts): ASME rating constants
        grade (str): Steel grade
        supply_pressure_MPa (float): Pressure to rate to
        dn (float): DN of pipe

    Returns:
        tuple[float, str]: Thickness [mm], Schedule
    """
    (th, schedule, pressure, index) = bp_pa.get_viable_schedules(
        sch_list=SCHEDULES_STEEL_SCH[dn],
        design_option=design_option,
        ASME_params=ASME_params,
        grade=grade,
        p_max_mpa_g=supply_pressure_MPa,
        design_pressure_MPa=supply_pressure_MPa,
        DN=dn,
    )
    if index == -1:
        return np.nan, ""
    return th[index], schedule[index]


@dataclass
class IndexTracking:
    node: int = 1  # Since supply node was already added
    pipe: int = 0
    comp: int = 0
    dmnd: int = 0


def make_new_network(
    diam_inner_mm: float,
    thickness_mm: float,
    compressor_lengths: list,
    h2_design_params: new_design_params,
    grade: str,
) -> tuple[BlendPATH_network, bp_plc.Node]:
    """Generate new H2 pipeline network

    Args:
        diam_inner_mm (float): Pipe inner diameter [mm]
        thickness_mm (float): Pipe thickness [mm]
        compressor_lengths (list): List of cumulative compressor distances from supply node
        h2_design_params (new_design_params): Static H2 pipeline design parameters
        grade (str): Steel grade

    Returns:
        tuple[BlendPATH_network, bp_plc.Node]: New H2 pipeline network, and the last node in the network
    """
    # Setup dicts for network creation. Prepopulate with supply node
    nodes = {
        "supply_node": bp_plc.Node(
            name="supply_node",
            composition=h2_design_params.composition,
            p_max_mpa_g=h2_design_params.supply_pressure_MPa,
        )
    }
    supplys = {
        "supply_node": bp_plc.Supply_node(
            node=nodes["supply_node"],
            pressure_mpa=h2_design_params.supply_pressure_MPa,
            blend=1.0,
            name="supply_node",
        )
    }
    pipes, compressors, demands = {}, {}, {}

    # Combine lengths to get order of compressors and offtakes
    all_lengths = bp_mod_util.get_sorted_lengths(
        d_main=[],
        l_loop=0,
        all_mdot=list(h2_design_params.demands_original.values()),
        offtakes=list(h2_design_params.offtake_distance.keys()),
        l_comps=compressor_lengths,
        hhv=1,
        cumulative_offtakes=False,
    )

    # Set up indices for nodes, pipes, compressors.
    indexes = IndexTracking()
    # Tracking length added, for subtracting with cumulative lengths
    prev_length = 0
    # Previously added node
    prev_node = nodes["supply_node"]

    # Loop through the offtakes and compressors
    for val in all_lengths:
        length = val.length_km
        if val.val_type == "offtake":
            # Add node and demand node
            pipe_to_node = add_node(
                nodes=nodes,
                h2_design_params=h2_design_params,
                node_prefix="ot",
                node_index=indexes.node,
            )
            add_offtake(
                demands=demands,
                node_index=indexes.dmnd,
                node=pipe_to_node,
                flowrate_MW=val.mw,
            )
            indexes.node += 1
            indexes.dmnd += 1

            # Update nodes
            next_node = pipe_to_node

        elif val.val_type == "comp":
            # Make compressor from node
            pipe_to_node = add_node(
                nodes,
                h2_design_params,
                "c",
                indexes.node,
            )
            indexes.node += 1

            # Make compressor to node
            comp_to_node = add_node(
                nodes,
                h2_design_params,
                "c",
                indexes.node,
            )
            indexes.node += 1

            # Make compressor
            add_compressor(
                compressors,
                pipe_to_node,
                comp_to_node,
                indexes.comp,
                h2_design_params.supply_pressure_MPa,
                h2_design_params,
            )
            indexes.comp += 1

            next_node = comp_to_node

        # Add connecting pipe
        add_pipe(
            pipes=pipes,
            pipe_index=indexes.pipe,
            from_node=prev_node,
            to_node=pipe_to_node,
            length_km=length - prev_length,
            diameter_mm=diam_inner_mm,
            thickness_mm=thickness_mm,
            roughness_mm=h2_design_params.pipe_roughness_mm,
            grade=grade,
        )
        # update lengths and node
        prev_length = length
        indexes.pipe += 1
        prev_node = next_node

    return BlendPATH_network(
        name="new_network",
        pipes=pipes,
        nodes=nodes,
        demand_nodes=demands,
        supply_nodes=supplys,
        compressors=compressors,
        composition=h2_design_params.composition,
        thermo_curvefit=True,
        eos=h2_design_params.eos,
        ff_type=h2_design_params.ff_type,
    ), prev_node


def add_node(
    nodes: dict[str, bp_plc.Node],
    h2_design_params: new_design_params,
    node_prefix: str,
    node_index: int,
) -> bp_plc.Node:
    """Add new node to dict of nodes for network creation

    Args:
        nodes (dict[str, bp_plc.Node]): Dict of nodes
        h2_design_params (new_design_params): Static design parameters
        node_prefix (str): Prefix for node name
        node_index (int): Node incrementer

    Returns:
        bp_plc.Node: Added node
    """
    name = f"{node_prefix}_{node_index}"
    nodes[name] = bp_plc.Node(
        name=name,
        composition=h2_design_params.composition,
        p_max_mpa_g=h2_design_params.supply_pressure_MPa,
    )
    return nodes[name]


def add_offtake(
    demands: dict[str, bp_plc.Demand_node],
    node_index: int,
    node: bp_plc.Node,
    flowrate_MW: float,
) -> None:
    """Add demand node to dict

    Args:
        demands (dict[str, bp_plc.Demand_node]): Dict of demand nodes
        node_index (int): Incrementer for name. Same as node incrementer
        node (bp_plc.Node): Node that the demand node is at
        flowrate_MW (float): Energy demand
    """
    d_name = f"demand_{node_index}"
    demands[d_name] = bp_plc.Demand_node(
        name=d_name,
        node=node,
        flowrate_MW=flowrate_MW,
    )


def add_compressor(
    compressors: dict[str, bp_plc.Compressor],
    from_node: bp_plc.Node,
    to_node: bp_plc.Node,
    comp_index: int,
    pressure_out_mpa_g: float,
    h2_design_params: new_design_params,
) -> None:
    """Add compressor to dict

    Args:
        compressors (dict[str, bp_plc.Compressor]): Dict of compressors
        from_node (bp_plc.Node): Compressor low pressure node
        to_node (bp_plc.Node): Compressor high pressure node
        comp_index (int): Name incrementer
        pressure_out_mpa_g (float): Outlet pressure of compressor
        h2_design_params (new_design_params): Static design parameters
    """
    comp_name = f"c_{comp_index}"
    compressors[comp_name] = bp_plc.Compressor(
        name=comp_name,
        from_node=from_node,
        to_node=to_node,
        pressure_out_mpa_g=pressure_out_mpa_g,
        original_rating_MW=0,
        fuel_extract=not h2_design_params.new_comps_elec,
    )
    setattr(
        compressors[comp_name],
        f"eta_comp_s{'' if not h2_design_params.new_comps_elec else '_elec'}",
        h2_design_params.assign_eta_s,
    )
    setattr(
        compressors[comp_name],
        f"eta_driver{'' if not h2_design_params.new_comps_elec else '_elec'}",
        h2_design_params.assign_eta_driver,
    )


def add_pipe(
    pipes: dict[str, bp_plc.Pipe],
    pipe_index: int,
    from_node: bp_plc.Node,
    to_node: bp_plc.Node,
    length_km: float,
    diameter_mm: float,
    thickness_mm: float,
    roughness_mm: float,
    grade: str,
) -> None:
    """Add pipe to pipes dict

    Args:
        pipes (dict[str, bp_plc.Pipe]): Dict of pipes
        pipe_index (int): Pipe name incrementer
        from_node (bp_plc.Node): From node
        to_node (bp_plc.Node): To node
        length_km (float): Pipe length [km]
        diameter_mm (float): Pipe inner diameter [mm]
        thickness_mm (float): Pipe thickness [mm]
        roughness_mm (float): Pipe internal roughness [mm]
        grade (str): Steel grade
    """
    p_name = f"pipe_{pipe_index}"
    pipes[p_name] = bp_plc.Pipe(
        name=p_name,
        from_node=from_node,
        to_node=to_node,
        diameter_mm=diameter_mm,
        thickness_mm=thickness_mm,
        length_km=length_km,
        roughness_mm=roughness_mm,
        rating_code=grade,
    )


def get_stacked_demand_mdot(
    network: BlendPATH_network, blend_ratio_energy: float
) -> dict[str, float]:
    """Get the total MW demand at each offtake

    Args:
        network (BlendPATH_network): Original network with demands to investigate

    Returns:
        dict[str, float]: Keys are node names, values are sum of flowrate_MW
    """
    demand_mdot = defaultdict(float)
    for dn in network.demand_nodes.values():
        demand_mdot[dn.node.name] += dn.flowrate_MW * blend_ratio_energy
    return demand_mdot


def get_DN_values(
    network: BlendPATH_network,
    design_params: Design_params,
    total_length_km: float,
    m_dot=float,
) -> dict[float, float]:
    """Get DN values greater than or equal to a estimated diameter

    Args:
        network (BlendPATH_network): Original network

    Returns:
        dict[float, float]: Keys are outer diameter [mm], values are DN
    """

    pipe = next(iter(network.pipes.values()))
    p_in = next(iter(network.supply_nodes.values())).pressure_mpa * gl.MPA2PA
    p_out = design_params.final_outlet_pressure_mpa_g * gl.MPA2PA
    L = total_length_km / (MAX_COMPRESSORS + 1)
    p_eqn = (p_in**2 - p_out**2) ** 0.5
    p_mid = (p_in + p_out) / 2
    mw = network.composition.get_mw(p_gauge_pa=p_mid, x=1.0)
    _, z = network.composition.get_rho_z(p_gauge_pa=p_mid, x=1.0, mw=mw)
    zrt = z * ct.gas_constant * gl.T_FIXED
    d = ((m_dot * 4 / p_eqn / math.pi) ** 2 * zrt * pipe.f * L / mw) ** (
        1 / 5
    ) / gl.MM2M

    # min_dn = 0  # min([pipe.DN for pipe in network.pipes.values()])
    # divide by 2 for extra margin
    return {od: dn for od, dn in SCHEDULES_STEEL_DN.items() if dn >= d / 2}


def check_CR(network: BlendPATH_network, max_CR: float) -> bool:
    """Function for validity check that no compressor is over the max compression ratio

    Args:
        network (BlendPATH_network): Network with compressors to evaluate
        max_CR (float): Max compression ratio allowed

    Returns:
        bool: True if any compressions with too high of a CR
    """
    return any([cs.compression_ratio > max_CR for cs in network.compressors.values()])


def check_outlet_pressure(final_node: bp_plc.Node, outlet_pressure_MPa: float) -> bool:
    """Function for validity check that minimum outlet pressure is not violated

    Args:
        final_node (bp_plc.Node): Last node in the network to check pressure with
        outlet_pressure_MPa (float): Minimum outlet pressure [MPa]

    Returns:
        bool: True if final node pressure < outlet pressure lower limit
    """
    return final_node.pressure / gl.MPA2PA < outlet_pressure_MPa


def make_network_file(filename: str, network: BlendPATH_network) -> None:
    """Generate network file for modified network

    Args:
        filename (str): Filename to save to
        network (BlendPATH_network): New network
    """
    # Make new nodes
    new_nodes = {x: [] for x in mod_file_util.nodes_cols()}
    for node in network.nodes.values():
        if node._report_out:
            new_nodes["node_name"].append(node.name)
            new_nodes["p_max_mpa_g"].append(node.p_max_mpa_g)

    # Make new pipes for network file
    new_pipes = {x: [] for x in mod_file_util.pipes_cols()}
    for pipe in network.parent_pipes.values():
        new_pipes["pipe_name"].append(pipe.name)
        new_pipes["from_node"].append(pipe.from_node.name)
        new_pipes["to_node"].append(pipe.to_node.name)
        new_pipes["length_km"].append(pipe.length_km)
        new_pipes["roughness_mm"].append(pipe.roughness_mm)
        new_pipes["diameter_mm"].append(pipe.diameter_mm)
        new_pipes["thickness_mm"].append(pipe.thickness_mm)
        new_pipes["rating_code"].append(pipe.grade)

    # Supply nodes
    new_supply = {x: [] for x in mod_file_util.supply_cols()}
    for supply in network.supply_nodes.values():
        new_supply["supply_name"].append(supply.name)
        new_supply["node_name"].append(supply.node.name)
        new_supply["pressure_mpa_g"].append(supply.pressure_mpa)
        new_supply["flowrate_MW"].append("")
        new_supply["blend"].append(1.0)

    # Make new demands
    new_demand = {x: [] for x in mod_file_util.demand_cols()}
    for demand in network.demand_nodes.values():
        new_demand["demand_name"].append(demand.name)
        new_demand["node_name"].append(demand.node.name)
        new_demand["flowrate_MW"].append(demand.flowrate_MW)

    # Make composition for network file
    new_composition = {x: [] for x in mod_file_util.composition_cols()}
    for species, x in network.composition.x.items():
        new_composition["SPECIES"].append(species)
        new_composition["X"].append(x)

    # Compressors
    new_comps = {x: [] for x in mod_file_util.comps_cols()}
    for comp in network.compressors.values():
        new_comps["compressor_name"].append(comp.name)
        new_comps["from_node"].append(comp.from_node.name)
        new_comps["to_node"].append(comp.to_node.name)
        new_comps["rating_MW"].append(comp.original_rating_MW)
        new_comps["extract_fuel"].append(comp.fuel_extract)
        new_comps["eta_s"].append(
            comp.eta_comp_s if comp.fuel_extract else comp.eta_comp_s_elec
        )
        new_comps["eta_driver"].append(comp.eta_driver if comp.fuel_extract else "")
        new_comps["pressure_out_mpa_g"].append(comp.pressure_out_mpa_g)

    mod_file_util.write_to_network_file(
        filename=filename,
        new_pipes=new_pipes,
        new_nodes=new_nodes,
        new_comps=new_comps,
        new_supply=new_supply,
        new_demand=new_demand,
        new_composition=new_composition,
    )
