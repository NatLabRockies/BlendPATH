from __future__ import annotations

from collections import deque

from BlendPATH.network.pipeline_components import PipeSegment, PipeSegmentDist
from BlendPATH.scenario_helper import Scenario_type

from .BlendPATH_network import BlendPATH_network
from .pipeline_components import Compressor, Node, Pipe, Regulator


def segment_pipe_transmission(network: BlendPATH_network):
    p_supply_node_name = next(iter(network.supply_nodes.values())).node.name
    pipe_segments = []
    start_nodes = [p_supply_node_name]

    visited = {n: None for n in network.nodes}

    for ps_i, s_node in enumerate(start_nodes):
        queue = deque([s_node])
        pipes = []
        diameter = None
        DN = None
        comps = []
        regs = []
        p_max_mpa_g = None
        design_pressure_MPa = None
        nodes = []
        demand_nodes = []
        offtake_lengths = [0]

        while queue:
            current_name = queue.popleft()
            current_node = network.nodes[current_name]
            nodes.append(current_node)
            if current_node.is_demand:
                demand_nodes.extend(
                    [
                        dn
                        for dn in network.demand_nodes.values()
                        if dn.node is current_node
                    ]
                )
                offtake_lengths.append(0)

            # If pipe connections, add to segment
            for pipe_conn in current_node.connections["Pipe"]:
                neighbor = (
                    pipe_conn.to_node
                    if pipe_conn.from_node == current_node
                    else pipe_conn.from_node
                )
                visited[current_name] = ps_i

                if visited[neighbor.name] is None:
                    # Check if diameter changes add to start nodes
                    if diameter is not None and diameter != pipe_conn.diameter_mm:
                        start_nodes.append(neighbor.name)
                        continue
                    queue.append(neighbor.name)

                    # Add to pipe segment values
                    pipes.append(pipe_conn)
                    diameter = pipe_conn.diameter_mm
                    DN = pipe_conn.DN
                    p_max_mpa_g = pipe_conn.p_max_mpa_g
                    design_pressure_MPa = pipe_conn.design_pressure_MPa
                    offtake_lengths[-1] += pipe_conn.length_km

            # If compressor, add to start nodes
            for comp_conn in current_node.connections["Comp"]:
                neighbor = (
                    comp_conn.to_node
                    if comp_conn.from_node == current_node
                    else comp_conn.from_node
                )
                if neighbor.name not in start_nodes and visited[neighbor.name] is None:
                    start_nodes.append(neighbor.name)

                    # Add compressor at end of segment
                    comps.append(comp_conn)

            # If compressor, add to start nodes
            for reg_conn in current_node.connections["Reg"]:
                neighbor = (
                    reg_conn.to_node
                    if reg_conn.from_node == current_node
                    else reg_conn.from_node
                )
                if neighbor.name not in start_nodes and visited[neighbor.name] is None:
                    start_nodes.append(neighbor.name)

                    # Add compressor at end of segment
                    regs.append(reg_conn)

        if offtake_lengths[0] == 0:
            offtake_lengths.pop(0)
        pipe_segments.append(
            PipeSegment(
                pipes=pipes,
                diameter=diameter,
                DN=DN,
                comps=comps,
                start_node=network.nodes[s_node],
                p_max_mpa_g=p_max_mpa_g,
                design_pressure_MPa=design_pressure_MPa,
                nodes=nodes,
                demand_nodes=demand_nodes,
                offtake_lengths=offtake_lengths,
            )
        )

    return pipe_segments


def segment_pipe_distribution(network: BlendPATH_network):
    """
    Segment distribution network into segments based on compressors, regulators, diameter change
    """

    pipe_segments = []
    start_nodes = [next(iter(network.supply_nodes.values())).node.name]
    start_nodes = (
        [sn.node.name for sn in network.supply_nodes.values()]
        + [comp.to_node.name for comp in network.compressors.values()]
        + [reg.to_node.name for reg in network.regulators.values()]
    )
    visited = {n: None for n in network.nodes}

    for ps_i, s_node in enumerate(start_nodes):
        if visited[s_node] is not None:
            continue
        queue = deque([s_node])
        pipes = []
        nodes = []

        while queue:
            # Get the current node
            current_name = queue.popleft()
            current_node = network.nodes[current_name]

            # Check current node pipe-connections
            for pipe_conn in current_node.connections["Pipe"]:
                neighbor = get_neighbor(cxn=pipe_conn, current_node=current_node)
                visited[current_name] = ps_i

                # If the neighbor is not visited yet, add to queue
                if visited[neighbor.name] is None:
                    queue.append(neighbor.name)

                    # Add to pipe segment values
                    pipes.append(pipe_conn)
                    nodes.append(current_node)

            # If compressor, add to start nodes
            for comp_conn in current_node.connections["Comp"]:
                neighbor = get_neighbor(cxn=comp_conn, current_node=current_node)
                if neighbor.name not in start_nodes and visited[neighbor.name] is None:
                    start_nodes.append(neighbor.name)

            # If regulator, add to start nodes
            for reg_conn in current_node.connections["Reg"]:
                neighbor = get_neighbor(cxn=reg_conn, current_node=current_node)
                if neighbor.name not in start_nodes and visited[neighbor.name] is None:
                    start_nodes.append(neighbor.name)

        if pipes:
            pipe_segments.append(PipeSegmentDist(pipes=list(set(pipes)), nodes=nodes))
    return pipe_segments


def get_neighbor(cxn: Pipe | Compressor | Regulator, current_node: Node) -> Node:
    return cxn.to_node if cxn.from_node == current_node else cxn.from_node


segment_pipe_hash = {
    Scenario_type.TRANSMISSION: segment_pipe_transmission,
    Scenario_type.DISTRIBUTION: segment_pipe_distribution,
}


def segment_pipe(network: BlendPATH_network):
    return segment_pipe_hash[network.scenario_type](network=network)
