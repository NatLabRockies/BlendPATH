import logging
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Union

import numpy as np
import scipy.sparse as sp

import BlendPATH.Global as gl
import BlendPATH.network.pipeline_components.cantera_util as ctu
from BlendPATH.network.pipeline_components import friction_factor

if TYPE_CHECKING:
    from BlendPATH import BlendPATH_network

logger = logging.getLogger(__name__)


def make_connections(network: "BlendPATH_network", shape_pn: tuple[int, int]):
    rows = []
    cols = []
    pipe_index = []
    coef = []
    nodal_rows = []
    nodal_data = [1, -1] * shape_pn[1]
    nodal_cols = [i for i in range(shape_pn[1]) for _ in range(2)]

    for p_i, pipe in enumerate(network.pipes.values()):
        to_adj = pipe.to_node.index_adj
        from_adj = pipe.from_node.index_adj

        nodal_rows.extend([pipe.to_node.index, pipe.from_node.index])

        if (to_adj is None) and (from_adj is None):
            continue

        # Off-diagonal
        if (to_adj is not None) and (from_adj is not None):
            rows.extend([to_adj, from_adj])
            cols.extend([from_adj, to_adj])
            pipe_index.extend([p_i, p_i])
            coef.extend([1, 1])

        # Diagonal
        if to_adj is not None:
            rows.append(to_adj)
            cols.append(to_adj)
            pipe_index.append(p_i)
            coef.append(-1)
        if from_adj is not None:
            rows.append(from_adj)
            cols.append(from_adj)
            pipe_index.append(p_i)
            coef.append(-1)
    logger.debug(f"Connection matrix rows: {rows}")
    logger.debug(f"Connection matrix rows: {cols}")
    logger.debug(f"Connection matrix pipe_index: {pipe_index}")
    logger.debug(f"Connection matrix coef: {coef}")

    return (
        rows,
        cols,
        pipe_index,
        np.asarray(coef),
        sp.csr_matrix((nodal_data, (nodal_rows, nodal_cols)), shape=shape_pn),
    )


def get_flow_targets(n_adj: float, network: "BlendPATH_network") -> tuple:
    
    for dn in network.demand_nodes.values():
        dn.recalc_mdot()
    
    m_dot_target = np.zeros(n_adj)

    # Get names of demand nodes
    dn_nodes = [
        node.node.index
        for node in network.demand_nodes.values()
        if node.node.index not in network.ignore_nodes
    ]
    # Add flow rate [kg/s] to m_dot_target
    for node in network.demand_nodes.values():
        if node.node.index in network.ignore_nodes:
            continue
        m_dot_target[node.node.index_adj] += node.flowrate_mdot
    # For supply nodes, add the m_dot_target if it is a flow rate supply (not pressure)
    for node in network.supply_nodes.values():
        if not node.is_pressure_supply:
            m_dot_target[node.node.index_adj] -= node.flowrate_mdot
            dn_nodes.append(node.node.index)

    # Get the adjusted node indexes (without ignore nodes)
    dn_node_adj = [network.get_n_adj(i) for i in dn_nodes]
    # If compressor or regulator at the end of segment. Add the from node, instead of to node
    # That way the flow rate can still be targetted
    for comp in network.compressors.values():
        # If there is a compressor at the end of the segment
        if not comp.to_node.connections["Pipe"]:
            dn_nodes.append(comp.from_node.index)
            dn_node_adj.append(comp.from_node.index_adj)
    for reg in network.regulators.values():
        # If there is a regulator at the end of the segment
        if not reg.to_node.connections["Pipe"]:
            dn_nodes.append(reg.from_node.index)
            dn_node_adj.append(reg.from_node.index_adj)

    return m_dot_target, dn_nodes, dn_node_adj


def initialize_handler(
    initializer: Union[int, dict], network: "BlendPATH_network", cr_max: float
):
    logger.debug(
        f"Initializing network: {network.name}, using initializer {initializer}"
    )
    # Initialize pressure and set the state of each node

    # This scales the pressure between high and low values based on distance
    if initializer == 0:
        p_init = initialize_dist_based(network=network, cr_max=cr_max)
    # This calculates the node to node pressure drops based on mass flow rate and f guesses
    elif initializer == 2:
        p_init = initialize_2(network=network, cr_max=cr_max)
    elif isinstance(initializer, dict):
        p_init = initialize_with_values(
            network=network, cr_max=cr_max, initializer=initializer
        )
    else:
        raise RuntimeError(f"Unknown initialization hand :{initializer}")
    p_init = np.delete(p_init, network.ignore_nodes)
    return p_init


def initialize_2(network: "BlendPATH_network", cr_max: float):
    # Reset pipe and node x_h2 and pressures to none
    for pipe in network.pipes.values():
        pipe.x_h2 = None
    for node in network.nodes.values():
        node.x_h2 = None
        node.pressure = None

    m_dot_demands = sum(
        [demand.flowrate_mdot for demand in network.demand_nodes.values()]
    )

    # Set all pinned nodes
    start_node_names = []
    termination_nodes = {}
    pinned_node_indexes = set()
    x_h2 = None
    x_h2_sn = {}
    for sn in network.supply_nodes.values():
        sn.node.update_state(T=gl.T_FIXED, p=sn.pressure_mpa * gl.MPA2PA, x_h2=sn.blend)
        start_node_names.append(sn.node.name)
        pinned_node_indexes.add(sn.node.index)
        if x_h2 is None:
            x_h2 = sn.blend
        mw = network.composition.get_curvefit_mw(x_h2)
        mu = network.composition.get_curvefit_mu(
            p_gauge_pa=sn.pressure_mpa * gl.MPA2PA, x=x_h2
        )
        x_h2_sn[sn.node.name] = sn.blend

    for comp in network.compressors.values():
        comp.to_node.update_state(
            T=gl.T_FIXED,
            p=comp.pressure_out_mpa_g * gl.MPA2PA,
            x_h2=network.composition.x["H2"],
        )
        start_node_names.append(comp.to_node.name)
        pinned_node_indexes.add(comp.to_node.index)
        if comp.name == "Supply compressor":
            continue
        p_in = comp.pressure_out_mpa_g * gl.MPA2PA / cr_max
        comp.from_node.update_state(
            T=gl.T_FIXED, p=p_in, x_h2=network.composition.x["H2"]
        )
        termination_nodes[comp.from_node.name] = p_in
        pinned_node_indexes.add(comp.from_node.index)

    for reg in network.regulators.values():
        p_reg = reg.pressure_out_mpa_g * gl.MPA2PA
        reg.to_node.update_state(
            T=gl.T_FIXED, p=p_reg, x_h2=network.composition.x["H2"]
        )
        start_node_names.append(reg.to_node.name)
        termination_nodes[reg.from_node.name] = p_reg
        pinned_node_indexes.add(reg.to_node.index)

    for dn in network.demand_nodes.values():
        termination_nodes[dn.node.name] = dn.min_pressure_mpa_g * gl.MPA2PA

    p_init = np.zeros((len(start_node_names), network.n_nodes))
    x_init = np.full_like(p_init, np.nan)
    for sn_i, start_node in enumerate(start_node_names):
        # BFS from supply node ignoring direction => BFS distance
        pressures = {n: None for n in network.nodes.keys()}
        pressures[start_node] = network.nodes[start_node].pressure
        p_init[sn_i, network.nodes[start_node].index] = network.nodes[
            start_node
        ].pressure
        x_h2_seg = x_h2_sn.get(start_node, x_h2)
        if network.nodes[start_node].connections["Comp"]:
            if (
                x_h2_comp := network.compressors[
                    network.nodes[start_node].connections["Comp"][0].name
                ].to_node.x_h2
            ) is not None:
                x_h2_seg = x_h2_comp
        x_init[sn_i, network.nodes[start_node].index] = x_h2_seg

        queue = deque([start_node])

        # Set distances
        while queue:
            current_name = queue.popleft()
            current_node = network.nodes[current_name]
            current_pres = pressures[current_name]

            # Explore neighbors via pipes
            for pipe_conn in current_node.connections["Pipe"]:
                neighbor = (
                    pipe_conn.to_node
                    if pipe_conn.from_node == current_node
                    else pipe_conn.from_node
                )

                n_splits = min(max(len(current_node.connections["Pipe"]) - 1, 1), 3)
                m_dot_local = m_dot_demands / n_splits

                Re = 4 * m_dot_local / np.pi / (pipe_conn.diameter_mm * gl.MM2M) / mu
                f = friction_factor.get_friction_factor_vector(
                    Re=Re,
                    roughness_mm=pipe_conn.roughness_mm,
                    diameter_mm=pipe_conn.diameter_mm,
                    ff_type=network.ff_type,
                )
                p_out = get_p_out(
                    p_in=current_pres,
                    m_dot=m_dot_local,
                    A=np.pi * (pipe_conn.diameter_mm * gl.MM2M) ** 2 / 4,
                    mw=mw,
                    zrt=1 * ctu.gas_constant * gl.T_FIXED,
                    d=pipe_conn.diameter_mm * gl.MM2M,
                    f=f,
                    L=pipe_conn.length_km * gl.KM2M,
                )

                if pressures[neighbor.name] is None:
                    pressures[neighbor.name] = p_out
                    p_init[sn_i, network.nodes[neighbor.name].index] = p_out
                    x_init[sn_i, network.nodes[neighbor.name].index] = x_h2_seg
                    queue.append(neighbor.name)

    p_set = np.max(p_init, axis=0)
    x_set = np.nanmean(x_init, axis=0)
    for node in network.nodes.values():
        # Check nearby nodes
        p_set_value = p_set[node.index]
        for pipe_conn in node.connections["Pipe"]:
            if (
                pipe_conn.from_node is not node
                and pipe_conn.from_node.pressure == p_set_value
            ):
                p_set[node.index] += 0.01
            if (
                pipe_conn.to_node is not node
                and pipe_conn.to_node.pressure == p_set_value
            ):
                p_set[node.index] += 0.01
            pipe_conn.x_h2 = x_h2_sn.get(node.name, x_set[node.index])
        node.update_state(
            T=gl.T_FIXED,
            p=p_set[node.index],
            x_h2=x_h2_sn.get(node.name, x_set[node.index]),
        )
    return p_set


def get_p_out(
    p_in: float,
    m_dot: float,
    A: float,
    mw: float,
    zrt: float,
    d: float,
    f: float,
    L: float,
) -> float:
    """
    Calculate the outlet pressure (for initialization). Helper for init 2
    """
    c = (m_dot / (A * (mw / zrt * d / f / L) ** 0.5)) ** 2
    if c > p_in**2:
        return p_in * 0.99
    return (p_in**2 - c) ** 0.5


def initialize_dist_based(network: "BlendPATH_network", cr_max: float):
    # Preset array of initial pressures

    # Reset pipe and node x_h2 and pressures to none
    for pipe in network.pipes.values():
        pipe.x_h2 = None
    for node in network.nodes.values():
        node.x_h2 = None
        node.pressure = None

    # Set all pinned nodes
    start_node_names = []
    termination_nodes = {}
    pinned_node_indexes = set()
    x_h2 = None
    x_h2_sn = {}
    for sn in network.supply_nodes.values():
        if sn.pressure_mpa is None or np.isnan(sn.pressure_mpa):
            sn_p_set = sn.node.p_max_mpa_g
        else:
            sn_p_set = sn.pressure_mpa
        sn.node.update_state(T=gl.T_FIXED, p=sn_p_set * gl.MPA2PA, x_h2=sn.blend)
        start_node_names.append(sn.node.name)
        pinned_node_indexes.add(sn.node.index)
        x_h2 = float(sn.blend)
        if x_h2 is None:
            x_h2 = float(sn.blend)
        x_h2_sn[sn.node.name] = float(sn.blend)

    for comp in network.compressors.values():
        comp.to_node.update_state(
            T=gl.T_FIXED,
            p=comp.pressure_out_mpa_g * gl.MPA2PA,
            x_h2=network.composition.x["H2"],
        )
        start_node_names.append(comp.to_node.name)
        pinned_node_indexes.add(comp.to_node.index)
        if comp.name == "Supply compressor":
            continue
        p_in = comp.pressure_out_mpa_g * gl.MPA2PA / cr_max
        comp.from_node.update_state(
            T=gl.T_FIXED, p=p_in, x_h2=network.composition.x["H2"]
        )
        termination_nodes[comp.from_node.name] = p_in
        pinned_node_indexes.add(comp.from_node.index)

    for reg in network.regulators.values():
        p_reg = reg.pressure_out_mpa_g * gl.MPA2PA
        reg.to_node.update_state(
            T=gl.T_FIXED, p=p_reg, x_h2=network.composition.x["H2"]
        )
        start_node_names.append(reg.to_node.name)
        termination_nodes[reg.from_node.name] = p_reg
        pinned_node_indexes.add(reg.to_node.index)

    for dn in network.demand_nodes.values():
        termination_nodes[dn.node.name] = dn.min_pressure_mpa_g * gl.MPA2PA

    p_init = np.zeros((len(start_node_names), network.n_nodes))
    x_init = np.full_like(p_init.astype(float), np.nan)
    for sn_i, start_node in enumerate(start_node_names):
        # BFS from supply node ignoring direction => BFS distance
        distances = {n: None for n in network.nodes}
        distances[start_node] = 0.0
        x_h2_seg = x_h2_sn.get(start_node, x_h2)
        x_init[sn_i, network.nodes[start_node].index] = x_h2_seg

        queue = deque([start_node])
        term_pres = -1
        max_term_len = -1

        # Set distances
        while queue:
            current_name = queue.popleft()
            current_node = network.nodes[current_name]
            current_dist = distances[current_name]

            # Explore neighbors via pipes
            for pipe_conn in current_node.connections["Pipe"]:
                neighbor = (
                    pipe_conn.to_node.name
                    if pipe_conn.from_node.name == current_node.name
                    else pipe_conn.from_node.name
                )
                if pipe_conn.length_km == 0.0:
                    continue
                if distances[neighbor] is None:
                    distances[neighbor] = current_dist + pipe_conn.length_km
                    queue.append(neighbor)
                    x_init[sn_i, network.nodes[neighbor].index] = x_h2_seg
                if neighbor in termination_nodes:
                    term_pres = max(term_pres, termination_nodes[neighbor])
                    max_term_len = max(max_term_len, distances[neighbor])

        min_pres = max(term_pres, network.min_pressure_bound)
        numerator = max(network.nodes[start_node].pressure - min_pres, 0.0)
        slope = numerator / max(
            [d for d in distances.values() if d is not None] + [1e-12]
        )

        for n_name, dist in distances.items():
            if dist is None:
                continue
            p_init[sn_i, network.nodes[n_name].index] = (
                network.nodes[start_node].pressure - slope * dist
            )

    p_set = np.max(p_init, axis=0)
    x_set = np.nanmean(x_init, axis=0)
    for node in network.nodes.values():
        logger.debug(
            f"Initializing node: {node.name} to pressure={p_set[node.index]} and x={x_h2_sn.get(node.name, x_set[node.index])}"
        )
        node.update_state(
            T=gl.T_FIXED,
            p=p_set[node.index],
            x_h2=x_h2_sn.get(node.name, x_set[node.index]),
        )
        p_set_value = p_set[node.index]

        for pipe_conn in node.connections["Pipe"]:
            pipe_conn.x_h2 = x_h2_sn.get(node.name, x_set[node.index])
            if node.index in network.ignore_nodes:
                continue
            if (
                pipe_conn.from_node is not node
                and pipe_conn.from_node.pressure == p_set_value
            ):
                p_set[node.index] -= 0.01
            if (
                pipe_conn.to_node is not node
                and pipe_conn.to_node.pressure == p_set_value
            ):
                p_set[node.index] -= 0.01
    for pipe in network.pipes.values():
        logger.debug(f"Initializing pipe: {pipe.name} to x={x_h2}")
        pipe.x_h2 = x_h2

    return p_set


def get_stacked_demand_mdot(network: "BlendPATH_network"):
    nodes = [comp.to_node.name for comp in network.compressors.values()] + [
        reg.to_node.name for reg in network.regulators.values()
    ]

    demand_mdot = defaultdict(float)
    for dn in network.demand_nodes.values():
        if dn.node.name in nodes:
            demand_mdot[dn.node.name] += dn.flowrate_mdot

    return demand_mdot


def get_compressor_p_and_x(network: "BlendPATH_network", x_out: list):
    return np.array(
        [
            [
                x_out[comp.from_node.index],
                comp.from_node.pressure,
                comp.to_node.pressure,
            ]
            for comp in network.compressors.values()
        ]
    )


def initialize_with_values(
    network: "BlendPATH_network", cr_max: float, initializer: dict
):
    p_init = np.zeros(network.n_nodes)
    # Reset pipe and node x_h2 and pressures to none
    for node in network.nodes.values():
        pressure, x_h2 = initializer.get(node.name, (0, 0))
        node.x_h2 = x_h2
        node.pressure = pressure
        p_init[node.index] = pressure

    for sn in network.supply_nodes.values():
        if sn.pressure_mpa is None or np.isnan(sn.pressure_mpa):
            sn_p_set = sn.node.p_max_mpa_g
        else:
            sn_p_set = sn.pressure_mpa
        sn.node.update_state(T=gl.T_FIXED, p=sn_p_set * gl.MPA2PA, x_h2=sn.blend)
        p_init[sn.node.index] = sn_p_set * gl.MPA2PA

    for comp in network.compressors.values():
        comp.to_node.update_state(
            T=gl.T_FIXED,
            p=comp.pressure_out_mpa_g * gl.MPA2PA,
            x_h2=network.composition.x["H2"],
        )
        p_init[comp.to_node.index] = comp.pressure_out_mpa_g * gl.MPA2PA
        if comp.name == "Supply compressor":
            continue
        p_in = comp.pressure_out_mpa_g * gl.MPA2PA / cr_max
        comp.from_node.update_state(
            T=gl.T_FIXED, p=p_in, x_h2=network.composition.x["H2"]
        )

    for reg in network.regulators.values():
        p_reg = reg.pressure_out_mpa_g * gl.MPA2PA
        reg.to_node.update_state(
            T=gl.T_FIXED, p=p_reg, x_h2=network.composition.x["H2"]
        )
        p_init[reg.to_node.index] = p_reg

    for pipe in network.pipes.values():
        pipe.x_h2 = (pipe.from_node.x_h2 + pipe.to_node.x_h2) / 2

    return p_init


def check_h2_balance(network: "BlendPATH_network") -> None:
    h2_mw = 2.016

    h2_dn_out = sum(
        [
            dn.flowrate_mdot * dn.node.x_h2 * 2.016 / dn.node.mw
            for dn in network.demand_nodes.values()
        ]
    )
    h2_comp_out = sum(
        [
            compressor.fuel_mdot
            * compressor.from_node.x_h2
            * h2_mw
            / compressor.from_node.mw
            for compressor in network.compressors.values()
        ]
    )
    h2_out = h2_dn_out + h2_comp_out

    h2_in = abs(
        sum(
            [sn.mdot * sn.blend * h2_mw / sn.mw for sn in network.supply_nodes.values()]
        )
    )
    if h2_out <= 0:
        if abs(h2_in - h2_out) > 0.001:
            raise RuntimeError("H2 mass conservation does not balance")
    elif abs(h2_in - h2_out) / h2_out > 0.01:
        raise RuntimeError("H2 mass conservation does not balance")
