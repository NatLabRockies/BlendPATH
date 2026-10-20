import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import BlendPATH.Global as gl
import BlendPATH.network.pipeline_components as plc
import BlendPATH.network.pipeline_components.cantera_util as ctu
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg
from BlendPATH.network.pipeline_components import friction_factor
from BlendPATH.network.pipeline_components.regulator import RegulatorMode

if TYPE_CHECKING:
    from BlendPATH import BlendPATH_network

logger = logging.getLogger(__name__)


@dataclass
class Network_cxns:
    rows: list[int]
    cols: list[int]
    pipe_index: list[int]
    direction: list[int]
    cxn_matrix: sp.csr_matrix


def make_connections(
    network: "BlendPATH_network", shape_pn: tuple[int, int]
) -> Network_cxns:
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

    return Network_cxns(
        rows=rows,
        cols=cols,
        pipe_index=pipe_index,
        direction=np.asarray(coef),
        cxn_matrix=sp.csr_matrix(
            (nodal_data, (nodal_rows, nodal_cols)), shape=shape_pn
        ),
    )


def get_flow_targets(
    n_adj: int,
    network: "BlendPATH_network",
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Build nodal flow targets for a network containing regulators.
    Regulator flow is handled as a branch unknown in the Newton system.
    """
    for dn in network.demand_nodes.values():
        dn.recalc_mdot()

    m_dot_target = np.zeros(n_adj)

    balance_nodes: list[int] = []

    for dn in network.demand_nodes.values():
        if dn.node.index in network.ignore_nodes:
            continue
        m_dot_target[dn.node.index_adj] += dn.flowrate_mdot
        balance_nodes.append(dn.node.index)

    for sn in network.supply_nodes.values():
        if sn.is_pressure_supply:
            continue
        if sn.node.index in network.ignore_nodes:
            continue
        m_dot_target[sn.node.index_adj] -= sn.flowrate_mdot
        balance_nodes.append(sn.node.index)

    # Keep the existing compressor handling.
    for comp in network.compressors.values():
        if (
            not comp.to_node.connections["Pipe"]
            and comp.from_node.index_adj is not None
        ):
            balance_nodes.append(comp.from_node.index)

    # Get unique
    balance_nodes = list(dict.fromkeys(balance_nodes))
    balance_node_adj = [
        network.get_n_adj(node_idx)
        for node_idx in balance_nodes
        if network.get_n_adj(node_idx) is not None
    ]

    return m_dot_target, balance_nodes, balance_node_adj


def _build_regulator_mass_blocks(
    network: "BlendPATH_network",
    n_adj: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Build regulator branch-flow incidence blocks.
    Positive regulator flow is defined from reg.from_node to reg.to_node.
    In nodal-flow convention:
        from_node: -m_reg
        to_node:   +m_reg
    """

    if not network.regulators:
        return sp.csr_matrix(0), sp.csr_matrix(0)

    n_reg = len(network.regulators.values())

    rows_adj: list[int] = []
    cols_adj: list[int] = []
    data_adj: list[float] = []

    rows_full: list[int] = []
    cols_full: list[int] = []
    data_full: list[float] = []

    for reg_i, reg in enumerate(network.regulators.values()):
        to_adj = reg.to_node.index_adj
        from_adj = reg.from_node.index_adj

        if to_adj is None:
            raise ValueError(
                f"Regulator '{reg.name}' outlet node '{reg.to_node.name}' "
                "must be a solved pressure unknown"
            )

        rows_adj.append(to_adj)
        cols_adj.append(reg_i)
        data_adj.append(1.0)

        if from_adj is not None:
            rows_adj.append(from_adj)
            cols_adj.append(reg_i)
            data_adj.append(-1.0)

        rows_full.extend([reg.to_node.index, reg.from_node.index])
        cols_full.extend([reg_i, reg_i])
        data_full.extend([1.0, -1.0])

    reg_mass_adj = sp.csr_matrix(
        (data_adj, (rows_adj, cols_adj)),
        shape=(n_adj, n_reg),
    )

    reg_mass_full = sp.csr_matrix(
        (data_full, (rows_full, cols_full)),
        shape=(network.n_nodes, n_reg),
    )

    return reg_mass_adj, reg_mass_full


def _build_regulator_equation_block(
    regulators: list[plc.Regulator],
    n_adj: int,
    reg_flow_solving: np.ndarray,
) -> tuple:
    """
    Build regulator mode-equation blocks.
    CONTROL mode:
        p_out - p_set = 0
    WIDE_OPEN mode:
        p_out - p_in = 0
    CLOSED mode:
        m_reg = 0
    """
    n_reg = len(regulators)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    flow_rows: list[int] = []
    flow_cols: list[int] = []
    flow_data: list[float] = []

    rhs = np.zeros(n_reg)

    abs_residuals = []
    abs_flow_residuals = []

    for reg_i, reg in enumerate(regulators):
        to_adj = reg.to_node.index_adj
        from_adj = reg.from_node.index_adj

        if to_adj is None:
            raise ValueError(
                f"Regulator '{reg.name}' outlet node must be a solved pressure node"
            )

        if reg.mode == RegulatorMode.CONTROL:
            p_out = reg.to_node.pressure

            if p_out is None:
                raise ValueError(f"Regulator '{reg.name}' outlet pressure is not set")

            residual = p_out - reg.setpoint_pa

            rows.append(reg_i)
            cols.append(to_adj)
            data.append(1.0)

            abs_residuals.append(abs(residual))

        elif reg.mode == RegulatorMode.OPEN:
            p_out = reg.to_node.pressure
            p_in = reg.from_node.pressure

            if p_out is None:
                raise ValueError(f"Regulator '{reg.name}' outlet pressure is not set")

            if p_in is None:
                raise ValueError(f"Regulator '{reg.name}' inlet pressure is not set")

            residual = p_out - p_in

            rows.append(reg_i)
            cols.append(to_adj)
            data.append(1.0)

            if from_adj is not None:
                rows.append(reg_i)
                cols.append(from_adj)
                data.append(-1.0)

            abs_residuals.append(abs(residual))

        elif reg.mode == RegulatorMode.CLOSED:
            residual = reg_flow_solving[reg_i]

            flow_rows.append(reg_i)
            flow_cols.append(reg_i)
            flow_data.append(1.0)

            abs_flow_residuals.append(abs(residual))

        else:
            raise ValueError(f"Unknown regulator mode: {reg.mode}")

        rhs[reg_i] = -residual

    max_abs_flow_residual = max(abs_flow_residuals, default=0)
    max_abs_residual = max(abs_residuals, default=0)

    reg_eq_block = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_reg, n_adj),
    )

    reg_flow_block = sp.csr_matrix(
        (flow_data, (flow_rows, flow_cols)),
        shape=(n_reg, n_reg),
    )

    return (
        reg_eq_block,
        reg_flow_block,
        rhs,
        max_abs_residual,
        max_abs_flow_residual,
    )


def initialize_handler(
    initializer: int | dict, network: "BlendPATH_network", cr_max: float
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
        pressures = {n: None for n in network.nodes}
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


def initialization(
    initializer: int | dict[str, float],
    network: "BlendPATH_network",
    cr_max: Network_cxns,
):
    p_init = initialize_handler(initializer=initializer, network=network, cr_max=cr_max)

    # Assign node pressures
    for node in network.nodes.values():
        if node.index in network.ignore_nodes:
            continue
        node.pressure = p_init[node.index_adj]
    # Initialize regulator states
    for reg in network.regulators.values():
        reg.reset_solve_state()

    return p_init


@dataclass
class Pipe_jacobian_info:
    indices: list[tuple[float, float]]
    jacobian: sp.csr_matrix
    cxns: Network_cxns
    pipe_static_props: list[list[float]]


def pipe_jacobian_setup(network: "BlendPATH_network") -> Pipe_jacobian_info:
    cxns = make_connections(
        network=network, shape_pn=(network.n_nodes, len(network.pipes))
    )
    n_adj = network.n_nodes - len(network.ignore_nodes)
    # Generate jacobian shape with zeros
    jacobian = sp.csr_matrix(
        ([0.0] * len(cxns.rows), (cxns.rows, cxns.cols)), shape=(n_adj, n_adj)
    )
    # Store indices map
    j_inds = [
        (i, j)
        for i in range(n_adj)
        for j in jacobian.indices[jacobian.indptr[i] : jacobian.indptr[i + 1]]
    ]

    pipe_static_props = np.array(
        [pipe.get_pipe_static_properties() for pipe in network.pipes.values()]
    )

    return Pipe_jacobian_info(
        indices=j_inds,
        jacobian=jacobian,
        cxns=cxns,
        pipe_static_props=pipe_static_props,
    )


@dataclass
class Reg_jacobian_info:
    regulators: list[plc.Regulator]
    mass_adj: sp.csr_matrix
    mass_full: sp.csr_matrix


def reg_jacobian_setup(network: "BlendPATH_network"):
    regulators = list(network.regulators.values())

    n_adj = network.n_nodes - len(network.ignore_nodes)
    # TODO only have data updating
    reg_mass_adj, reg_mass_full = _build_regulator_mass_blocks(
        n_adj=n_adj, network=network
    )
    return Reg_jacobian_info(
        regulators=regulators, mass_adj=reg_mass_adj, mass_full=reg_mass_full
    )


@dataclass
class Regulator_jacobian:
    eq_block: sp.csr_matrix
    flow_block: sp.csr_matrix
    rhs: sp.csr_matrix
    mass_adj: sp.csr_matrix
    err_abs_pa: float
    err_abs_flow: float


def make_regulator_jacobian(
    reg_flow_solving: npt.NDArray[np.float64],
    n_adj: int,
    reg_jacobian_info: Reg_jacobian_info,
) -> tuple[npt.NDArray[np.float64] | float, Regulator_jacobian | None]:
    if not reg_jacobian_info.regulators:
        return 0.0, Regulator_jacobian(
            eq_block=sp.csr_matrix(0),
            flow_block=sp.csr_matrix(0),
            rhs=np.empty(0),
            mass_adj=reg_jacobian_info.mass_adj,
            err_abs_pa=0.0,
            err_abs_flow=0.0,
        )

    (
        reg_eq_block,
        reg_flow_block,
        rhs,
        reg_err_abs,
        reg_flow_err_abs,
    ) = _build_regulator_equation_block(
        regulators=reg_jacobian_info.regulators,
        n_adj=n_adj,
        reg_flow_solving=reg_flow_solving,
    )

    nodal_flow_reg = reg_jacobian_info.mass_full @ reg_flow_solving

    return (
        nodal_flow_reg,
        Regulator_jacobian(
            eq_block=reg_eq_block,
            flow_block=reg_flow_block,
            rhs=rhs,
            mass_adj=reg_jacobian_info.mass_adj,
            err_abs_pa=reg_err_abs,
            err_abs_flow=reg_flow_err_abs,
        ),
    )


def solve_linalg(
    pipe_jacobian: sp.csr_matrix,
    regulator_jacobian: Regulator_jacobian | None,
    delta_flow: npt.NDArray[np.float64],
    n_adj: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # If including regulators
    if regulator_jacobian.rhs.size > 0:
        delta_vals = scipy.sparse.linalg.spsolve(
            A=sp.bmat(
                [
                    [pipe_jacobian, regulator_jacobian.mass_adj],
                    [regulator_jacobian.eq_block, regulator_jacobian.flow_block],
                ],
                format="csr",
            ),
            b=np.concatenate([delta_flow, regulator_jacobian.rhs]),
        )
    # Otherwise, treat as normal (only solving pipes)
    else:
        delta_vals = scipy.sparse.linalg.spsolve(pipe_jacobian, delta_flow)

    if not np.all(np.isfinite(delta_vals)):
        raise ValueError("Non-finite Newton update in regulator solve")

    return delta_vals[:n_adj], delta_vals[n_adj:]


def err_calculation(demand_target, demand_actual, delta_flow):
    err_abs = np.max(np.abs(delta_flow))
    if demand_target.size > 0:
        valid = np.abs(demand_target) > 1e-12

        if np.any(valid):
            err_rel = np.max(
                np.abs(
                    (demand_target[valid] - demand_actual[valid]) / demand_target[valid]
                )
            )
        else:
            err_rel = 0.0
    else:
        err_rel = 0.0

    return err_abs, err_rel


def check_negative_pressure(
    p_solving: npt.NDArray[np.float64], delta_p: npt.NDArray[np.float64], c_relax: float
):
    # Check if negative pressure. Apply correction to see if it resolves
    if np.any(p_solving < 0):
        ind = np.where(p_solving < 0)
        p_solving[ind] -= delta_p[ind] * 0.99 / c_relax

        if np.any(p_solving < 0):
            raise ValueError("Negative pressure")


def check_reg_flows(reg_flow_solving: npt.NDArray[np.float64]):
    if not np.all(np.isfinite(reg_flow_solving)):
        raise ValueError("Non-finite regulator flow")


def exit_solving_loop(n_iter: int, limits: list[bool]) -> bool:
    if n_iter > gl.MAX_ITER:
        raise ValueError(f"Could not converge in {gl.MAX_ITER} iterations")

    # Return true if all conditions are met
    return all(limits)


@dataclass
class Tols:
    tol_abs: float
    tol_rel: float
    regulator_tol_pa: float
    regulator_recover_tol_pa: float | None

    def __post_init__(self):
        if self.regulator_recover_tol_pa is None:
            self.regulator_recover_tol_pa = self.regulator_tol_pa


def regulator_mode_updates(regulators: dict[str, plc.Regulator], tols: Tols):
    mode_changed = False

    for reg in regulators.values():
        changed = reg.update_mode(
            eps_pa=tols.regulator_tol_pa,
            recover_eps_pa=tols.regulator_recover_tol_pa,
            eps_mdot=tols.tol_abs,
        )

        if changed and reg.mode == RegulatorMode.CLOSED:
            reg.flow_mdot = 0.0

        mode_changed = mode_changed or changed

    return mode_changed


def convergence_checks(
    network: "BlendPATH_network",
    tols: Tols,
    p_solving: npt.NDArray[np.float64],
    low_p_buffer: float,
):
    check_regulator_violations(regulators=network.regulators, tols=tols)
    check_h2_balance(network=network)
    check_low_p_buffer(network=network, p_solving=p_solving, low_p_buffer=low_p_buffer)
    check_demand_p_limits(network=network)


def check_regulator_violations(regulators: dict[str, plc.Regulator], tols: Tols):
    for reg in regulators.values():
        reg.check_limits(
            tol_pa=tols.regulator_tol_pa,
            tol_mdot=tols.tol_abs,
        )


def check_low_p_buffer(
    network: "BlendPATH_network",
    p_solving: npt.NDArray[np.float64],
    low_p_buffer: float,
):
    if np.any(
        network.min_pressure_bound - p_solving
        >= network.min_pressure_bound * low_p_buffer
    ):
        raise ValueError("Pressure below threshold")


def check_demand_p_limits(
    network: "BlendPATH_network",
):
    for dn in network.demand_nodes.values():
        if dn.node.pressure is None:
            raise ValueError(
                f"Demand node '{dn.name}' at node '{dn.node.name}' "
                "has no solved pressure"
            )

        if dn.node.pressure < dn.min_pressure_mpa_g * gl.MPA2PA:
            raise ValueError(
                f"Demand node '{dn.name}' at node '{dn.node.name}' "
                f"has pressure {dn.node.pressure / gl.BAR2PA:.2f} bar, "
                f"below required minimum of {dn.min_pressure_mpa_g} MPa-g"
            )


def update_demand_node_mdots(
    network: "BlendPATH_network", nodal_flow_pipe: npt.NDArray[np.float64]
) -> None:
    stacking_demand = defaultdict(float)
    for dnode in network.demand_nodes.values():
        stacking_demand[dnode.node.index] += dnode.flowrate_mdot
    for dnode in network.demand_nodes.values():
        dnode.flowrate_mdot_sim = (
            nodal_flow_pipe[dnode.node.index]
            * dnode.flowrate_mdot
            / stacking_demand[dnode.node.index]
        )
