import math
from typing import TYPE_CHECKING

import numpy as np
import xlsxwriter

import BlendPATH.file_writing.file_writing as bp_file_w
import BlendPATH.Global as gl
import BlendPATH.network.pipeline_components.cantera_util as ctu

if TYPE_CHECKING:
    from BlendPATH import BlendPATH_network


def _excel_safe(value):
    """
    Convert values into xlsxwriter-safe cell values.

    Handles:
    - numpy scalars -> python scalars
    - numpy arrays  -> scalar if size==1, otherwise string
    - lists/tuples/sets -> string
    - NaN / Inf -> None
    """
    if value is None:
        return None

    # numpy scalar types: np.float64, np.int64, np.bool_, etc.
    if isinstance(value, np.generic):
        return _excel_safe(value.item())

    # numpy array
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _excel_safe(value.item())
        if value.size == 0:
            return None
        if value.size == 1:
            return _excel_safe(value.reshape(-1)[0].item())
        return str(value.tolist())

    # container types that xlsxwriter cannot write directly as a cell
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)

    # plain python floats: guard against nan / inf
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    return value


def _write_safe_row(wks, row: int, col: int, values) -> None:
    wks.write_row(row, col, [_excel_safe(v) for v in values])


def write_info_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Info")
    info_out = []
    info_out.append(("Network name", network.name))
    info_out.append(("EOS", network.eos))

    row = 0
    for i in info_out:
        _write_safe_row(wks, row, 0, i)
        row += 1


def write_nodes_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Nodes
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Nodes")

    nodes_out = []
    for node in network.nodes.values():
        if not node._report_out:
            continue

        plot_xy = getattr(node, "plot_xy", None)
        if plot_xy is None:
            plot_x, plot_y = None, None
        else:
            plot_x, plot_y = plot_xy

        nodes_out.append(
            (
                node.name,
                node.pressure,
                node.x_h2,
                plot_x,
                plot_y,
            )
        )

    columns = ["Name", "Pressure (Pa)", "X_H2", "plot_x", "plot_y"]

    row = 0
    _write_safe_row(wks, row, 0, columns)
    row += 1
    for i in nodes_out:
        _write_safe_row(wks, row, 0, i)
        row += 1


def write_pipes_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Pipes
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Pipes")

    pipes_out = [
        (
            pipe.name,
            pipe.from_node.name,
            pipe.to_node.name,
            pipe.m_dot,
            pipe.length_km,
            pipe.from_node.pressure,
            pipe.to_node.pressure,
            pipe.v_max,
            min(pipe.v_from, pipe.v_to),
            pipe.Re,
            pipe.f,
            pipe.x_h2,
        )
        for pipe in network.parent_pipes.values()
    ]

    columns = [
        "Name",
        "From node",
        "To node",
        "Flow rate (kg/s)",
        "Length (km)",
        "From node pressure (Pa)",
        "To node pressure (Pa)",
        "Max velocity (m/s)",
        "Min velocity (m/s)",
        "Reynolds number",
        "Friction factor",
        "Hydrogen concentration (molar)",
    ]

    row = 0
    _write_safe_row(wks, row, 0, columns)
    row += 1
    for i in pipes_out:
        _write_safe_row(wks, row, 0, i)
        row += 1


def write_comps_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Compressors
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Compressors")

    comps_out = [
        (
            comp.name,
            comp.from_node.name,
            comp.to_node.name,
            comp.compression_ratio,
            comp.fuel_use_MMBTU_hr,
            comp.shaft_power_MW,
            comp.shaft_power_MW * gl.MW2HP,
            comp.eta_comp_s,
            comp.eta_driver,
            comp.flow_mdot,
            comp.fuel_mdot,
        )
        for comp in network.compressors.values()
    ]

    columns = (
        "Name",
        "From node",
        "To node",
        "Pressure Ratio",
        "Fuel Consumption [MMBTU/hr]",
        "Shaft power [MW]",
        "Shaft power [hp]",
        "Isentropic efficiency",
        "Mechanical efficiency",
        "Flow rate through compressor (kg/s)",
        "Fuel flow rate (kg/s)",
    )

    row = 0
    _write_safe_row(wks, row, 0, columns)
    row += 1
    for i in comps_out:
        _write_safe_row(wks, row, 0, i)
        row += 1


def write_composition_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Composition
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Composition")

    composition_out = [(name, x) for name, x in network.composition.x.items()]
    columns = ["Name", "Molar fraction"]
    row = 0
    _write_safe_row(wks, row, 0, columns)
    row += 1
    for i in composition_out:
        _write_safe_row(wks, row, 0, i)
        row += 1

    row += 1
    _write_safe_row(wks, row, 0, ["Density at STP (25C)"])
    ctu.gas.TPX = 273.15 + 25, 101325, network.composition.x
    row += 1
    _write_safe_row(wks, row, 0, [ctu.gas.density_mass, "kg/m3"])


def write_regs_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Regulators
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Regulators")
    regs_out = [
        (reg.name, reg.from_node.name, reg.to_node.name, reg.pressure_ratio)
        for reg in network.regulators.values()
    ]
    columns = [
        "Name",
        "From node",
        "To node",
        "Pressure Ratio",
    ]
    row = 0
    _write_safe_row(wks, row, 0, columns)
    row += 1
    for i in regs_out:
        _write_safe_row(wks, row, 0, i)
        row += 1


def write_demands_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Demand nodes
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Demand")
    demands_out = [
        (
            dn.name,
            dn.node.name,
            dn.flowrate_MW,
            dn.flowrate_MMBTU_day,
            dn.flowrate_mdot,
            dn.node.heating_value(),
        )
        for dn in network.demand_nodes.values()
    ]
    columns = [
        "Name",
        "Node name",
        "Flowrate set point (MW)",
        "Flowrate set point (MMBTU/day)",
        "Flowrate simulated (kg/s)",
        "HHV (MJ/kg)",
    ]
    row = 0
    _write_safe_row(wks, row, 0, columns)
    row += 1
    for i in demands_out:
        _write_safe_row(wks, row, 0, i)
        row += 1


def write_supply_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Demand nodes
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Supply")
    supply_out = [
        (sn.name, sn.node.name, sn.mdot, sn.hhv, sn.node.pressure)
        for sn in network.supply_nodes.values()
    ]
    columns = [
        "Name",
        "Node name",
        "Flowrate simulated (kg/s)",
        "HHV (MJ/kg)",
        "Pressure (Pa_g)",
    ]
    row = 0
    _write_safe_row(wks, row, 0, columns)
    row += 1
    for i in supply_out:
        _write_safe_row(wks, row, 0, i)
        row += 1
