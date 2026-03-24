from typing import TYPE_CHECKING

import xlsxwriter

import BlendPATH.file_writing.file_writing as bp_file_w
import BlendPATH.Global as gl

if TYPE_CHECKING:
    from BlendPATH import BlendPATH_network


def write_info_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Info")
    info_out = [("Name", "Pressure (Pa)")]
    info_out.append(("Network name", network.name))
    info_out.append(("EOS", network.eos))

    row = 0
    for i in info_out:
        wks.write_row(row, 0, i)
        row += 1


def write_nodes_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Nodes
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Nodes")

    nodes_out = [
        (node.name, node.pressure, node.x_h2)
        for node in network.nodes.values()
        if node._report_out
    ]
    columns = ["Name", "Pressure (Pa)", "X_H2"]

    row = 0
    wks.write_row(row, 0, columns)
    row += 1
    for i in nodes_out:
        wks.write_row(row, 0, i)
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
    wks.write_row(row, 0, columns)
    row += 1
    for i in pipes_out:
        wks.write_row(row, 0, i)
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
    )

    row = 0
    wks.write_row(row, 0, columns)
    row += 1
    for i in comps_out:
        wks.write_row(row, 0, i)
        row += 1


def write_composition_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Composition
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Composition")

    composition_out = [(name, x) for name, x in network.composition.x.items()]
    columns = ["Name", "Molar fraction"]
    row = 0
    wks.write_row(row, 0, columns)
    row += 1
    for i in composition_out:
        wks.write_row(row, 0, i)
        row += 1


def write_regs_sheet(
    workbook: xlsxwriter.Workbook, network: "BlendPATH_network"
) -> None:
    # Composition
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
    wks.write_row(row, 0, columns)
    row += 1
    for i in regs_out:
        wks.write_row(row, 0, i)
        row += 1
