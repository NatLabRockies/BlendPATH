from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xlsxwriter

import BlendPATH.file_writing.file_writing as bp_file_w
import BlendPATH.Global as gl
from BlendPATH.costing.profast_util import (
    ProFAST_name_capital,
    ProFAST_name_feedstocks,
    ProFAST_name_fixedom,
)
from BlendPATH.network.pipeline_components import PipeSegmentDist
from BlendPATH.scenario_helper import Scenario_type
from BlendPATH.util.disclaimer import disclaimer_message
from BlendPATH.util.pipe_assessment import _POLYMER_CODES

if TYPE_CHECKING:
    from BlendPATH import BlendPATH_network, BlendPATH_scenario
    from BlendPATH.costing.costing import Costing_params
    from BlendPATH.modifications.mod_util import Mod_costing_params, Mod_result

# Levelized cost rows that only apply to transmission networks. In distribution
# result files they are left out when they are zero.
TRANSMISSION_ONLY_LCOT = {
    f"LCOT: {name}"
    for name in [
        *ProFAST_name_feedstocks,
        ProFAST_name_fixedom.ILI,
        ProFAST_name_capital.NEW_COMPR,
        ProFAST_name_capital.REFURB_COMPR,
        ProFAST_name_capital.SUPPLY_COMPR,
        ProFAST_name_capital.METER_TRANS,
    ]
}


def write_disclaimer_sheet(workbook: xlsxwriter.Workbook) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Disclaimer")
    wks.write(1, 0, disclaimer_message())


def write_inputs_sheet(
    workbook: xlsxwriter.Workbook, scenario: "BlendPATH_scenario"
) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Inputs")
    for row, i in enumerate(format_inputs(scenario)):
        wks.write_row(row, 0, i)


def write_results_sheet(
    workbook: xlsxwriter.Workbook,
    network: "BlendPATH_network",
    new_network: "BlendPATH_network",
    mod_result: "Mod_result",
    mod_costing_params: "Mod_costing_params",
    price_breakdown: dict[str, float],
    costing_params: "Costing_params",
    emissions: dict | None = None,
) -> None:

    wks_results = bp_file_w.add_worksheet(workbook=workbook, name="Results")
    is_distribution = network.scenario_type == Scenario_type.DISTRIBUTION

    comp_breakdown, supply_comp, comp_addl_rating, comp_cost_total = (
        new_network.get_comp_breakdown()
    )

    startrow = 0
    params_out = format_top_level_results(
        price_breakdown=price_breakdown,
        costing_params=costing_params,
        capacity_MMBTU_day=new_network.capacity_MMBTU_day,
        mod_result=mod_result,
        mod_costing_params=mod_costing_params,
        blend_ratio_energy=new_network.blend_ratio_energy,
        supply_comp=supply_comp,
        comp_addl_rating=comp_addl_rating,
        comp_cost_total=comp_cost_total,
        emissions=emissions,
        scenario_type=network.scenario_type,
    )

    for i in params_out:
        wks_results.write_row(startrow, 0, i)
        startrow += 1

    # Add 4 row gap
    startrow += 4

    # Write out any velocity warnings:
    # mach number or erosional velocity
    flag_mach, flag_erosional = new_network.velocity_violations
    velocity_warnings = format_velocity_warnings(
        flag_mach=flag_mach, flag_erosional=flag_erosional
    )
    for v_warning in velocity_warnings:
        color_text = workbook.add_format({"font_color": v_warning[1]})
        wks_results.write(
            startrow,
            0,
            v_warning[0],
            color_text,
        )
        startrow += 1
    startrow += (flag_mach or flag_erosional) * 2

    #
    # Breakdown of original pipe
    # Loop thru pipes
    if is_distribution:
        orig_title = (
            "Breakdown of original pipe by diameter, thickness/schedule, and material"
        )
        orig_pipes = format_original_pipes_distribution(network)
    else:
        orig_title = "Breakdown of original pipe by diameter, schedule and grade"
        orig_pipes = format_original_pipes(network)
    wks_results.write(startrow, 0, orig_title)
    startrow += 1
    for i in orig_pipes:
        wks_results.write_row(startrow, 0, i)
        startrow += 1

    startrow += 4

    #
    # Breakdown of new pipe
    if is_distribution:
        new_title = (
            "Breakdown of new pipe by diameter, thickness/schedule, and material"
        )
        new_pipe_condensed = format_new_pipes_distribution(mod_result=mod_result)
    else:
        new_title = "Breakdown of new pipe by diameter, schedule and grade"
        new_pipe_condensed = format_new_pipes(mod_result=mod_result)
    wks_results.write(startrow, 0, new_title)
    startrow += 1

    for i in new_pipe_condensed:
        wks_results.write_row(startrow, 0, i)
        startrow += 1
    startrow += 4

    # Distribution networks without compressors: skip the compressor table
    if is_distribution and not comp_breakdown:
        return

    #
    # Breakdown of compressors
    wks_results.write(startrow, 0, "Breakdown of compressor by station")
    startrow += 1
    wks_results.write_row(
        startrow,
        0,
        [
            "Station ID",
            "Shaft power (MW)",
            "Original Capacity (MW)",
            "Required Additional Capacity (hp)",
            "Cost (2020$)",
            "Revamp Cost (2020$)",
            "Fuel usage (MMBTU/hr)",
            "Electric power (kW)",
        ],
    )
    startrow += 1
    for i in comp_breakdown:
        wks_results.write_row(startrow, 0, i)
        startrow += 1

    startrow += 4


def write_mod_network_sheet(
    workbook: xlsxwriter.Workbook,
    network: "BlendPATH_network",
    new_network: "BlendPATH_network",
    mod_result: "Mod_result",
    mod_type: str,
) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Modified network design")

    nodes_map = {}
    for ps_i, ps in enumerate(network.pipe_segments):
        nodes_map.update({x.name: ps_i for x in ps.nodes})

    pipe_profile = []
    segment = 0
    new_pipes_map = {pipe.name: pipe.pressure for pipe in mod_result.new_pipes.values()}
    for pipe in new_network.parent_pipes.values():
        # Get pipe segment
        if pipe.from_node.name in nodes_map:
            segment = nodes_map[pipe.from_node.name]
        elif pipe.to_node.name in nodes_map:
            segment = nodes_map[pipe.to_node.name]
        # Get new/existing
        new_old = "Existing"
        maop = pipe.p_max_mpa_g
        if mod_type in ["pl"]:
            if (
                network.scenario_type == Scenario_type.TRANSMISSION
                and pipe.name in new_pipes_map
            ):
                new_old = "New"
                maop = new_pipes_map[pipe.name]
            if (
                network.scenario_type == Scenario_type.DISTRIBUTION
                and pipe.name.endswith("_mod_looped")
            ):
                new_old = "New"
                maop = mod_result.new_pipes[pipe.name].pressure

        if mod_type in ["dr"] and pipe.name in mod_result.new_pipes:
            new_old = "New"
            maop = mod_result.new_pipes[pipe.name].pressure
        if mod_type == "newh2":
            new_old = "New"

        # Polymer pipes report their SDR and polymer designation
        if getattr(pipe, "material", "") == "polymer":
            schedule = f"SDR {round(pipe.dimension_ratio, 1):g}"
            material = f"{pipe.polymer}{pipe.polymer_designation}"
        else:
            schedule = getattr(pipe, "schedule", "")
            material = getattr(pipe, "grade", "")

        pipe_profile.append(
            (
                segment,
                pipe.name,
                pipe.from_node.name,
                pipe.to_node.name,
                new_old,
                pipe.m_dot,
                pipe.DN,
                schedule,
                pipe.thickness_mm,
                material,
                maop,
                pipe.length_km,
                pipe.length_km * gl.KM2MI,
                pipe.from_node.pressure,
                pipe.to_node.pressure,
                pipe.v_max,
                pipe.mach_number,
                pipe.erosional_velocity_ASME,
            )
        )
    is_distribution = network.scenario_type == Scenario_type.DISTRIBUTION
    columns = [
        "Pipe segment",
        "Pipe name",
        "FromName",
        "ToName",
        "Type",
        "Flow rate (kg/s)",
        "DN",
        "Schedule/SDR" if is_distribution else "Schedule",
        "Thickness (mm)",
        "Material" if is_distribution else "Steel grade",
        "MAOP (MPa-g)",
        "Length (km)",
        "Length (mi)",
        "Inlet pressure (Pa-g)",
        "Outlet pressure (Pa-g)",
        "Max velocity (m/s)",
        "Max Mach number",
        "Erosional velocity (m/s)",
    ]
    row = 0
    wks.write_row(row, 0, columns)
    row = 1
    for i in pipe_profile:
        wks.write_row(row, 0, i)
        row += 1


def write_comp_sheet(
    workbook: xlsxwriter.Workbook,
    network: "BlendPATH_network",
    new_network: "BlendPATH_network",
    mod_result: "Mod_result",
    mod_type: str,
) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Compressor design")

    ps_lengths = [0]
    existing_cs_lengths = {}
    for ps in network.pipe_segments:
        if isinstance(ps, PipeSegmentDist):
            continue
        ps_lengths.append(ps.length_km + ps_lengths[-1])
        if ps.comps:
            existing_cs_lengths[ps.comps[0].name] = ps_lengths[-1]
    new_comp_breakdown = []
    comp_i = 0
    ps_i = 0
    for comp in new_network.compressors.values():
        segment_name = ""
        comp_type = "Existing"
        if comp.original_rating_MW == 0:
            comp_type = "New"
        if comp.original_rating_MW == 0 and mod_type in [
            "ac",
            "additional_compressors",
        ]:
            comp_type = "New"
            if comp.name == "Supply compressor":
                comp_length = 0
            else:
                segment_name = ps_i
                while not mod_result.l_comps[ps_i]:
                    ps_i += 1
                length = mod_result.l_comps[ps_i][comp_i]
                comp_length = length + ps_lengths[ps_i]
                comp_i += 1
                if comp_i == mod_result.n_comps[ps_i]:
                    ps_i += 1
                    comp_i = 0

        elif comp.name in existing_cs_lengths:
            comp_length = existing_cs_lengths[comp.name]
        else:
            comp_length = 0
        eta_s = comp.eta_comp_s if comp.fuel_extract else comp.eta_comp_s_elec
        eta_driver = comp.eta_driver if comp.fuel_extract else comp.eta_driver_elec_used

        new_comp_breakdown.append(
            (
                segment_name,
                comp.name,
                comp.from_node.name,
                comp.to_node.name,
                comp_type,
                comp_length,
                comp_length * gl.KM2MI,
                comp.compression_ratio,
                comp.fuel_use_MMBTU_hr,
                comp.shaft_power_MW,
                comp.shaft_power_MW * gl.MW2HP,
                comp.fuel_electric_W / gl.KW2W,
                max([comp.shaft_power_MW, comp.original_rating_MW]),
                eta_s,
                eta_driver if not np.isnan(eta_driver) else "",
                comp.cost,
                comp.revamp_cost,
                comp.cost + comp.revamp_cost,
            )
        )
    columns = [
        "Segment",
        "Name",
        "FromName",
        "ToName",
        "Type",
        "Cumulative length (km)",
        "Cumulative length (mi)",
        "Pressure Ratio",
        "Fuel consumption (MMBTU/hr)",
        "Shaft power (MW)",
        "Shaft power (hp)",
        "Electric power (kW)",
        "Rating (MW)",
        "Isentropic efficiency",
        "Driver efficiency",
        "Cost ($)",
        "Revamp cost ($)",
        "Total cost ($)",
    ]
    row = 0
    wks.write_row(row, 0, columns)
    row = 1
    for i in new_comp_breakdown:
        wks.write_row(row, 0, i)
        row += 1


def write_pressure_sheet(
    workbook: xlsxwriter.Workbook, new_network: "BlendPATH_network"
) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Pressure profile")

    pressure_breakdown = []

    for node in new_network.nodes.values():
        if not node._report_out:
            continue

        plot_xy = getattr(node, "plot_xy", None)
        if plot_xy is None:
            plot_x, plot_y = None, None
        else:
            plot_x, plot_y = plot_xy

        pressure_breakdown.append(
            (
                node.name,
                node.pressure,
                plot_x,
                plot_y,
            )
        )

    columns = ["Node", "Pressure (Pa-g)", "plot_x", "plot_y"]

    row = 0
    wks.write_row(row, 0, columns)
    row = 1

    for i in pressure_breakdown:
        wks.write_row(row, 0, i)
        row += 1


def write_demand_sheet(
    workbook: xlsxwriter.Workbook, new_network: "BlendPATH_network"
) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Demand error")
    demand_error = []

    columns = [
        "Demand node name",
        "Flow rate set point (kg/s)",
        "Flow rate calculated (kg/s)",
        "Higher heating value (MJ/kg)",
        "Energy set point (MW)",
        "Energy calculated (MW)",
        "Error in energy (%)",
    ]

    for d_node in new_network.demand_nodes.values():
        hhv = d_node.node.heating_value()
        demand_error.append(
            (
                d_node.name,
                d_node.flowrate_mdot,
                d_node.flowrate_mdot_sim,
                hhv,
                d_node.flowrate_mdot * hhv,
                d_node.flowrate_mdot_sim * hhv,
                (d_node.flowrate_mdot_sim - d_node.flowrate_mdot)
                / d_node.flowrate_mdot
                * 100,
            )
        )

    row = 0
    wks.write_row(row, 0, columns)
    row = 1
    for i in demand_error:
        wks.write_row(row, 0, i)
        row += 1


def write_profast_sheet(workbook: xlsxwriter.Workbook, pf_inputs: dict) -> None:
    ### ProFAST inputs for repeating analysis
    sheet_name = "ProFAST"
    worksheet = bp_file_w.add_worksheet(workbook=workbook, name=sheet_name)
    startrow = 0
    for value in pf_inputs.values():
        for i, v in value.items():
            worksheet.write(startrow, 0, i)
            if isinstance(v, dict):
                col_adder = 0
                for j, w in v.items():
                    worksheet.write(startrow, 1 + col_adder, j)
                    worksheet.write(startrow, 1 + col_adder + 1, str(w))
                    col_adder += 2
            else:
                worksheet.write(startrow, 1, str(v))
            startrow += 1


def write_capital_sheet(
    workbook: xlsxwriter.Workbook, mod_costing_params: "Mod_costing_params"
) -> None:
    ### ProFAST inputs for repeating analysis
    sheet_name = "Capital"
    worksheet = bp_file_w.add_worksheet(workbook=workbook, name=sheet_name)
    startrow = 0

    worksheet.write_row(startrow, 0, ["Item", "Value ($)", "Category"])
    startrow += 1

    cost_categories = [
        ("Network", mod_costing_params.capital.network_costs()),
        ("User", mod_costing_params.capital.user_costs()),
    ]

    for category, cat_dict in cost_categories:
        for name, value in cat_dict.items():
            worksheet.write_row(
                startrow,
                0,
                [name, value, category],
            )
            startrow += 1


def format_dict_to_str(dict_in: dict) -> str:
    if isinstance(dict_in, dict):
        return "{" + ",".join([f"{i}:{v}" for i, v in dict_in.items()]) + "}"
    return dict_in


def format_inputs(scenario: "BlendPATH_scenario") -> list:
    return [
        ("Name", "Value"),
        ("Network name", scenario.casestudy_name),
        ("Save directory", scenario.results_dir),
        ("Design option - original", scenario.design_option),
        ("Location class", scenario.design_params.asme.location_class),
        ("Joint factor", scenario.design_params.asme.joint_factor),
        ("T de-rating factor", scenario.design_params.asme.T_rating),
        ("Compression ratio", str(scenario.design_params.max_CR)),
        ("Blending ratio", scenario.blend),
        (
            "Natural gas cost ($/MMBTU)",
            format_dict_to_str(scenario.costing_params.ng_price),
        ),
        ("H2 cost ($/kg)", format_dict_to_str(scenario.costing_params.h2_price)),
        (
            "Electricity cost ($/kWh)",
            format_dict_to_str(scenario.costing_params.elec_price),
        ),
        (
            "Final outlet pressure (MPa-g)",
            scenario.design_params.final_outlet_pressure_mpa_g,
        ),
        ("Region", scenario.costing_params.region),
        ("Modification method", scenario.mod_type),
        ("Modification design option", scenario.design_option_new),
        ("Equation of state", scenario.eos),
        ("Inline inspection inspection interval", scenario.costing_params.ili_interval),
        (
            "Original pipeline depreciated cost",
            scenario.costing_params.original_pipeline_cost,
        ),
        (
            "Are added compressors electric?",
            scenario.design_params.new_comp_elec,
        ),
        (
            "Are existing compressors converted to electric?",
            scenario.design_params.existing_comp_elec,
        ),
        (
            "New gas compressor isentropic efficiency",
            scenario.design_params.new_comp_eta_s,
        ),
        (
            "New electric compressor isentropic efficiency",
            scenario.design_params.new_comp_eta_s_elec,
        ),
        (
            "New gas compressor driver efficiency",
            scenario.design_params.new_comp_eta_driver,
        ),
        (
            "New electric compressor driver efficiency",
            (
                ""
                if np.isnan(scenario.design_params.new_comp_eta_driver_elec)
                else scenario.design_params.new_comp_eta_driver_elec
            ),
        ),
        (
            "Pipe price markup",
            scenario.costing_params.pipe_markup,
        ),
        (
            "Compressor price markup",
            scenario.costing_params.compressor_markup,
        ),
        (
            "Financial overrides",
            str(scenario.financial_overrides),
        ),
        (
            "Filename suffix",
            scenario.filename_suffix,
        ),
        (
            "Thermodynamic curvefits enabled",
            scenario.thermo_curvefit,
        ),
        (
            "Composition tracking enabled",
            scenario.composition_tracking,
        ),
        (
            "Scenario type",
            scenario.scenario_type,
        ),
        (
            "Polymer cost type",
            scenario.costing_params.polymer_cost_type,
        ),
        (
            "Max velocity allowed in distribution networks",
            scenario.design_params.max_dist_velocity,
        ),
    ]


def format_top_level_results(
    price_breakdown: pd.DataFrame,
    costing_params: "Costing_params",
    capacity_MMBTU_day: float,
    mod_result: "Mod_result",
    mod_costing_params: "Mod_costing_params",
    blend_ratio_energy: float,
    supply_comp: float,
    comp_addl_rating: float,
    comp_cost_total: float,
    emissions: dict | None = None,
    scenario_type: str | None = None,
) -> list:
    is_distribution = scenario_type == Scenario_type.DISTRIBUTION
    # Fill in levelized costs
    params_out = [("Parameter", "Value", "Units")]
    for i, v in price_breakdown.items():
        if is_distribution and i in TRANSMISSION_ONLY_LCOT and v == 0:
            continue
        params_out.append((i, v, "$/MMBTU"))
    # Fill in other params
    params_out.append(
        (
            "Hydrogen injection price",
            format_dict_to_str(costing_params.h2_price),
            "$/kg",
        )
    )
    params_out.append(
        ("Natural gas price", format_dict_to_str(costing_params.ng_price), "$/MMBTU")
    )
    if isinstance(costing_params.ng_price, dict) or isinstance(
        costing_params.h2_price, dict
    ):
        params_out.append(
            (
                "Blended gas price",
                format_dict_to_str(costing_params.cf_price),
                "$/MMBTU",
            )
        )
    else:
        params_out.append(
            (
                "Blended gas price",
                costing_params.cf_price[min(costing_params.cf_price.keys())],
                "$/MMBTU",
            )
        )
    params_out.append(("Pipeline capacity (daily)", capacity_MMBTU_day, "MMBTU/day"))
    params_out.append(
        (
            "Pipeline capacity (hour)",
            capacity_MMBTU_day / gl.DAY2HR,
            "MMBTU/hr",
        )
    )
    pipeline_added_km = sum(mod_result.cap_cost["length"])
    params_out.append(("Added pipeline", pipeline_added_km, "km"))
    params_out.append(("Added pipeline", pipeline_added_km * gl.KM2MI, "mi"))

    compressor_rows = [
        ("Added compressor stations", sum(mod_result.n_comps) + supply_comp, ""),
        ("Added compressor capacity", comp_addl_rating, "hp"),
        (
            "Compressor fuel usage",
            mod_costing_params.feedstocks.all_fuel_MW * gl.MW2MMBTUDAY,
            "MMBTU/day",
        ),
        (
            "Compressor fuel usage",
            mod_costing_params.feedstocks.all_fuel_MW * gl.MW2MMBTUDAY / gl.DAY2HR,
            "MMBTU/hr",
        ),
        (
            "Compressor fuel usage (electric)",
            mod_costing_params.feedstocks.all_fuel_elec_kW,
            "kW",
        ),
    ]
    # Distribution networks without compressors: skip the compressor rows
    if not (is_distribution and all(row[1] == 0 for row in compressor_rows)):
        params_out.extend(compressor_rows)
    params_out.append(
        (
            "New pipe",
            (
                0
                if not mod_result.cap_cost["total cost"]
                else mod_result.cap_cost["total cost"]
            ),
            "$",
        )
    )
    compressor_cost_rows = [
        ("New compressor stations", comp_cost_total, "$"),
        (
            "Compressor station refurbishment",
            sum(mod_costing_params.capital.revamped_comp_capex),
            "$",
        ),
    ]
    if not (is_distribution and all(row[1] == 0 for row in compressor_cost_rows)):
        params_out.extend(compressor_cost_rows)
    if not is_distribution:
        params_out.append(
            ("Meter station modification", mod_costing_params.capital.meter_cost, "$")
        )
    params_out.append(
        ("Valve modifications", mod_costing_params.capital.valve_cost, "$")
    )
    # params_out.append(("Original network residual value", 0, "$"))

    params_out.append(("Meter set assembly", mod_costing_params.capital.msa_cost, "$"))

    params_out.append(("Hydrogen energy ratio", blend_ratio_energy * 100, "%"))

    # --- Emissions outputs (transmission only) ---
    if emissions is not None and not is_distribution:
        if (intensity := emissions.get("intensity_kgCO2e_per_MMBTU")) is not None:
            params_out.append(
                ("Emissions intensity (total)", intensity, "kgCO2e/MMBTU")
            )
        if (annual := emissions.get("annual_MMTCO2e_per_yr")) is not None:
            params_out.append(("Total emissions (annual)", annual, "MMTCO2e/yr"))

    return params_out


def mach_number_warning_text() -> str:
    return "ERROR: Mach number exceeds 1, results may not be valid"


def erosional_velocity_warning_text() -> str:
    return "WARNING: Pipeline gas velocities in pipeline exceed the ASME B31.12 para. I-3.4.5 erosional velocity. Consider changing BlendPATH input parameters to lower gas velocities or further investigate integrity risks associated with the high gas velocities in the modified pipeline design."


def format_velocity_warnings(flag_mach: bool, flag_erosional: bool):
    return_text = []
    if flag_mach:
        return_text.append((mach_number_warning_text(), "red"))
    if flag_erosional:
        return_text.append((erosional_velocity_warning_text(), "orange"))
    return return_text


def format_original_pipes(network: "BlendPATH_network") -> list:
    orig_pipes = {
        x: []
        for x in [
            "DSG",
            "Pipe Nominal Diameter",
            "Length (km)",
            "Grade",
            "Schedule",
            "Polymer",
        ]
    }
    for pipe in network.pipes.values():
        dsg = f"{pipe.DN}_{getattr(pipe, 'grade', '')}_{getattr(pipe, 'schedule', '')}"
        if dsg in orig_pipes["DSG"]:
            dsg_ind = orig_pipes["DSG"].index(dsg)
            orig_pipes["Length (km)"][dsg_ind] += pipe.length_km
        else:
            orig_pipes["DSG"].append(dsg)
            orig_pipes["Pipe Nominal Diameter"].append(pipe.DN)
            orig_pipes["Length (km)"].append(pipe.length_km)
            orig_pipes["Grade"].append(getattr(pipe, "grade", ""))
            orig_pipes["Schedule"].append(getattr(pipe, "schedule", ""))
            orig_pipes["Polymer"].append(
                f"{getattr(pipe, 'polymer', '')}{getattr(pipe, 'polymer_designation', '')}"
            )
    orig_pipes.pop("DSG")

    list_out = []
    list_out.append(list(orig_pipes.keys()))
    for i, val in enumerate(orig_pipes["Pipe Nominal Diameter"]):
        list_out.append([orig_pipes[col][i] for col in orig_pipes])
    return list_out


def format_new_pipes(mod_result: "Mod_result") -> list:
    new_pipe_condensed = {
        x: []
        for x in [
            "Pipe Nominal Diameter",
            "Length (km)",
            "Steel Grade",
            "Schedule",
            "Material Cost ($)",
            "Labor Cost ($)",
            "Misc ($)",
            "ROW ($)",
        ]
    }
    if mod_result.cap_cost["D_S_G"]:
        new_pipe_condensed["Pipe Nominal Diameter"] = mod_result.cap_cost["DN"]
        new_pipe_condensed["Length (km)"] = mod_result.cap_cost["length"]
        new_pipe_condensed["Steel Grade"] = mod_result.cap_cost["grade"]
        new_pipe_condensed["Schedule"] = mod_result.cap_cost["sch"]
        new_pipe_condensed["Material Cost ($)"] = mod_result.cap_cost["mat_cost"]
        for i in mod_result.cap_cost["other_pipe_cost"]:
            new_pipe_condensed["Labor Cost ($)"].append(i.get("Labor", 0))
            new_pipe_condensed["Misc ($)"].append(i.get("Misc", 0))
            new_pipe_condensed["ROW ($)"].append(i.get("ROW", 0))

    list_out = []
    list_out.append(list(new_pipe_condensed.keys()))
    for i, val in enumerate(new_pipe_condensed["Pipe Nominal Diameter"]):
        list_out.append([new_pipe_condensed[col][i] for col in new_pipe_condensed])

    return list_out


def format_original_pipes_distribution(network: "BlendPATH_network") -> list:
    """
    Original pipe breakdown for distribution networks, where steel and polymer
    pipes share one material column and one schedule/SDR column
    """
    columns = ["Pipe Nominal Diameter", "Length (km)", "Material", "Schedule/SDR"]
    groups = {}
    for pipe in network.pipes.values():
        if getattr(pipe, "material", "") == "polymer":
            material = f"{pipe.polymer}{pipe.polymer_designation}"
            schedule_sdr = f"SDR {round(pipe.dimension_ratio, 1):g}"
        else:
            # steel pipes that do not match a schedule report their wall thickness
            material = getattr(pipe, "grade", "")
            schedule_sdr = (
                getattr(pipe, "schedule", None) or f"{round(pipe.thickness_mm, 2):g} mm"
            )
        key = (pipe.DN, material, schedule_sdr)
        groups[key] = groups.get(key, 0) + pipe.length_km

    list_out = [columns]
    for (dn, material, schedule_sdr), length_km in groups.items():
        list_out.append([dn, length_km, material, schedule_sdr])
    return list_out


def format_new_pipes_distribution(mod_result: "Mod_result") -> list:
    """
    New pipe breakdown for distribution networks. Pipe costs come from a single
    all-in unit cost, so only the total is reported.
    """
    columns = [
        "Pipe Nominal Diameter",
        "Length (km)",
        "Material",
        "Schedule/SDR",
        "Material Cost ($)",
        "Labor Cost ($)",
        "Misc ($)",
        "ROW ($)",
        "Total ($)",
    ]
    cap_cost = mod_result.cap_cost

    # SDR (polymer) or wall thickness (steel) of the new pipes in each cost row
    schedule_sdr = {}
    for new_pipe in mod_result.new_pipes.values():
        if new_pipe.th <= 0:
            continue
        if str(new_pipe.grade).strip().upper()[:2] in _POLYMER_CODES:
            sdr = (new_pipe.inner_diameter + 2 * new_pipe.th) / new_pipe.th
            label = f"SDR {round(sdr, 1):g}"
        else:
            label = f"{round(new_pipe.th, 2):g} mm"
        schedule_sdr.setdefault(new_pipe.D_S_G, set()).add(label)

    list_out = [columns]
    for i, dsg in enumerate(cap_cost["D_S_G"]):
        other_cost = sum(
            cap_cost["other_pipe_cost"][i][x] for x in mod_result.pipe_cost_types
        )
        list_out.append(
            [
                cap_cost["DN"][i],
                cap_cost["length"][i],
                cap_cost["grade"][i],
                cap_cost["sch"][i] or ", ".join(sorted(schedule_sdr.get(dsg, []))),
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                cap_cost["mat_cost"][i] + other_cost,
            ]
        )
    return list_out


def write_regulators_sheet(
    workbook: xlsxwriter.Workbook, new_network: "BlendPATH_network"
) -> None:
    """
    Regulator operating state in the modified network
    """
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Regulators")
    columns = [
        "Regulator",
        "From node",
        "To node",
        "Mode",
        "Setpoint (Pa-g)",
        "Inlet pressure (Pa-g)",
        "Outlet pressure (Pa-g)",
        "Flow rate (kg/s)",
    ]
    wks.write_row(0, 0, columns)
    for row, regulator in enumerate(new_network.regulators.values(), start=1):
        wks.write_row(
            row,
            0,
            [
                regulator.name,
                regulator.from_node.name,
                regulator.to_node.name,
                str(regulator.mode),
                regulator.setpoint_pa,
                regulator.from_node.pressure,
                regulator.to_node.pressure,
                regulator.flow_mdot,
            ],
        )
