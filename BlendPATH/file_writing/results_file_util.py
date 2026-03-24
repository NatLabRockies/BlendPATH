from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xlsxwriter

import BlendPATH.file_writing.file_writing as bp_file_w
import BlendPATH.Global as gl
from BlendPATH.scenario_helper import Scenario_type
from BlendPATH.util.disclaimer import disclaimer_message

if TYPE_CHECKING:
    from BlendPATH import BlendPATH_network, BlendPATH_scenario
    from BlendPATH.costing.costing import Costing_params
    from BlendPATH.modifications.mod_util import Mod_costing_params, Mod_result


def write_disclaimer_sheet(workbook: xlsxwriter.Workbook) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Disclaimer")
    wks.write(1, 0, disclaimer_message())
    wks.set_column("A:A", 100, workbook.add_format({"text_wrap": True}))


def write_inputs_sheet(
    workbook: xlsxwriter.Workbook, scenario: "BlendPATH_scenario"
) -> None:
    wks = bp_file_w.add_worksheet(workbook=workbook, name="Inputs")
    row = 0
    for i in format_inputs(scenario):
        wks.write_row(row, 0, i)
        row += 1


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
    wks_results.write(
        startrow, 0, "Breakdown of original pipe by diameter, schedule and grade"
    )
    startrow += 1
    orig_pipes = format_original_pipes(network)
    for i in orig_pipes:
        wks_results.write_row(startrow, 0, i)
        startrow += 1

    startrow += 4

    #
    # Breakdown of new pipe
    wks_results.write(
        startrow, 0, "Breakdown of new pipe by diameter, schedule and grade"
    )
    startrow += 1
    new_pipe_condensed = format_new_pipes(mod_result=mod_result)

    for i in new_pipe_condensed:
        wks_results.write_row(startrow, 0, i)
        startrow += 1
    startrow += 4

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

        if mod_type in ["dr"] and pipe.name in mod_result.new_pipes.keys():
            new_old = "New"
            maop = mod_result.new_pipes[pipe.name].pressure
        if mod_type == "newh2":
            new_old = "New"

        pipe_profile.append(
            (
                segment,
                pipe.name,
                pipe.from_node.name,
                pipe.to_node.name,
                new_old,
                pipe.m_dot,
                pipe.DN,
                getattr(pipe, "schedule", ""),
                pipe.thickness_mm,
                getattr(pipe, "grade", ""),
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
    columns = [
        "Pipe segment",
        "Pipe name",
        "FromName",
        "ToName",
        "Type",
        "Flow rate (kg/s)",
        "DN",
        "Schedule",
        "Thickness (mm)",
        "Steel grade",
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
        ps_lengths.append(ps.length_km + ps_lengths[-1])
        if ps.comps:
            existing_cs_lengths[ps.comps[0].name] = ps_lengths[-1]
    new_comp_breakdown = []
    comp_i = 0
    ps_i = 0
    for comp in new_network.compressors.values():
        segment_name = ""
        comp_type = "Existing"
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

        elif comp.name in existing_cs_lengths.keys():
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
        pressure_breakdown.append((node.name, node.pressure))
    columns = ["Node", "Pressure (Pa-g)"]

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
    for pf_type in pf_inputs:
        for i, v in pf_inputs[pf_type].items():
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
) -> list:
    # Fill in levelized costs
    params_out = [("Parameter", "Value", "Units")]
    for i, v in price_breakdown.items():
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

    params_out.append(
        ("Added compressor stations", sum(mod_result.n_comps) + supply_comp, "")
    )
    params_out.append(("Added compressor capacity", comp_addl_rating, "hp"))
    params_out.append(
        (
            "Compressor fuel usage",
            mod_costing_params.all_fuel_MW * gl.MW2MMBTUDAY,
            "MMBTU/day",
        )
    )
    params_out.append(
        (
            "Compressor fuel usage",
            mod_costing_params.all_fuel_MW * gl.MW2MMBTUDAY / gl.DAY2HR,
            "MMBTU/hr",
        )
    )
    params_out.append(
        (
            "Compressor fuel usage (electric)",
            mod_costing_params.all_fuel_elec_kW,
            "kW",
        )
    )
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
    params_out.append(("New compressor stations", comp_cost_total, "$"))
    params_out.append(
        (
            "Compressor station refurbishment",
            sum(mod_costing_params.revamped_comp_capex),
            "$",
        )
    )
    params_out.append(
        ("Meter station modification", mod_costing_params.meter_cost, "$")
    )
    params_out.append(("Valve modifications", mod_costing_params.valve_cost, "$"))
    # params_out.append(("Original network residual value", 0, "$"))

    params_out.append(("Hydrogen energy ratio", blend_ratio_energy * 100, "%"))

    # --- Emissions outputs ---
    if emissions is not None:
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
    orig_pipes.pop("DSG")

    list_out = []
    list_out.append(list((orig_pipes.keys())))
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
            new_pipe_condensed["Labor Cost ($)"].append(
                i["Labor"] if "Labor" in i else 0
            )
            new_pipe_condensed["Misc ($)"].append(i["Misc"] if "Misc" in i else 0)
            new_pipe_condensed["ROW ($)"].append(i["ROW"] if "ROW" in i else 0)

    list_out = []
    list_out.append(list((new_pipe_condensed.keys())))
    for i, val in enumerate(new_pipe_condensed["Pipe Nominal Diameter"]):
        list_out.append([new_pipe_condensed[col][i] for col in new_pipe_condensed])

    return list_out
