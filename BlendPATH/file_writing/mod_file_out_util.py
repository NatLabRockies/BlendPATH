import math

import xlsxwriter


def pipes_cols() -> list:
    return [
        "pipe_name",
        "from_node",
        "to_node",
        "diameter_mm",
        "length_km",
        "roughness_mm",
        "thickness_mm",
        "rating_code",
    ]


def nodes_cols() -> list:
    return ["node_name", "p_max_mpa_g"]


def comps_cols() -> list:
    return [
        "compressor_name",
        "from_node",
        "to_node",
        "pressure_out_mpa_g",
        "rating_MW",
        "extract_fuel",
        "eta_s",
        "eta_driver",
    ]


def regs_cols() -> list:
    return [
        "regulator_name",
        "from_node",
        "to_node",
        "pressure_out_mpa_g",
    ]


def supply_cols() -> list:
    return ["supply_name", "node_name", "pressure_mpa_g", "flowrate_MW", "blend"]


def demand_cols() -> list:
    return ["demand_name", "node_name", "flowrate_MW"]


def composition_cols() -> list:
    return ["SPECIES", "X"]


def write_dict_by_rows(wks: xlsxwriter.worksheet, my_dict: dict):
    row = 0
    columns = list(my_dict.keys())
    wks.write_row(row, 0, columns)
    row += 1
    for i in range(len(my_dict[columns[0]])):
        wks.write_row(row, 0, [_clean_val(my_dict[col][i]) for col in columns])
        row += 1


def _clean_val(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return ""
    return val


def write_to_network_file(
    filename: str,
    new_pipes: dict,
    new_nodes: dict,
    new_comps: dict,
    new_supply: dict,
    new_demand: dict,
    new_composition: dict,
    new_regs: dict = None,
    sort_comps: bool = False,
) -> None:
    with xlsxwriter.Workbook(filename) as workbook:
        wks = workbook.add_worksheet("PIPES")
        write_dict_by_rows(wks=wks, my_dict=new_pipes)

        wks = workbook.add_worksheet("SUPPLY")
        write_dict_by_rows(wks=wks, my_dict=new_supply)

        wks = workbook.add_worksheet("DEMAND")
        write_dict_by_rows(wks=wks, my_dict=new_demand)

        wks = workbook.add_worksheet("COMPOSITION")
        write_dict_by_rows(wks=wks, my_dict=new_composition)

        if new_regs:
            wks = workbook.add_worksheet("REGULATORS")
            write_dict_by_rows(wks=wks, my_dict=new_regs)

        wks = workbook.add_worksheet("NODES")
        write_dict_by_rows(wks=wks, my_dict=new_nodes)

        wks_comp = workbook.add_worksheet("COMPRESSORS")

        if sort_comps:
            row = 0
            columns = list(new_comps.keys())
            wks_comp.write_row(row, 0, columns)
            row += 1
            for node in new_nodes["node_name"]:
                try:
                    loc = new_comps["from_node"].index(node)

                    wks_comp.write_row(
                        row, 0, [_clean_val(new_comps[col][loc]) for col in columns]
                    )
                    row += 1

                except ValueError:
                    pass
        else:
            write_dict_by_rows(wks=wks_comp, my_dict=new_comps)
