from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from python_calamine import CalamineWorkbook


class SheetName:
    PIPES = "PIPES"
    NODES = "NODES"
    COMPRESSORS = "COMPRESSORS"
    REGULATORS = "REGULATORS"
    SUPPLY = "SUPPLY"
    DEMAND = "DEMAND"
    COMPOSITION = "COMPOSITION"


@dataclass
class col_entry:
    default: Any = None
    formatting: list[Callable] = field(default_factory=list)


def cast_float(val):
    return float(val)


def cast_bool(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "yes", "y", "t"]


def str_uppercase(val):
    return val.upper()


def check_0_1(val):
    if not 0 < val <= 1:
        raise ValueError("Must be between 0 and 1")
    return val


_VALID_COLS = {
    SheetName.PIPES: {
        "pipe_name": col_entry(),
        "from_node": col_entry(),
        "to_node": col_entry(),
        "length_km": col_entry(formatting=[cast_float]),
        "roughness_mm": col_entry(formatting=[cast_float]),
        "diameter_mm": col_entry(formatting=[cast_float]),
        "rating_code": col_entry(formatting=[str_uppercase]),
        "thickness_mm": col_entry(formatting=[cast_float]),
    },
    SheetName.NODES: {
        "node_name": col_entry(),
        "p_max_mpa_g": col_entry(formatting=[cast_float]),
        "plot_x": col_entry(),
        "plot_y": col_entry(),
    },
    SheetName.COMPRESSORS: {
        "compressor_name": col_entry(),
        "from_node": col_entry(),
        "to_node": col_entry(),
        "pressure_out_mpa_g": col_entry(formatting=[cast_float]),
        "rating_MW": col_entry(formatting=[cast_float]),
        "extract_fuel": col_entry(formatting=[cast_bool]),
        "eta_s": col_entry(formatting=[cast_float, check_0_1]),
        "eta_driver": col_entry(formatting=[cast_float, check_0_1]),
    },
    SheetName.REGULATORS: {
        "regulator_name": col_entry(),
        "from_node": col_entry(),
        "to_node": col_entry(),
        "pressure_out_mpa_g": col_entry(formatting=[cast_float]),
    },
    SheetName.SUPPLY: {
        "supply_name": col_entry(),
        "node_name": col_entry(),
        "pressure_mpa_g": col_entry(),
        "flowrate_MW": col_entry(),
        "blend": col_entry(default=0.0, formatting=[cast_float]),
    },
    SheetName.DEMAND: {
        "demand_name": col_entry(),
        "node_name": col_entry(),
        "flowrate_MW": col_entry(default=0.0, formatting=[cast_float]),
        "min_pressure_mpa_g": col_entry(default=0.0, formatting=[cast_float]),
    },
    SheetName.COMPOSITION: {
        "SPECIES": col_entry(),
        "X": col_entry(formatting=[cast_float]),
    },
}


def read_workbook(filename: Path):
    workbook = CalamineWorkbook.from_path(filename)
    return {sheet_name: read_sheet(workbook, sheet_name) for sheet_name in _VALID_COLS}


def read_sheet(
    workbook,
    sheetname: str,
):
    output = {c: [] for c in _VALID_COLS[sheetname]}
    if sheetname not in workbook.sheet_names:
        return output
    rows = iter(workbook.get_sheet_by_name(sheetname).to_python())
    # read the header row
    col_order = get_header_order(row=next(rows), sheetname=sheetname)
    for row in rows:
        for col in _VALID_COLS[sheetname]:
            if col in col_order:
                if row[col_order[col]] == "":
                    output[col].append(_VALID_COLS[sheetname][col].default)
                    continue
                value = row[col_order[col]]
                for formatting_fxn in _VALID_COLS[sheetname][col].formatting:
                    value = formatting_fxn(value)
                output[col].append(value)
            else:
                output[col].append(_VALID_COLS[sheetname][col].default)
    return output


def get_header_order(row, sheetname):
    return {col: i for i, col in enumerate(row) if col in _VALID_COLS[sheetname]}
