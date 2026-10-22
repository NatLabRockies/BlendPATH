import csv
import importlib.resources as pkg_res
from pathlib import Path
from typing import Final, Union

import BlendPATH.Global as gl
from BlendPATH.util.schedules import POLYMER_SDR_TABLE

_HDB_COLS: Final[str] = [
    "polymer",  # "PE" / "PA" / …
    "designation",  # e.g. 4710
    "hdb_psi_73F",  # Hydrostatic-Design Basis @ 73 °F [psi]
]
_HDB_FILENAME: str = "pe_hdb_table.csv"


def _load_hdb_table() -> dict:
    """
    Load in HDB table from csv with polymer type and designation
    """
    csv_path: Path = pkg_res.files("BlendPATH.util").joinpath(_HDB_FILENAME)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find {csv_path}")

    hbd_table = {"PE": {}, "PA": {}}

    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        # Check required fieldnames are included
        for fn in reader.fieldnames:
            if fn not in _HDB_COLS:
                raise ValueError(f"{_HDB_FILENAME} is missing column {fn}")

        for row in reader:
            if row["polymer"] not in hbd_table:
                raise ValueError(f"{row['polymer']} is not a valid polymer type")
            hbd_table[row["polymer"]][row["designation"]] = float(row["hdb_psi_73F"])

    return hbd_table


_HDB_TBL: dict = _load_hdb_table()


def get_hdb_psi(polymer: str, designation: Union[str, int]) -> float:
    """
    Return the HDB [psi] for a given polymer & designation.
    Examples
    --------
    >>> get_hdb_psi("PE", 4710)   # → 1600
    """
    polymer = polymer.upper().strip()
    design = str(designation).strip()

    try:
        _HDB_TBL[polymer][design]
    except KeyError:
        raise ValueError(f"Unsupported rating code {polymer} {design}")

    return _HDB_TBL[polymer][design]


def list_polymer_designations() -> dict:
    """Return a copy of the full polymer–designation–HDB table."""
    return _HDB_TBL


def design_pressure_thermoplastic(
    design_factor: float, sdr: float, polymer: str, polymer_designation: str
) -> float:
    """
    Return allowable gauge pressure [MPa] for PE/PA pipe using
    ASTM D2513 Annex X1.
    """
    hdb_psi = get_hdb_psi(polymer, polymer_designation)
    P_psi = 2.0 * hdb_psi * design_factor / (sdr - 1.0)
    return P_psi * gl.PSI2MPA


def get_sdr(od_mm: float, th_mm: float) -> float:
    key = (round(od_mm, 2), round(th_mm, 2))
    if key in POLYMER_SDR_TABLE and POLYMER_SDR_TABLE[key] is not None:
        return POLYMER_SDR_TABLE[key]
    return round(od_mm / th_mm, 2)
