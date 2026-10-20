"""
Bundled costs for distribution network operations expenses
Based on AGA benchmarking Performance bnechmarks for natural gas utilities
https://www.aga.org/research-policy/resource-library/performance-benchmarks-for-natural-gas-utilities/
"""

from typing import Literal

import BlendPATH.Global as gl

OM_COST_TYPES = Literal["AGA_bundled"]


def aga_operations() -> float:
    """$/mi operations expenses for distribution. 2016-2023 average. Items 870-881

    Returns:
        flaot: $/mi value
    """
    return 45_000


def aga_maintenance() -> float:
    """$/mi maintenance expenses for distribution. 2016-2023 average. Items 885-894

    Returns:
        float: $/mi value
    """
    return 2_700


def get_o_and_m_cost(cost_type: OM_COST_TYPES, pipe_km: float) -> tuple:
    """Calculate the annual operations and maintenance cost

    Args:
        cost_type (OM_COST_TYPES): Cost type selector
        pipe_km (float): Length of pipe [km]

    Raises:
        RuntimeError: If unknown cost type

    Returns:
        float: Annual O&M cost
    """
    if cost_type == "AGA_bundled":
        pipe_mi = pipe_km * gl.KM2MI
        return aga_operations() * pipe_mi, aga_maintenance() * pipe_mi
    else:
        raise RuntimeError(f"Unknown cost type: {cost_type}")
