"""
Module for determining odorization costs
"""

import BlendPATH.costing.costing as bp_cost

DEFAULT_ODORANT_COST = 3  # [$/lb]
DEFAULT_ODORANT_USE = 0.5  # [lb/mmscf]
H2_LEL = 0.04
CH4_LEL = 0.05


def get_odorization_cost(
    sm3_s: float, costing_params: bp_cost.Costing_params, blend: float
) -> float:
    """_summary_

    Args:
        sm3_s (float): Increase network in standard m3 per s
        costing_params (bp_cost.Costing_params): Csoting parameter in case there are overrides
        blend (float): Blend fraction of H2

    Returns:
        float: Annual increase in odorization cost
    """
    mmscf_year = sm3_s / 28316.846592 * 3600 * 24 * 365

    # linear scale between LELs [lb/mmscf]
    odorant_use_rate = DEFAULT_ODORANT_USE * CH4_LEL / H2_LEL * blend

    return odorant_use_rate * mmscf_year * costing_params.odorant_cost
