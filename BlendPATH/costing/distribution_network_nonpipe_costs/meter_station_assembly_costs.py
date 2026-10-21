"""
Costs for meter station assemblies
Based on https://www.socalgas.com/sites/default/files/Chapter-10.pdf
Total MSA Costs (Dollars) (F)
"""

DEFAULT_MSA_BLEND_LIMIT = 0.20

import numpy as np
from numpy import typing as npt

MSA_COST_BINS = {
    "medium_pressure_lt_60psig": {
        "bins": np.asarray([0, 276, 426, 631, 801, 1_101, 1_501, 2_001, 3_001, 5_001]),
        "costs": np.asarray(
            [
                280.7,
                700.73,
                1_090.64,
                1_480.72,
                1_567.97,
                2_924.91,
                4_526.11,
                4_398.32,
                5_340.12,
                5_656.72,
            ]
        ),
    },
    "high_pressure_gt_60psig": {
        "bins": np.asarray(
            [
                0,
                941,
                1_051,
                1_501,
                2_701,
                4_001,
                6_601,
                9_201,
                14_501,
                21_401,
                24_001,
                46_001,
                79_001,
            ]
        ),
        "costs": np.asarray(
            [
                2_526.12,
                4_847.83,
                2_916.03,
                5_019.87,
                4_892.07,
                6_559.03,
                7_164.29,
                8_001.93,
                8_141.24,
                16_152.84,
                17_129.15,
                29_165.57,
                53_076.55,
            ]
        ),
    },
}


def get_msa_cost(
    flowrate_cfh: npt.NDArray[np.float64],
    pressure_psig: npt.NDArray[np.float64],
) -> float:
    """_summary_

    Args:
        flowrate_cfh (npt.NDArray[np.float64]): Demand node flowrates in cubic feet per hour
        pressure_psig (npt.NDArray[np.float64]): Demand node pressures in psi_g

    Returns:
        npt.NDArray[np.float64]: Total meter set assembly cost
    """

    costs = np.zeros_like(flowrate_cfh, dtype=np.float64)

    pressure_categories = {
        "med": {
            "label": "medium_pressure_lt_60psig",
            "mask": pressure_psig <= 60,
        },
        "high": {
            "label": "high_pressure_gt_60psig",
            "mask": pressure_psig > 60,
        },
    }
    for p_cat in pressure_categories.values():
        costs[p_cat["mask"]] = MSA_COST_BINS[p_cat["label"]]["costs"][
            np.digitize(
                flowrate_cfh[p_cat["mask"]],
                MSA_COST_BINS[p_cat["label"]]["bins"],
            )
            - 1
        ]
    return sum(costs)
