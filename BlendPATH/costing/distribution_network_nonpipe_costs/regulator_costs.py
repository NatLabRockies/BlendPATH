from typing import TYPE_CHECKING

import numpy as np

import BlendPATH.costing.costing as bp_cost
import BlendPATH.Global as gl

if TYPE_CHECKING:
    from BlendPATH.network.pipeline_components.regulator import Regulator
    from BlendPATH.network.pipeline_components.supply_node import Supply_node

SUPPLY_NODE_REGULATOR_COST = 400_000
DEFAULT_REG_REFURBISHENT_FRACTION = 0.3
DEFAULT_REG_MAINTENANCE_FRACTION = 0.0046
DEFAULT_BLEND_LIMIT = 0.1

# REG_PRESSURE_LIMITS_BAR = np.array([5, 16, 41, 80])
# REG_COST_PER_BIN = np.array([20_000, 25_000, 400_000, 3_000_000])


def get_regulator_refubishment_cost(
    supply_nodes: dict[str, "Supply_node"],
    blend: float,
    costing_params: bp_cost.Costing_params,
) -> float:
    if blend < costing_params.regulator_blend_limit:
        return 0

    return (
        len(supply_nodes)
        * costing_params.supply_regulator_cost
        * costing_params.regulator_refurbishment_fraction
    )


def get_regulator_maintenance_cost(
    n_regulators: int, costing_params: bp_cost.Costing_params
) -> float:
    return (
        n_regulators
        * costing_params.supply_regulator_cost
        * costing_params.regulator_maintenance_fraction
    )


def get_inline_regulator_refurbishment_cost(
    original_regulators: dict[str, "Regulator"],
    regulators: dict[str, "Regulator"],
    blend: float,
    costing_params: bp_cost.Costing_params,
) -> float:

    inlet_pressure_pa = np.array(
        [
            reg.from_node.pressure
            for reg in regulators.values()
            if check_blend_limit(
                original_regulators[reg.name].blend_limit,
                blend,
                costing_params.regulator_blend_limit,
            )
        ]
    )

    limits = np.array(list(costing_params.inline_regulator_cost.keys()))
    costs = np.array(list(costing_params.inline_regulator_cost.values()))
    # Clip to max length-1 because digitize will return len() if value is past upper bound
    return sum(
        costs[
            np.clip(
                np.digitize(
                    x=inlet_pressure_pa / gl.BAR2PA,
                    bins=limits,
                    right=True,
                ),
                0,
                len(limits) - 1,
            )
        ]
        * costing_params.regulator_refurbishment_fraction
    )


def check_blend_limit(
    reg_value: float | None, limit: float, default_value: float
) -> bool:

    return (reg_value if reg_value is not None else default_value) < limit


def get_inline_regulator_maintenance_cost(
    regulator_cost: float, costing_params: bp_cost.Costing_params
) -> float:
    return regulator_cost * costing_params.regulator_maintenance_fraction
