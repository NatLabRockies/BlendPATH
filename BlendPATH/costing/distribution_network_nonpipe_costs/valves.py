import BlendPATH.costing.costing as bp_cost

DEFAULT_VALVE_DENSITY = 1  # 1 valve per km
DEFAULT_VALVE_COST = 2_000
DEFAULT_VALVE_REPLACEMENT_FRACTION = 0.075
DEFAULT_VALVE_REPLACEMENT_LIMIT = 0.25
import numpy as np


def get_valve_cost(
    total_pipe_length: float,
    costing_params: bp_cost.Costing_params,
    blend: float = 0,
) -> float:
    # No change if below blend limit for replacement
    if blend < costing_params.valve_blend_limit:
        return 0.0
    return (
        np.ceil(total_pipe_length * costing_params.valve_density)
        * costing_params.valve_cost_distribution
        * costing_params.valve_replacement_fraction
    )
