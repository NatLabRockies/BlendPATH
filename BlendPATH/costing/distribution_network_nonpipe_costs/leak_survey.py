import BlendPATH.Global as gl

DEFAULT_LEAK_SURVEY_COST_PER_KM = 560 * gl.KM2MI
DEFAULT_LEAK_SURVEY_FREQUENCY_YRS = 1


def get_leak_survey_cost(new_pipe_length: float, mod_type: str) -> float:
    """Calculates additional leak survey cost for new pipe lengths. Does not apply for DR (same length of pipe).

    Args:
        new_pipe_length (float): Length of new pipe [km]

    Returns:
        float: Annual leak survey cost [$/yr]
    """
    if mod_type == "dr":
        return 0.0
    return (
        DEFAULT_LEAK_SURVEY_COST_PER_KM
        * new_pipe_length
        / DEFAULT_LEAK_SURVEY_FREQUENCY_YRS
    )
