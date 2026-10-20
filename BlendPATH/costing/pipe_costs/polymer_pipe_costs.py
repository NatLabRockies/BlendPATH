from typing import Callable


def polymer_pipe_cost_socalgas(nps_diam_in: float) -> float:
    """
    https://www.socalgas.com/regulatory/documents/a-15-07-014/ORA-07.docx report curve fit
    Data exists for 0.5 to 8in
    In 2024$
    """
    return 17.938 * nps_diam_in + 91.077


def polymer_pipe_cost_nexant_rural(nps_diam_in: float) -> float:
    """
    https://www1.eere.energy.gov/hydrogenandfuelcells/pdfs/nexant_h2a.pdf
    In 2024$
    """
    return 2.283 * nps_diam_in + 9.686


def polymer_pipe_cost_nexant_suburban(nps_diam_in: float) -> float:
    """
    https://www1.eere.energy.gov/hydrogenandfuelcells/pdfs/nexant_h2a.pdf
    In 2024$
    """
    return 2.767 * nps_diam_in + 11.070


def polymer_pipe_cost_nexant_urban(nps_diam_in: float) -> float:
    """
    https://www1.eere.energy.gov/hydrogenandfuelcells/pdfs/nexant_h2a.pdf
    In 2024$
    """
    return 4.843 * nps_diam_in + 20.756


def polymer_pipe_cost_nexant_downtown(nps_diam_in: float) -> float:
    """
    https://www1.eere.energy.gov/hydrogenandfuelcells/pdfs/nexant_h2a.pdf
    In 2024$
    """
    return 27.675 * nps_diam_in + 117.617


polymer_pipe_fxns: dict[str, Callable] = {
    "socal": polymer_pipe_cost_socalgas,
    "nexant_rural": polymer_pipe_cost_nexant_rural,
    "nexant_suburban": polymer_pipe_cost_nexant_suburban,
    "nexant_urban": polymer_pipe_cost_nexant_urban,
    "nexant_downtown": polymer_pipe_cost_nexant_downtown,
}


def get_polymer_pipe_cost_per_ft(cost_type: str, nps_diam_in: float) -> float:
    """
    Handler for polymer pipe cost types
    """
    if cost_type not in polymer_pipe_fxns:
        raise ValueError(f"Cost type of {cost_type} not found")
    if nps_diam_in <= 0:
        raise ValueError(f"Nominal diameter of {nps_diam_in} inches is not valid")
    return polymer_pipe_fxns[cost_type](nps_diam_in=nps_diam_in)
