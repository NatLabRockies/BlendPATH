"""
Erosion rate based on DNV RP501
"""

import math
from dataclasses import dataclass

import BlendPATH.Global as gl


@dataclass
class MaterialProp:
    k: float
    n: float
    rho: float


# Steel: DNV RP O501 - Revision 4.2 - 2007 Table 7-2
# Polymer estimates from Solid-particle erosion behavior of high-performance thermoplastic polymers; Suresh Arjula Æ A. P. Harsha Æ M. K. Ghosh
MATERIAL_PROPS: dict[str, MaterialProp] = {
    "steel": MaterialProp(k=2.0e-9, n=2.6, rho=7800),
    "polymer": MaterialProp(k=1e-9, n=2.3, rho=1000),
}


def erosion_rate_velocity(
    diameter_mm: float,
    thickness_mm: float,
    fraction_of_thickness: float,
    years: float,
    rho_g: float,
    material: str,
    loading_ppm_w: float = 1.0,
) -> float:
    """Calculate velocity that would correlate with erosion rate

    Args:
        diameter_mm (float): Inner diameter of pipe [mm]
        thickness_mm (float): Thickness of pipe [mm]
        fraction_of_thickness (float): Fraction of thickness allowable
        years (float): Lifetime of operation of pipe [yrs]
        rho_g (float): Density of gas [kg/m3]
        material (str): Material of pipe "steel" or "polymer"

    Returns:
        float: Velocity [m/s]
    """
    E_L_mm_year = thickness_mm * fraction_of_thickness / years
    D_m = diameter_mm * gl.MM2M
    # TODO: C should update for polymer pipes
    C = 0.000_025
    # PPM_weight to kg/kg
    loading = loading_ppm_w * 1e-6
    n = MATERIAL_PROPS[material].n
    A = math.pi * D_m**2 / 4

    return (E_L_mm_year * D_m**2 / C / loading / rho_g / A) ** (1 / (n + 1))


def erosion_rate(
    diameter_mm: float,
    length_km: float,
    mixture_kg_s: float,
    rho_g: float,
    material: str = "steel",
    alpha: float = 30,
) -> float:
    """Calculate erosion rate

    Args:
        diameter_mm (float): Inner diameter [mm]
        length_km (float): Pipe length [km]
        mixture_kg_s (float): Mixture flow rate [kg/s]
        rho_g (float): Gas density [kg/m3]
        material (str, optional): Pipe material, either 'steel' or 'polymer'. Defaults to "steel".
        alpha (float, optional): Impact angle [deg]. Defaults to 30.

    Returns:
        float: Erosion rate [mm/year]
    """
    # Ductility
    F = get_F_ductility(alpha=alpha)
    # Material constant
    K = MATERIAL_PROPS[material].k
    # Velocity exponent
    n = MATERIAL_PROPS[material].n
    # Pipe density
    rho = MATERIAL_PROPS[material].rho
    # Unit conversion m/s -> mm/year
    C_UNIT = gl.S2HR * gl.HR2YR / gl.MM2M
    # Area exposed to erosion (pipe internal surface area)
    A = math.pi * (diameter_mm * gl.MM2M) * (length_km * gl.KM2M)
    # Particle flow rate
    m_p = get_particle_flow_kg_s(mixture_kg_s=mixture_kg_s)
    # Particle impact velocity
    U_p = get_impact_velocity(
        diameter_mm=diameter_mm, rho_g=rho_g, mixture_kg_s=mixture_kg_s
    )

    return m_p * K * U_p**n * F / rho / A * C_UNIT


def get_F_ductility(alpha: float) -> float:
    """Get F, which characterizes the ductility of the target material. Ductile materials attain maximum erosion attacks for impact angles in the range 15 - 30deg


    Args:
        alpha (float): Impact angle in degrees

    Raises:
        ValueError: Alpha must be between 0 and 90 degrees

    Returns:
        float: F value
    """
    if not 0 <= alpha <= 90:
        raise ValueError(f"alpha must be between 0 and 90, {alpha} was provided")
    coef = [9.37, 42.295, 110.864, 175.804, 170.137, 98.398, 31.211, 4.17]
    return sum(
        [
            coef[i - 1] * (alpha * math.pi / 180) ** i * (-1) ** (i + 1)
            for i in range(1, 9)
        ]
    )


def get_particle_flow_kg_s(mixture_kg_s: float) -> float:
    """Calculate particle mass flow rate

    Args:
        mixture_kg_s (float): Mass flow rate of mixture (gas+particles)

    Returns:
        float: Particle flow rate
    """
    # Particle ppm weight basis
    # TODO: This may be a output
    loading_ppmW = 1
    return mixture_kg_s * loading_ppmW / 1_000_000


def get_impact_velocity(diameter_mm: float, rho_g: float, mixture_kg_s: float) -> float:
    """Calculate impact velocity

    Args:
        diameter_mm (float): Inner pipe diameter [mm]
        rho_g (float): Gas density [kg/m3]
        mixture_kg_s (float): Mixture flow rate [kg/s]

    Returns:
        float: Impact velocity [m/s]
    """
    diameter_m = diameter_mm * gl.MM2M
    return 4 * mixture_kg_s / diameter_m**2 / math.pi / rho_g
