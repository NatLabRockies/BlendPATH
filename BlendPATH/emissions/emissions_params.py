from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmissionsParams:
    """
    Emissions assumptions for BlendPATH emissions intensity calculations.
      - EF_*_gCO2_per_MJ : gCO2 per MJ of energy
      - EF_FUG_* : gCO2e per gram leaked species
      - leak_trans_CH4_m3_per_m3_per_km : volumetric leak coefficient
      - leak_dist_CH4_m3_per_m3         : distribution volumetric leak fraction [m^3 CH4 / m^3 NG]
    """

    # Energy-based emission factors
    EF_NG_COMB_gCO2_per_MJ: float = 51.0
    EF_NG_UPSTREAM_gCO2_per_MJ: float = 8.45
    EF_H2_UPSTREAM_gCO2_per_MJ: float = 0.0
    EF_ELEC_gCO2_per_MJ: float = 120.0

    # Fugitive “CO2e multipliers”
    EF_FUG_NG_gCO2e_per_gCH4: float = 30.0
    EF_FUG_H2_gCO2e_per_gH2: float = 8.0

    # Leakage model coefficient
    leak_trans_CH4_m3_per_m3_per_km: float = 3.33e-6
    leak_dist_CH4_m3_per_m3: float = 1.05e-3

    # Scope switch
    include_combustion: bool = True
