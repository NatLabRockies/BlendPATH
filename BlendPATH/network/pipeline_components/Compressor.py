import numpy as np

import BlendPATH.costing.costing as bp_cost
import BlendPATH.Global as gl

from .Node import Node


class Compressor:
    """
    Compressor component in pipeline network, raises pressure to outlet pressure
    """

    def __init__(
        self,
        from_node: Node,
        to_node: Node,
        name: str = "",
        pressure_out_mpa_g: float = 0,
        fuel_extract: bool = False,
        original_rating_MW: float = 0,
        eta_comp_s: float = 0.78,
        eta_comp_s_elec: float = 0.88,
        eta_driver: float = 0.357,
        eta_driver_elec: float = np.nan,
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.name = name
        self.pressure_out_mpa_g = pressure_out_mpa_g
        self.fuel_extract = fuel_extract
        self.original_rating_MW = original_rating_MW
        self.eta_comp_s = eta_comp_s
        self.eta_comp_s_elec = eta_comp_s_elec
        self.eta_driver = eta_driver
        self.eta_driver_elec = eta_driver_elec
        self.eta_driver_elec_calc = np.nan
        self.eta_driver_elec_used = np.nan

        self.to_node.pressure = self.pressure_out_mpa_g * gl.MPA2PA

    @property
    def compression_ratio(self) -> float:
        """
        Get compression ratio - to_node is high pressure than from_node
        """
        return self.to_node.pressure / self.from_node.pressure

    def get_power(self, h_1: float, s_1: float, h_2_s: float, m_dot: float) -> float:
        """
        Calculate power based isentropic efficiency
        """

        # Enthalpy change (isentropic)
        delta_h_s = h_2_s - h_1  # Adiabatic head
        # Enthalpy real
        delta_h = delta_h_s / self.eta_comp_s
        if not self.fuel_extract:  # Electrical isentropic effic
            delta_h = delta_h_s / self.eta_comp_s_elec

        # Get shaft work
        W_dot_shaft = max(0, m_dot * delta_h)
        self.shaft_power_MW = W_dot_shaft * gl.W2MW
        return W_dot_shaft

    def get_fuel_use(self, h_1: float, s_1: float, h_2_s: float, m_dot: float) -> float:
        """
        Calculate compressor fuel usage
        """
        # Leave this before return to calculate power
        W_dot_shaft = self.get_power(h_1, s_1, h_2_s, m_dot)
        self.flow_mdot = m_dot

        # But still requies calculating power
        self.fuel_mdot = 0
        self.fuel_electric_W = 0
        self.fuel_w = 0

        # if not fuel extraction, must be electrically driven
        fuel_electric_W = 0
        fuel_mdot_kg_s = 0
        driver_fuel_rate_W = 0
        if W_dot_shaft > 0:
            if not self.fuel_extract:
                if np.isnan(self.eta_driver_elec):
                    kW_shaft = W_dot_shaft / gl.KW2W
                    eta_coefs = [
                        0.7617,
                        0.0311,
                        0.0061,
                        -0.0015,
                        8e-5,
                    ]
                    eta_driver_elec = np.dot(
                        eta_coefs, [np.log(kW_shaft) ** x for x in range(5)]
                    )  # Nexant report
                    eta_driver_elec = np.minimum(eta_driver_elec, 1)
                    self.eta_driver_elec_calc = eta_driver_elec
                else:
                    eta_driver_elec = self.eta_driver_elec
                self.eta_driver_elec_used = eta_driver_elec
                fuel_electric_W = W_dot_shaft / eta_driver_elec
            else:
                # Assign mechanical efficiency
                driver_fuel_rate_W = W_dot_shaft / self.eta_driver
                # Calculate flow rate based on power need and HHV of gas
                fuel_mdot_kg_s = driver_fuel_rate_W / (
                    self.from_node.heating_value() * gl.MJ2J
                )

        self.fuel_electric_W = fuel_electric_W
        self.fuel_mdot = fuel_mdot_kg_s
        self.fuel_w = driver_fuel_rate_W

        return fuel_mdot_kg_s

    @property
    def fuel_use_MMBTU_hr(self) -> float:
        """
        Get fuel use in units of MMBTU/hr
        """
        return self.fuel_w * gl.W2MW * gl.MW2MMBTUDAY / gl.DAY2HR

    def get_cap_cost(self, cp, revamp: bool = False, to_electric: bool = True) -> float:
        """
        Calculate compressor capital cost
        """
        # Set min horsepower for single compressors
        hp_min = 3_000

        # Check if compresser is existing or new based on original rating != 0
        existing = True
        if self.original_rating_MW == 0:
            existing = False

        # Check if an existing gas compressor is being converted to electric
        convert = False
        if to_electric and existing and self.fuel_extract:
            convert = True

        # If new compressor no revamp cost required
        if not existing and revamp:
            self.revamp_cost = 0
            return 0

        # Determine additional rating relative to original rating
        ratio = 1
        self.addl_rating = (
            max(0, self.shaft_power_MW - self.original_rating_MW) * gl.MW2HP
        )
        cap_basis = self.addl_rating

        # If requesting revamp costs, only cost out revamping the original rating.
        # at same time check if it is being converted to electric. Same cost will apply
        if revamp and (existing or convert):
            ratio = 0.66
            avg_cs_cap = self.original_rating_MW * gl.MW2HP

            cap_basis = avg_cs_cap

        # If no additional rating, ando not checking revamp or convert to electric,
        # then current compressor rating is satisfactory, return 0 cost
        if self.addl_rating <= 0 and not revamp:
            self.cost = 0
            self.num_CS_new = 0
            self.avg_CS_cap_new = 0
            self.revamp_cost = 0
            return self.cost

        num_CS = 1
        avg_CS_cap = hp_min if cap_basis < hp_min else cap_basis / num_CS
        cost = bp_cost.get_compressor_cost(cp, avg_CS_cap, num_CS, ratio)

        # If the compressor is being converted to electric or is costed as electric
        # Then apply 1.3 markup based on Rui 2011
        if convert or not self.fuel_extract:
            electric_markup = 1.3  # Rui 2011
            cost *= electric_markup

        if not revamp:
            self.num_CS_new = num_CS
            self.avg_CS_cap_new = avg_CS_cap
            self.cost = cost
        else:
            self.revamp_cost = cost

        return cost
