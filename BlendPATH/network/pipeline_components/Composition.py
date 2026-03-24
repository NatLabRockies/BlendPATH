import logging

import cantera as ct
import numpy as np
import pandas as pd
import scipy.interpolate

import BlendPATH.Global as gl
from BlendPATH.network.pipeline_components import eos
from BlendPATH.scenario_helper import SCENARIO_VALUES, Scenario_type

from . import cantera_util as ctu

logger = logging.getLogger(__name__)

_t_k = 273.15
_CRIT_C = {
    "H2": (-239.95 + _t_k, 1297000),
    "CH4": (-82.595 + _t_k, 4598800),
    "C2H6": (32.68 + _t_k, 4880000),
    "C3H8": (96.67 + _t_k, 4250000),
    "CO2": (31.05 + _t_k, 7386000),
    "N2": (-146.95 + _t_k, 3390000),
    "C4H10": (151.99 + _t_k, 3784000),
    "C5H12": (196.54 + _t_k, 3364000),
    "iC4H10": (134.98 + _t_k, 3648000),
}

_VALID_SPECIES = ["H2", "CH4", "C2H6", "C3H8", "CO2", "N2", "C4H10", "C5H12", "iC4H10"]


class Composition:
    """
    Gas-phase composition
    """

    def __init__(
        self,
        pure_x: dict | None = None,
        composition_tracking: bool = False,
        min_pres: float = SCENARIO_VALUES[Scenario_type.TRANSMISSION].min_pressure_pa,
        max_pres: float = SCENARIO_VALUES[Scenario_type.TRANSMISSION].max_pressure_pa,
        comp_interps_status: tuple = (None, None),
        thermo_curvefit: bool = True,
        eos_type: eos._EOS_OPTIONS = "rk",
        blend: float | None = None,
    ):
        self.pure_x = pure_x if pure_x is not None else {}
        self.composition_tracking = composition_tracking
        self.min_pres = min_pres
        self.max_pres = max_pres
        self.comp_interps_status = comp_interps_status
        self.thermo_curvefit = thermo_curvefit
        self.eos_type = eos_type

        self.x = self.pure_x.copy()
        self.x["H2"] = self.x.get("H2", 0.0)

        self.blendH2(blend)

    @property
    def x_no_h2(self):
        x_no_h2 = self.pure_x.copy()
        x_no_h2["H2"] = 0
        norm = sum(x_no_h2.values())
        if norm == 0:
            return self.pure_x
        x_no_h2 = {x: v / norm for x, v in x_no_h2.items()}
        return x_no_h2

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        min_pres: float = SCENARIO_VALUES[Scenario_type.TRANSMISSION].min_pressure_pa,
        max_pres: float = SCENARIO_VALUES[Scenario_type.TRANSMISSION].max_pressure_pa,
        composition_tracking: bool = False,
        thermo_curvefit: bool = True,
        eos: eos._EOS_OPTIONS = None,
    ):
        """
        Create composition from a pandas df
        """
        species = df["SPECIES"].values.tolist()
        mole_frac = df["X"].values.tolist()

        composition = dict(zip(species, mole_frac))

        return Composition(
            pure_x=composition,
            min_pres=min_pres,
            max_pres=max_pres,
            composition_tracking=composition_tracking,
            thermo_curvefit=thermo_curvefit,
            eos_type=eos,
        )

    def blendH2(self, blend: float) -> None:
        """
        Update composition based on a new H2 mole fraction. Note that this requires no H2 in the original composition
        """
        # Return early if blend is already set
        if blend is None:
            blend = self.x["H2"]
        elif (
            hasattr(self, "curve_fit_hhv") and "H2" in self.x and self.x["H2"] == blend
        ):
            return
        if not (0 <= blend <= 1):
            raise ValueError(
                "Blend percent must be represented as a fraction between 0 and 1"
            )
        blend = float(blend)
        self.x = {a: b * (1 - blend) for a, b in self.x_no_h2.items() if a != "H2"}
        self.x["H2"] = blend
        # self.get_comp()
        # self.calc_heating_value()
        self.make_interps(min_pres=self.min_pres, max_pres=self.max_pres)

    def just_fuel(self, x_dict: dict = None) -> str:
        """
        Get the fuel components of the gas, used in HHV calculation
        """
        if x_dict is None:
            x = self.x
        else:
            x = x_dict
        not_fuels = ["CO2", "N2"]
        fuels_x = {x: v for x, v in x.items() if x not in not_fuels}
        new_total = sum(fuels_x.values())
        fuels_x = {x: v / new_total for x, v in fuels_x.items()}
        return fuels_x

    @property
    def x_str(self) -> str:
        """
        Assigns str value for input into Cantera
        """
        return str(self.x).strip("{}").replace(" ", "").replace("'", "")

    @property
    def tc(self) -> float:
        return sum([_CRIT_C[i][0] * v for i, v in self.x.items()])

    @property
    def pc(self) -> float:
        return sum([_CRIT_C[i][1] * v for i, v in self.x.items()])

    def calc_HHV(self, x_dict: dict = None) -> float:
        logger.debug(f"Calculating HHV for composition: {x_dict}")
        ctu.gas.TPX = 298.15, ct.one_atm, x_dict
        mole_frac_fuel = 1 - ctu.gas["N2"].Y[0] - ctu.gas["CO2"].Y[0]
        jf_X = self.just_fuel(x_dict)
        ctu.gas.set_equivalence_ratio(1.0, jf_X, "O2:1.0")
        h1 = ctu.gas.enthalpy_mass
        Y_fuel = 1 - ctu.gas["O2"].Y[0]
        X_products = {
            "CO2": ctu.gas.elemental_mole_fraction("C"),
            "H2O": 0.5 * ctu.gas.elemental_mole_fraction("H"),
            "N2": 0.5 * ctu.gas.elemental_mole_fraction("N"),
        }
        ctu.gas.TPX = None, None, X_products
        Y_H2O = ctu.gas["H2O"].Y[0]
        h2 = ctu.gas.enthalpy_mass
        HHV = -(h2 - h1 + ctu.h_water * Y_H2O) / Y_fuel / gl.MJ2J * mole_frac_fuel
        return HHV

    def make_2d_interp(self, min_pres, max_pres):
        logger.info(
            f"Running interpolations (blend={self.x['H2']}) between {min_pres} and {max_pres} Pa"
        )
        p_val_len = 50
        p_vals = np.linspace(min_pres, max_pres, p_val_len)
        rho_vals = np.zeros(len(p_vals))
        mu_vals = np.zeros(len(p_vals))
        cpcv_vals = np.zeros(len(p_vals))

        p_vals_final = np.linspace(min_pres, max_pres, p_val_len)

        self.curve_fit_mw = eos.get_props(
            input_type="TPX",
            input_vals=(gl.T_FIXED, p_vals[-1] + ct.one_atm, self.x),
            eos_type=self.eos_type,
            properties=("mean_molecular_weight",),
            set_state_flag=True,
        )
        self.curve_fit_hhv = self.calc_HHV(self.x)

        for p_i, p in enumerate(p_vals):
            p_a = p + ct.one_atm

            p_a = eos.set_state(
                input_type="TPX",
                input_vals=(gl.T_FIXED, p_a, self.x),
                eos_type=self.eos_type,
            )

            p_vals_final[p_i] = p_a

            rho_vals[p_i], mu_vals[p_i], cp, cv = eos.get_props(
                input_type="TPX",
                input_vals=(gl.T_FIXED, p_a, self.x),
                eos_type=self.eos_type,
                properties=("density", "viscosity", "cp_mass", "cv_mass"),
                set_state_flag=False,
            )
            cpcv_vals[p_i] = cp / cv

        self.curve_fit_rho = (p_vals_final, np.array(rho_vals))
        self.curve_fit_mu = (p_vals_final, np.array(mu_vals))
        self.curve_fit_cpcv = (p_vals_final, np.array(cpcv_vals))

    def get_curvefit_rho_z(
        self, p_gauge_pa: np.ndarray, x: np.ndarray, mw: np.ndarray = None
    ) -> np.ndarray:
        if np.any(p_gauge_pa > self.max_pres * gl.MPA2PA):
            raise RuntimeError(f"Pressure {p_gauge_pa} is outside of curvefit bounds")
        if self.composition_tracking:
            return self.get_curvefit_rho_z_3d(x=x, p_gauge_pa=p_gauge_pa, mw=mw)
        else:
            return self.get_curvefit_rho_z_2d(p_gauge_pa=p_gauge_pa, mw=mw)

    def get_curvefit_rho_z_2d(
        self, p_gauge_pa: np.ndarray, mw: np.ndarray
    ) -> np.ndarray:
        p_abs_pa = p_gauge_pa + ct.one_atm

        rho = np.interp(p_abs_pa, self.curve_fit_rho[0], self.curve_fit_rho[1])
        # mw = self.get_curvefit_mw(p_gauge_pa)
        z = p_abs_pa * mw / ctu.R_GAS / gl.T_FIXED / rho
        if np.any(rho < 0):
            raise ValueError("Negative pressure")
        return rho, z

    def get_curvefit_rho_z_3d(
        self, x: np.ndarray, p_gauge_pa: np.ndarray, mw: np.ndarray = None
    ) -> np.ndarray:
        p_abs_pa = p_gauge_pa + ct.one_atm

        if mw is None:
            mw = self.get_curvefit_mw(x)

        rho = self.curve_fit_rho.ev(x, p_abs_pa)

        # Calculate z
        z = p_abs_pa * mw / ctu.R_GAS / gl.T_FIXED / rho

        if np.any(rho < 0):
            raise ValueError(
                "Negative pressure (rho < 0) encountered in get_curvefit_rho_z"
            )

        return rho, z

    def get_curvefit_mu(self, p_gauge_pa: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.composition_tracking:
            return self.get_curvefit_mu_3d(x=x, p_gauge_pa=p_gauge_pa)
        else:
            return self.get_curvefit_mu_2d(p_gauge_pa=p_gauge_pa)

    def get_curvefit_mu_2d(self, p_gauge_pa: np.ndarray) -> np.ndarray:
        mu = np.interp(
            p_gauge_pa + ct.one_atm, self.curve_fit_mu[0], self.curve_fit_mu[1]
        )
        return mu

    def get_curvefit_mu_3d(self, x: np.ndarray, p_gauge_pa: np.ndarray) -> np.ndarray:
        return self.curve_fit_mu.ev(x, p_gauge_pa + ct.one_atm)

    def get_curvefit_h_2d(self, p_gauge_pa: np.ndarray) -> np.ndarray:
        return np.interp(
            p_gauge_pa + ct.one_atm, self.curve_fit_h[0], self.curve_fit_h[1]
        )

    def get_curvefit_h_3d(self, x: np.ndarray, p_gauge_pa: np.ndarray) -> np.ndarray:
        return self.curve_fit_h.ev(x, p_gauge_pa + ct.one_atm)

    def get_curvefit_s(self, p_gauge_pa: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.composition_tracking:
            return self.get_curvefit_s_3d(x=x, p_gauge_pa=p_gauge_pa)
        else:
            return self.get_curvefit_s_2d(p_gauge_pa=p_gauge_pa)

    def get_curvefit_s_2d(self, p_gauge_pa: np.ndarray) -> np.ndarray:
        return np.interp(
            p_gauge_pa + ct.one_atm, self.curve_fit_s[0], self.curve_fit_s[1]
        )

    def get_curvefit_s_3d(self, x: np.ndarray, p_gauge_pa: np.ndarray) -> np.ndarray:
        return self.curve_fit_s.ev(x, p_gauge_pa + ct.one_atm)

    def get_curvefit_hs(
        self, p_gauge_pa: np.ndarray, x: np.ndarray, s: np.ndarray
    ) -> np.ndarray:
        if self.composition_tracking:
            return self.get_curvefit_hs_3d(x=x, p_gauge_pa=p_gauge_pa, s=s)
        else:
            return self.get_curvefit_hs_2d(p_gauge_pa=p_gauge_pa, s=s)

    def get_curvefit_hs_2d(self, p_gauge_pa: np.ndarray, s: np.ndarray) -> np.ndarray:
        return self.curve_fit_h_2d.ev(p_gauge_pa + ct.one_atm, s)

    def get_curvefit_hs_3d(
        self, x: np.ndarray, p_gauge_pa: np.ndarray, s: np.ndarray
    ) -> np.ndarray:
        return self.curve_fit_h_3d((x, p_gauge_pa + ct.one_atm, s))

    def get_curvefit_mw(self, x: np.ndarray) -> np.ndarray:
        if self.composition_tracking:
            return self.curve_fit_mw(x)
        else:
            return np.full_like(x, self.curve_fit_mw)

    def get_curvefit_hhv(self, x: np.ndarray) -> np.ndarray:
        if self.composition_tracking:
            return self.curve_fit_hhv(x)
        else:
            return self.curve_fit_hhv

    def get_hhv(self, x: np.ndarray):
        if self.thermo_curvefit:
            return self.get_curvefit_hhv(x=x)
        else:
            return self.calc_HHV(x_dict=self.x)

    def get_mu(self, p_gauge_pa: np.ndarray, x: np.ndarray):
        if self.thermo_curvefit:
            return self.get_curvefit_mu(p_gauge_pa=p_gauge_pa, x=x)
        else:
            return eos.get_props(
                "TPX",
                (gl.T_FIXED, p_gauge_pa + ct.one_atm, self.x),
                self.eos_type,
                properties=["viscosity"],
            )

    def get_mw(self, p_gauge_pa: np.ndarray, x: np.ndarray):
        if self.thermo_curvefit:
            return self.get_curvefit_mw(x=x)
        else:
            return eos.get_props(
                "TPX",
                (gl.T_FIXED, p_gauge_pa + ct.one_atm, self.x),
                self.eos_type,
                properties=["mean_molecular_weight"],
            )

    def get_rho_z(self, p_gauge_pa: np.ndarray, x: np.ndarray, mw: np.ndarray):
        if self.thermo_curvefit:
            return self.get_curvefit_rho_z(p_gauge_pa=p_gauge_pa, x=x, mw=mw)
        else:
            return eos.get_rz(
                p_gauge=p_gauge_pa,
                T_K=gl.T_FIXED,
                X=self,
                eos=self.eos_type,
                mw=mw,
            )

    def get_cpcv(self, p_gauge_pa: np.ndarray, x: np.ndarray):
        if self.thermo_curvefit:
            if self.composition_tracking:
                return self.get_curvefit_cpcv_3d(x=x, p_gauge_pa=p_gauge_pa)
            else:
                return self.get_curvefit_cpcv_2d(p_gauge_pa=p_gauge_pa)
        else:
            cp, cv = eos.get_props(
                "TPX",
                (gl.T_FIXED, p_gauge_pa + ct.one_atm, self.x),
                self.eos_type,
                properties=["cp_mass", "cv_mass"],
            )
            return cp / cv

    def get_curvefit_cpcv_2d(self, p_gauge_pa: np.ndarray) -> np.ndarray:
        return np.interp(
            p_gauge_pa + ct.one_atm, self.curve_fit_cpcv[0], self.curve_fit_cpcv[1]
        )

    def get_curvefit_cpcv_3d(self, x: np.ndarray, p_gauge_pa: np.ndarray) -> np.ndarray:
        return self.curve_fit_cpcv.ev(x, p_gauge_pa + ct.one_atm)

    def make_interps(self, min_pres, max_pres) -> None:
        # Check if curvefits are needed
        if not self.thermo_curvefit:
            return

        if self.eos_type == "papay":
            raise RuntimeError("Papay EOS is not valid with curvefits")

        # Check if interpolation was already made for this combo
        if self.comp_interps_status[0] and hasattr(self, "curve_fit_rho"):
            return

        if self.comp_interps_status[1] == self.x.get("H2", 0):
            return

        if self.composition_tracking:
            self.make_3d_interp(min_pres, max_pres)
        else:
            self.make_2d_interp(min_pres, max_pres)
        self.comp_interps_status = (self.composition_tracking, self.x.get("H2", 0))

    def make_3d_interp(self, min_pres, max_pres):
        logger.info(
            f"Running interpolations (pressure, blend) between {min_pres} and {max_pres} Pa"
        )

        p_val_len: int = 30
        # s_range_len: int = 30
        x_range_len: int = 21

        p_vals = np.linspace(min_pres, max_pres, p_val_len)

        x_vals: np.ndarray = np.linspace(0, 1, x_range_len)
        if self.x_no_h2["H2"] == 1.0:
            x_vals = np.asarray([0.0, 1.0])
            x_range_len = len(x_vals)

        # Pre allocate solutions
        rho_vals: np.ndarray = np.zeros((x_range_len, p_val_len))
        mu_vals: np.ndarray = np.zeros((x_range_len, p_val_len))
        hhv_vals: np.ndarray = np.zeros(x_range_len)
        mw_vals: np.ndarray = np.zeros(x_range_len)
        cpcv_vals = np.zeros((x_range_len, p_val_len))

        # A second array of the assigned pressure. This is in case density is negative
        p_vals_final: np.ndarray = np.linspace(min_pres, max_pres, p_val_len)

        for x_i, x in enumerate(x_vals):
            # Update blend
            tmp_x = {a: b * (1 - x) for a, b in self.x_no_h2.items() if a != "H2"}
            tmp_x["H2"] = x
            if sum([molefrac for molefrac in tmp_x.values()]) == 0.0:
                tmp_x["H2"] = 1.0

            # Calculate HHV:
            hhv_vals[x_i] = self.calc_HHV(tmp_x)

            for p_i, p in enumerate(p_vals):
                p_a = p + ct.one_atm

                p_a = eos.set_state(
                    input_type="TPX",
                    input_vals=(gl.T_FIXED, p_a, tmp_x),
                    eos_type=self.eos_type,
                )

                # Set Cantera state
                p_vals_final[p_i] = p_a

                # Extract properties
                (
                    rho_vals[x_i, p_i],
                    mu_vals[x_i, p_i],
                    mw_vals[x_i],
                    cp,
                    cv,
                ) = eos.get_props(
                    input_type="TPX",
                    input_vals=(gl.T_FIXED, p_a, tmp_x),
                    eos_type=self.eos_type,
                    properties=(
                        "density",
                        "viscosity",
                        "mean_molecular_weight",
                        "cp_mass",
                        "cv_mass",
                    ),
                    set_state_flag=False,
                )
                cpcv_vals[x_i, p_i] = cp / cv

        # Make interpolators for each (indexed to absolute pressure)
        self.curve_fit_rho = scipy.interpolate.RectBivariateSpline(
            x_vals, p_vals_final, rho_vals, kx=1, ky=1
        )
        self.curve_fit_mu = scipy.interpolate.RectBivariateSpline(
            x_vals, p_vals_final, mu_vals, kx=1, ky=1
        )
        self.curve_fit_mw = scipy.interpolate.interp1d(x_vals, mw_vals)
        self.curve_fit_hhv = scipy.interpolate.interp1d(x_vals, hhv_vals)
        self.curve_fit_cpcv = scipy.interpolate.RectBivariateSpline(
            x_vals, p_vals_final, cpcv_vals, kx=1, ky=1
        )

    def pure_h2_hhv_gcv(self):
        hhv = self.calc_HHV({"H2": 1})
        ctu.gas.TPX = 273.15, ct.one_atm, "H2:1"
        mw = ctu.gas.mean_molecular_weight
        v = ctu.gas.volume_mole
        gcv = hhv * mw / v
        return hhv, gcv

    def pure_ng_hhv_gcv(self):
        hhv = self.calc_HHV(self.x_no_h2)
        ctu.gas.TPX = (
            273.15,
            ct.one_atm,
            self.x_no_h2,
        )
        mw = ctu.gas.mean_molecular_weight
        v = ctu.gas.volume_mole
        gcv = hhv * mw / v
        return hhv, gcv

    def set_composition_tracking(self, composition_tracking: bool) -> None:
        self.composition_tracking = composition_tracking
        self.make_interps(self.min_pres, self.max_pres)
