from enum import IntEnum
from typing import TYPE_CHECKING, Union

import cantera as ct
import numpy as np
import numpy.typing as npt

import BlendPATH.Global as gl
from BlendPATH.network.pipeline_components import friction_factor

from . import cantera_util as ctu

if TYPE_CHECKING:
    from .Node import Node


class Pipe:
    """
    A pipe that connects two nodes
    """

    def __init__(
        self,
        from_node: "Node",
        to_node: "Node",
        name: str = "",
        diameter_mm: float = 0,
        length_km: float = 0,
        roughness_mm: float = 0.012,
        thickness_mm: float = 0,
        rating_code: str = "",
        p_max_mpa_g: float = 0,
        ff_type: friction_factor.FF_TYPES = "hofer",
        grade: str | None = None,
        material: str | None = None,
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.name = name
        self.diameter_mm = diameter_mm
        self.length_km = length_km
        self.roughness_mm = roughness_mm
        self.thickness_mm = thickness_mm
        self.rating_code = rating_code
        self.p_max_mpa_g = p_max_mpa_g
        self.m_dot = None
        self.ff_type = ff_type
        self.grade = grade
        self._parent_pipe = None
        self.material = material

    def get_derivative(self, coef: float, p_eqn: float) -> float:
        """
        Calculate the derivative dm/dp for the solver
        """
        if p_eqn == 0:
            return np.inf
        return coef * p_eqn**-0.5

    def get_mdot(self, coef: float, p_eqn: float) -> float:
        """
        Get the mass flow rate through the pipe
        """
        C_p_eqn = coef * p_eqn**0.5
        self.m_dot = C_p_eqn * self.direction
        return self.m_dot

    def get_d_and_mdot(
        self,
        rho_avg: float,
        z_avg: float,
        mu: float,
        mw: float,
    ) -> tuple:
        # Call flow eqn
        coef, p_eqn = self.get_flow_eqn_const(
            rho_avg=rho_avg, z_avg=z_avg, mu=mu, mw=mw
        )

        # get derivative term
        C_p_eqn_deriv = self.get_derivative(coef=coef, p_eqn=p_eqn)

        # Get mass flow rate
        mdot = self.get_mdot(coef=coef, p_eqn=p_eqn)

        return C_p_eqn_deriv, mdot

    def get_flow_eqn_const(
        self,
        rho_avg: float,
        z_avg: float,
        mu: float,
        mw: float,
    ) -> float:
        """
        Calculate momentum equation coefficient
        """
        p_in = self.from_node.pressure + ct.one_atm
        p_out = self.to_node.pressure + ct.one_atm
        if p_in < p_out:
            p_in, p_out = p_out, p_in

        A = self.A_m2
        D = self.diameter_mm * gl.MM2M
        L = self.length_km * gl.KM2M

        if self.m_dot is None:
            self.Re = 1e8
        else:
            m_dot_abs = abs(self.m_dot)
            self.v_avg = m_dot_abs / rho_avg / A
            self.Re = rho_avg * self.v_avg * D / mu
            self.v_from = m_dot_abs / self.from_node.rho / A
            self.v_to = m_dot_abs / self.to_node.rho / A

        self.f = friction_factor.get_friction_factor_vector(
            Re=self.Re,
            roughness_mm=self.roughness_mm,
            diameter_mm=self.diameter_mm,
            ff_type=self.ff_type,
        )

        coef = A * (mw / z_avg / ctu.R_GAS / gl.T_FIXED * D / self.f / L) ** (0.5)

        p_eqn = p_in**2 - p_out**2
        if p_eqn < 1e-6:
            p_eqn = 1e-6

        return coef, p_eqn

    @property
    def v_from(self) -> Union[float, None]:
        return abs(self.m_dot) / self.from_node.rho / self.A_m2

    @property
    def v_to(self) -> Union[float, None]:
        return abs(self.m_dot) / self.to_node.rho / self.A_m2

    @property
    def v_avg(self) -> Union[float, None]:
        return (self.v_from + self.v_to) / 2

    @property
    def v_max(self) -> Union[float, None]:
        return max(self.v_from, self.v_to)

    @property
    def p_avg(self) -> float:
        p_in = self.from_node.pressure + ct.one_atm
        p_out = self.to_node.pressure + ct.one_atm
        return 2 / 3 * (p_in + p_out - p_in * p_out / (p_in + p_out)) - ct.one_atm

    @property
    def direction(self) -> float:
        """
        Return direction of pipe flow
        """
        if self.to_node.pressure > self.from_node.pressure:
            return -1
        return 1

    @property
    def mach_number(self) -> float:
        """
        Calculate mach number
        """
        if (
            self.v_from is None
            or self.from_node.pressure is None
            or self.to_node.pressure is None
        ):
            return 0
        v = [self.v_from, self.v_to]
        c = [
            np.sqrt(x.cpcv * x.pressure / x.rho) for x in [self.from_node, self.to_node]
        ]
        m = [v[x] / c[x] for x in [0, 1]]
        return max(m)

    @property
    def erosional_velocity_ASME(self) -> float:
        """
        Calculate ASME B31.12 erosional velocity
        """

        KG2LB = 2.20462
        M32FT3 = 35.3147
        FT2M = 0.3048

        rho = min([x.rho for x in [self.from_node, self.to_node]])
        u = 100 / np.sqrt(rho * KG2LB / M32FT3)
        u_m_s = u * FT2M
        return u_m_s  # m/s

    @property
    def A_m2(self) -> float:
        return (self.diameter_mm * gl.MM2M) ** 2 / 4 * np.pi

    @property
    def diameter_out_mm(self) -> float:
        return self.diameter_mm + 2 * self.thickness_mm

    @property
    def dimension_ratio(self) -> float:
        return round(self.diameter_out_mm / self.thickness_mm, 2)

    @property
    def pressure_MPa(self) -> float:
        return max(self.from_node.pressure, self.to_node.pressure) / gl.MPA2PA

    @property
    def Re(self) -> float:
        if self.m_dot is None or np.isnan(self.m_dot):
            return 1e8
        mu_avg = self.to_node.composition.get_mu(p_gauge_pa=self.p_avg, x=self.x_h2)
        return (np.abs(self.m_dot) / self.A_m2) * self.diameter_mm * gl.MM2M / mu_avg

    @property
    def f(self) -> float:
        return friction_factor.get_friction_factor_vector(
            Re=np.asarray([self.Re]),
            roughness_mm=np.asarray([self.roughness_mm]),
            diameter_mm=np.asarray([self.diameter_mm]),
            ff_type=self.ff_type,
        )

    @staticmethod
    def get_flow_eqn_vector(
        p_in: np.ndarray,
        p_out: np.ndarray,
        z_avg: float,
        mw: float,
        A: np.ndarray,
        D: np.ndarray,
        L: np.ndarray,
        f: np.ndarray,
    ) -> tuple:
        """
        Calculate momentum equation coefficient
        """
        coef = A * (mw / z_avg / ctu.R_GAS / gl.T_FIXED * D / f / L) ** (0.5)

        p_eqn = np.maximum(np.abs(p_in**2 - p_out**2), 1e-6)

        return coef, p_eqn

    class STATIC_PROPS(IntEnum):
        """These read out the index order for get_pipe_static_properties"""

        A_M2 = 0
        D = 1
        L = 2
        Ro = 3

    def get_pipe_static_properties(self):
        return [
            self.A_m2,
            self.diameter_mm * gl.MM2M,
            self.length_km * gl.KM2M,
            self.roughness_mm,
        ]

    class DYN_PROPS(IntEnum):
        """These read out the index order for get_pipe_dynamic_properties"""

        P_FROM = 0
        P_TO = 1
        X_H2 = 2
        DIR = 3

    def get_pipe_dynamic_properties(self):
        return [
            self.from_node.pressure + ct.one_atm,
            self.to_node.pressure + ct.one_atm,
            self.x_h2,
            self.direction,
        ]

    @staticmethod
    def get_Re_vector(
        m_dot: np.ndarray, a_m2: np.ndarray, d_m: np.ndarray, mu: np.ndarray
    ):
        if np.any(np.isnan(m_dot)):
            return np.full_like(m_dot, 1e8, dtype=float)
        return np.abs(m_dot) / a_m2 * d_m / mu

    @staticmethod
    def p_avg_vector(p_in, p_out):
        return 2 / 3 * (p_in + p_out - p_in * p_out / (p_in + p_out)) - ct.one_atm

    @staticmethod
    def get_derivative_vector(coef: np.ndarray, p_eqn: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative dm/dp for the solver
        """
        return coef * p_eqn**-0.5

    @staticmethod
    def get_mdot_vector(
        coef: np.ndarray, p_eqn: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """
        Get the mass flow rate through the pipe
        """
        C_p_eqn = coef * p_eqn**0.5
        return C_p_eqn * direction

    @staticmethod
    def get_jacobian_components(
        dyn_props: npt.NDArray[np.float64],
        static_props: npt.NDArray[np.float64],
        z_avgs: npt.NDArray[np.float64],
        mw: npt.NDArray[np.float64],
        mu: npt.NDArray[np.float64],
        ff_type: bool,
        m_dot_prev: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        Re = Pipe.get_Re_vector(
            m_dot=m_dot_prev,
            a_m2=static_props[:, Pipe.STATIC_PROPS.A_M2],
            d_m=static_props[:, Pipe.STATIC_PROPS.D],
            mu=mu,
        )

        f = friction_factor.get_friction_factor_vector(
            Re=Re,
            roughness_mm=static_props[:, Pipe.STATIC_PROPS.Ro],
            diameter_mm=static_props[:, Pipe.STATIC_PROPS.D] / gl.MM2M,
            ff_type=ff_type,
        )

        coef, p_eqn = Pipe.get_flow_eqn_vector(
            p_in=dyn_props[:, Pipe.DYN_PROPS.P_FROM],
            p_out=dyn_props[:, Pipe.DYN_PROPS.P_TO],
            z_avg=z_avgs,
            mw=mw,
            A=static_props[:, Pipe.STATIC_PROPS.A_M2],
            D=static_props[:, Pipe.STATIC_PROPS.D],
            L=static_props[:, Pipe.STATIC_PROPS.L],
            f=f,
        )
        dm_dp = Pipe.get_derivative_vector(coef=coef, p_eqn=p_eqn)
        m_dot_pipe_v = Pipe.get_mdot_vector(
            coef=coef, p_eqn=p_eqn, direction=dyn_props[:, Pipe.DYN_PROPS.DIR]
        )
        return dm_dp, m_dot_pipe_v
