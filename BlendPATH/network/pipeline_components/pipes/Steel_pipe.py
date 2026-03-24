import numpy as np

import BlendPATH.util.pipe_assessment as bp_pa
import BlendPATH.util.pipe_dimensions
from BlendPATH.network.pipeline_components import Node, Pipe, friction_factor
from BlendPATH.util.schedules import SCHEDULES_STEEL_SCH


class Steel_pipe(Pipe):
    def __init__(
        self,
        from_node: Node,
        to_node: Node,
        name: str = "",
        diameter_mm: float = 0,
        length_km: float = 0,
        roughness_mm: float = 0.012,
        thickness_mm: float = 0,
        rating_code: str = "",
        p_max_mpa_g: float = 0,
        ff_type: friction_factor.FF_TYPES = "hofer",
    ):
        super().__init__(
            from_node=from_node,
            to_node=to_node,
            name=name,
            diameter_mm=diameter_mm,
            length_km=length_km,
            roughness_mm=roughness_mm,
            thickness_mm=thickness_mm,
            rating_code=rating_code,
            grade=rating_code,
            p_max_mpa_g=p_max_mpa_g,
            ff_type=ff_type,
        )
        self.material = "steel"
        self.assign_DN()
        self.assign_sch()

    def assign_DN(self) -> None:
        """
        Determine the DN of the pipe based on the SCHEDULE table
        """
        self.DN, self.NPS = BlendPATH.util.pipe_dimensions.get_DN_NPS(
            self.diameter_out_mm
        )

    def assign_sch(self) -> None:
        """
        Return the schedule based on the DN and lookup in SCHEDULES table
        """
        if self.DN not in SCHEDULES_STEEL_SCH:
            self.schedule = None
            return
        ths = np.array([*SCHEDULES_STEEL_SCH[self.DN].values()])
        schs = [*SCHEDULES_STEEL_SCH[self.DN].keys()]

        if len(ths) == 0:
            self.schedule = None
            return

        # If thickness is 0, just pick the thinnest schedule
        if self.thickness_mm == 0:
            self.schedule = schs[np.argmin(ths)]
            return

        self.schedule = schs[np.nanargmin(abs(ths - self.thickness_mm))]

    def pipe_assessment(
        self,
        design_option: str = "b",
        location_class: int = 1,
        joint_factor: float = 1,
        T_derating_factor: float = 1,
    ) -> None:
        """
        Reassign the ASME B31.12 design pressure
        """
        self.SMYS, self.SMTS = bp_pa.get_SMYS_SMTS(self.grade)
        self.design_pressure_MPa = self.design_pressure_ASME(
            design_option=design_option,
            location_class=location_class,
            joint_factor=joint_factor,
            T_derating_factor=T_derating_factor,
        )

    def design_pressure_ASME(
        self,
        design_option: str,
        location_class: int,
        joint_factor: int,
        T_derating_factor: int,
    ) -> float:
        """
        Calculates the ASME B31.12 design pressure
        """
        design_factor = bp_pa.get_design_factor(
            design_option=design_option, location_class=location_class
        )
        pressure_ASME_MPa = bp_pa.get_design_pressure_ASME(
            design_p_MPa=self.p_max_mpa_g,
            design_option=design_option,
            SMYS=self.SMYS,
            SMTS=self.SMTS,
            t=self.thickness_mm,
            D=self.DN,
            F=design_factor,
            E=joint_factor,
            T=T_derating_factor,
        )
        return pressure_ASME_MPa
