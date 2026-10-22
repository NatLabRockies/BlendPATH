import BlendPATH.util.pipe_dimensions
from BlendPATH.network.pipeline_components import Node, Pipe, friction_factor
from BlendPATH.util import plastic_pipe
from BlendPATH.util.plastic_pipe import get_sdr


class Polymer_pipe(Pipe):
    def __init__(
        self,
        from_node: Node,
        to_node: Node,
        polymer: str,
        polymer_designation: str,
        name: str = "",
        diameter_mm: float = 0,
        length_km: float = 0,
        roughness_mm: float = 0.012,
        thickness_mm: float = 0,
        rating_code: str = "",
        p_max_mpa_g: float = 0,
        ff_type: friction_factor.FF_TYPES = "hofer",
        velocity_limit: float | None = None,
        design_factor: float | None = None,
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
            p_max_mpa_g=p_max_mpa_g,
            ff_type=ff_type,
            velocity_limit=velocity_limit,
            design_factor=design_factor,
        )
        self.material = "polymer"
        self.polymer = polymer
        self.polymer_designation = polymer_designation
        self.assign_DN()

    def assign_DN(self) -> None:
        """
        Determine the DN of the pipe based on the SCHEDULE_PE table
        """
        self.DN, self.NPS = BlendPATH.util.pipe_dimensions.get_DN_NPS(
            self.diameter_out_mm
        )

    def pipe_assessment(
        self,
        design_option: float = 0.32,
        location_class: int = 1,
        joint_factor: float = 1,
        T_derating_factor: float = 1,
    ) -> None:
        """
        Reassign the ASTM D2513 Annex X1 design pressure
        Ignore location class, joint factor, and temperature derating factor
        Keep in for similarity to steel pipe
        """

        self.design_pressure_MPa = plastic_pipe.design_pressure_thermoplastic(
            design_factor=self.design_factor
            if self.design_factor is not None
            else design_option,
            sdr=self.dimension_ratio,
            polymer=self.polymer,
            polymer_designation=self.polymer_designation,
        )

    @property
    def dimension_ratio(self) -> float:
        return get_sdr(od_mm=self.diameter_out_mm, th_mm=self.thickness_mm)
