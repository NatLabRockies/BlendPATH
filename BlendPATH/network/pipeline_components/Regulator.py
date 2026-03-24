import BlendPATH.Global as gl

from .Node import Node


class Regulator:
    """
    A simple pressure regulator object.
    Forces 'to_node' pressure <= setpoint if 'from_node' is higher.
    """

    def __init__(
        self,
        name: str,
        from_node: Node,
        to_node: Node,
        pressure_out_mpa_g: float,
    ):
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        # the regulator setpoint in MPa gauge
        self.pressure_out_mpa_g = pressure_out_mpa_g

        if self.pressure_out_mpa_g < 0:
            raise ValueError("Regulator pressure out must be postive")

    def apply_regulation(self) -> None:
        """
        Enforce to_node pressure so that it does not exceed the regulator setpoint.
        If the inlet pressure is lower than the setpoint, the regulator does nothing (cannot boost pressure).
        """
        setpoint_pa = self.pressure_out_mpa_g * gl.MPA2PA
        inlet_pa = self.from_node.pressure
        self.to_node.pressure = min(inlet_pa, setpoint_pa)

    @property
    def pressure_ratio(self) -> float:
        """
        Get pressure ratio - to_node is high pressure than from_node
        """
        return self.to_node.pressure / self.from_node.pressure
