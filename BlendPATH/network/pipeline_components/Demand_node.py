import BlendPATH.Global as gl

from .Node import Node


class Demand_node:
    """
    A network node that has a specified energy flow rate
    """

    node: Node
    name: str = ""
    flowrate_MW: float = 0
    min_pressure_mpa_g: float = 0.0

    def __init__(
        self,
        node: Node,
        name: str = "",
        flowrate_MW: float = 0,
        min_pressure_mpa_g: float = 0.0,
    ):
        self.node = node
        self.name = name
        self.flowrate_MW = flowrate_MW
        self.min_pressure_mpa_g = min_pressure_mpa_g

        self.recalc_mdot()
        self.node.is_demand = True

    def recalc_mdot(self, hhv: float = None) -> None:
        """
        Calculate the new flow rate based on the HHV
        """
        if hhv is None:
            hhv = self.node.heating_value()
        self.flowrate_mdot = self.flowrate_MW / hhv  # kg/s
        return self.flowrate_mdot

    @property
    def flowrate_MMBTU_day(self):
        """
        Get flow rate in MMBTU/day
        """
        return self.flowrate_MW * gl.MW2MMBTUDAY
