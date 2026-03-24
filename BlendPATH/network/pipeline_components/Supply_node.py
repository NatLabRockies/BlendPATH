import BlendPATH.Global as gl

from .Demand_node import Demand_node
from .Node import Node


class Supply_node(Demand_node):
    """
    A supply node where pressure is fixed
    """

    def __init__(
        self,
        node: Node,
        name: str = "",
        flowrate_MW: float | None = 0,
        min_pressure_mpa_g: float = 0.0,
        pressure_mpa: float | None = None,
        blend: float = 0.0,
    ):
        super().__init__(
            node=node,
            name=name,
            flowrate_MW=flowrate_MW if flowrate_MW is not None else 0,
            min_pressure_mpa_g=min_pressure_mpa_g,
        )

        self.pressure_mpa = pressure_mpa
        self.is_pressure_supply = self.pressure_mpa is not None
        self.blend = blend
        self.node.is_demand = False
        self.node.is_supply = True
        self.blendH2()

    def blendH2(self):
        self.node.x_h2 = self.blend
        self.hhv = self.node.heating_value()

        if self.is_pressure_supply:
            self.node.pressure = self.pressure_mpa * gl.MPA2PA
        else:
            self.recalc_mdot(self.hhv)
        self.mw = self.node.mw

    @property
    def mdot(self):
        if not self.is_pressure_supply:
            return self.flowrate_mdot
        else:
            mdot_sum = 0
            for cxn in self.node.connections["Pipe"]:
                # Flowing into node
                pos_or_neg = -1

                # Flowing out of node
                if (cxn.from_node is self.node and cxn.direction == 1) or (
                    cxn.to_node is self.node and cxn.direction == -1
                ):
                    pos_or_neg = 1
                mdot_sum += abs(cxn.m_dot) * pos_or_neg
            for cxn in self.node.connections["Comp"]:
                if cxn.from_node is self.node:
                    mdot_sum += getattr(cxn, "flow_mdot", 0) + getattr(
                        cxn, "fuel_mdot", 0
                    )
            return max(0, mdot_sum)
