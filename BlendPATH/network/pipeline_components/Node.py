from BlendPATH.network import pipeline_components as bp_plc
from BlendPATH.network.pipeline_components.eos import _EOS_OPTIONS


class Node:
    """
    Nodes in a network. Must be connected to pipes or compressors. Can be a supply or demand as well.
    """

    def __init__(
        self,
        name: str = "",
        p_max_mpa_g: float = 0,
        index: int = 0,
        pressure: float = None,
        composition: bp_plc.Composition = None,
        x_h2: float = 0.0,
        is_demand: bool = False,
        is_supply: bool = False,
        eos_type: _EOS_OPTIONS = None,
        _report_out: bool = True,
        plot_xy: tuple = None,
    ):
        self.name = name
        self.p_max_mpa_g = p_max_mpa_g
        self.index = index
        self.pressure = pressure
        self.composition = composition
        self.x_h2 = x_h2
        self.connections = {"Pipe": [], "Comp": [], "Reg": []}
        self.is_demand = is_demand
        self.is_supply = is_supply
        self.eos_type = eos_type
        self._report_out = _report_out
        self.plot_xy = plot_xy

    def clear_connections(self) -> None:
        """
        Function to clear out connections
        """
        self.connections = {"Pipe": [], "Comp": [], "Reg": []}

    def add_connection(self, cxn) -> None:
        """
        Add pipe or compressor cxn to node
        """
        if isinstance(cxn, (bp_plc.Pipe, bp_plc.Steel_pipe)):
            self.connections["Pipe"].append(cxn)
        elif isinstance(cxn, bp_plc.Compressor):
            self.connections["Comp"].append(cxn)
        elif isinstance(cxn, bp_plc.Regulator):
            self.connections["Reg"].append(cxn)
        else:
            raise ValueError(
                f"Connection provided {type(cxn)} was not Pipe, Compressor, or Regulator"
            )

    def update_state(self, T: float, p: float, x_h2: float) -> None:
        """
        Update the temperature, pressure, and composition at the node. Calculate rho and z based on EOS
        """
        if self.composition.eos_type is None:
            raise RuntimeError("Node EOS type is not set")

        self.pressure = p
        self.x_h2 = float(x_h2)

    @property
    def rho(self):
        """
        Density kg/m3
        """
        rho, _ = self.composition.get_rho_z(
            p_gauge_pa=self.pressure, x=self.x_h2, mw=self.mw
        )
        return rho

    @property
    def z(self):
        """
        Compressibility
        """
        _, z = self.composition.get_rho_z(
            p_gauge_pa=self.pressure, x=self.x_h2, mw=self.mw
        )
        return z

    def heating_value(self) -> float:
        """
        Get the higher heating value at the node.
        """
        return self.composition.get_hhv(self.x_h2)

    @property
    def mw(self):
        """
        Molecular weight (kg/kmol).
        """

        return self.composition.get_mw(p_gauge_pa=self.pressure, x=self.x_h2)

    @property
    def mu(self):
        """
        Molecular weight (kg/kmol).
        """
        return self.composition.get_mu(p_gauge_pa=self.pressure, x=self.x_h2)

    @property
    def cpcv(self):
        """
        Heat capacity
        """
        return self.composition.get_cpcv(p_gauge_pa=self.pressure, x=self.x_h2)
