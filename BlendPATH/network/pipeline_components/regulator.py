from enum import StrEnum

import BlendPATH.Global as gl


class RegulatorMode(StrEnum):
    CONTROL = "control"
    OPEN = "open"
    CLOSED = "closed"


from .node import Node


class Regulator:
    """
    A pressure regulator with discrete hydraulic operating modes.
    CONTROL: p_out = p_set
    WIDE_OPEN: p_out = p_in
    CLOSED: flow_mdot = 0
    """

    def __init__(
        self,
        name: str,
        from_node: Node,
        to_node: Node,
        pressure_out_mpa_g: float,
        blend_limit: float | None = None,
    ) -> None:
        self.name = name
        self.from_node = from_node
        self.to_node = to_node
        # the regulator setpoint in MPa gauge
        self.pressure_out_mpa_g = pressure_out_mpa_g

        if self.pressure_out_mpa_g < 0:
            raise ValueError("Regulator outlet pressure must be non-negative")

        self.blend_limit = (
            float(blend_limit) if blend_limit is not None else blend_limit
        )

        self.mode: RegulatorMode = RegulatorMode.CONTROL
        self.flow_mdot: float = 0.0
        self.mode_switch_count: int = 0
        self.last_limit_violation: list[str] = []

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
        Outlet-to-inlet pressure ratio.
        In control and wide-open modes, this should normally be less than or
        equal to one. In closed mode, downstream pressure may exceed upstream
        pressure because the two pressure zones are isolated.
        """
        return self.to_node.pressure / self.from_node.pressure

    @property
    def setpoint_pa(self) -> float:
        """
        Regulator outlet pressure setpoint in Pa gauge.
        """
        return self.pressure_out_mpa_g * gl.MPA2PA

    def reset_solve_state(self) -> None:
        """
        Reset per-solve regulator state.
        The initial mode is estimated from inlet pressure. Closed mode is
        selected later if a forward-flow mode produces negative regulator flow.
        """
        self.flow_mdot = 0.0
        self.mode_switch_count = 0
        self.last_limit_violation = []

        if self.from_node.pressure is None:
            return

        if self.from_node.pressure >= self.setpoint_pa:
            self.mode = RegulatorMode.CONTROL
        else:
            self.mode = RegulatorMode.OPEN

    def update_mode(
        self,
        eps_pa: float = 0.0,
        recover_eps_pa: float | None = None,
        eps_mdot: float = 1e-7,
    ) -> bool:
        """
        Update the regulator operating mode.
        CONTROL:
            Switch to wide-open if the inlet cannot support the setpoint.
            Switch to closed if regulator flow is negative.
        WIDE_OPEN:
            Switch to control if the inlet can support the setpoint.
            Switch to closed if regulator flow is negative.
        CLOSED:
            Reopen when downstream pressure falls below the maximum outlet
            pressure the regulator can provide.
        Returns True if the mode changed.
        This method changes only the discrete operating mode. It does not
        directly assign pressure or flow.
        """
        if eps_pa < 0:
            raise ValueError("eps_pa must be non-negative")

        if recover_eps_pa is None:
            recover_eps_pa = eps_pa

        if recover_eps_pa < 0:
            raise ValueError("recover_eps_pa must be non-negative")

        if eps_mdot < 0:
            raise ValueError("eps_mdot must be non-negative")

        p_in = self.from_node.pressure
        p_out = self.to_node.pressure

        if p_in is None:
            return False

        old_mode = self.mode
        p_set = self.setpoint_pa

        if old_mode == RegulatorMode.CONTROL:
            if p_in < p_set - eps_pa:
                self.mode = RegulatorMode.OPEN
            elif self.flow_mdot < -eps_mdot:
                self.mode = RegulatorMode.CLOSED

        elif old_mode == RegulatorMode.OPEN:
            if self.flow_mdot < -eps_mdot:
                self.mode = RegulatorMode.CLOSED
            elif p_in > p_set + recover_eps_pa:
                self.mode = RegulatorMode.CONTROL

        elif old_mode == RegulatorMode.CLOSED:
            if p_out is None:
                return False

            available_outlet_pressure = min(p_in, p_set)

            if p_out < available_outlet_pressure - recover_eps_pa:
                if p_in >= p_set:
                    self.mode = RegulatorMode.CONTROL
                else:
                    self.mode = RegulatorMode.OPEN

        else:
            raise ValueError(f"Unknown regulator mode: {self.mode}")

        changed = self.mode != old_mode

        if changed:
            self.mode_switch_count += 1

        return changed

    def check_limits(
        self,
        tol_pa: float = 1.0,
        tol_mdot: float = 1e-7,
        tol_mdot_rel: float = 1e-8,
    ) -> None:
        """
        Check physical regulator limits.
        Forward-flow modes:
            p_out <= p_in
            flow_mdot >= 0
        Closed mode:
            flow_mdot = 0
        Downstream pressure may exceed upstream pressure in closed mode because
        the regulator isolates the two pressure zones.
        """
        if tol_pa < 0:
            raise ValueError("tol_pa must be non-negative")

        if tol_mdot < 0:
            raise ValueError("tol_mdot must be non-negative")

        if tol_mdot_rel < 0:
            raise ValueError("tol_mdot_rel must be non-negative")

        if self.mode not in RegulatorMode:
            raise ValueError(f"Unknown regulator mode: {self.mode}")

        p_in = self.from_node.pressure
        p_out = self.to_node.pressure
        flow_tol = max(
            tol_mdot,
            tol_mdot_rel * max(1.0, abs(self.flow_mdot)),
        )

        if p_in is not None and p_out is not None:
            if self.mode != RegulatorMode.CLOSED and p_out - p_in > tol_pa:
                raise ValueError(
                    f"Regulator '{self.name}' violates no-boost limit after regulator solve"
                )
            if self.mode == RegulatorMode.CONTROL and p_in < self.setpoint_pa - tol_pa:
                raise ValueError(
                    f"Regulator '{self.name}' cannot support its setpoint "
                    "after regulator solve"
                )

        if self.mode == RegulatorMode.CLOSED:
            if abs(self.flow_mdot) > flow_tol:
                raise ValueError(
                    f"Non-zero flow through closed regulator '{self.name}'"
                )
        elif self.flow_mdot < -flow_tol:
            raise ValueError(f"Reverse flow through regulator '{self.name}'")
