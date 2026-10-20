from __future__ import annotations

from typing import TYPE_CHECKING

import BlendPATH.Global as gl

if TYPE_CHECKING:
    from .node import Node
    from .pipe import Pipe


class PipeSegmentDist:
    def __init__(self, pipes: list[Pipe], nodes: list[Node]):
        self.pipes = pipes
        self.nodes = nodes

    def violating_pipes(self) -> tuple[list[str], float]:
        max_pressure_MPa = self.max_pressure_MPa
        return [
            pipe.name
            for pipe in self.pipes
            if pipe.design_pressure_MPa < max_pressure_MPa
        ], max_pressure_MPa

    @property
    def min_maop_pa(self):
        return min([pipe.design_pressure_MPa for pipe in self.pipes]) * gl.MPA2PA

    @property
    def max_pressure_MPa(self):
        return max([p.p_max for p in self.pipes]) / gl.MPA2PA
