from dataclasses import dataclass
from operator import itemgetter

import numpy as np

import BlendPATH.util.pipe_assessment as bp_pa
from BlendPATH.util.pipe_helper import parent_pipe_helper
from BlendPATH.util.schedules import SCHEDULES_STEEL_DN, SCHEDULES_STEEL_SCH

from .Node import Node


@dataclass
class PipeSegment:
    """
    Pipe segment of a network. Collection of pipes. Also compressor is if it at the end of the segment
    """

    pipes: list
    diameter: float
    DN: float
    comps: list
    start_node: Node = None
    nodes: list = None
    demand_nodes: list = None
    pressure_violation: bool = False
    p_max_mpa_g: float = 0
    design_pressure_MPa: float = 0
    offtake_lengths: list = None

    # def post_segmentation(self) -> None:
    #     """
    #     Assign overall parameters of the segment after segmentation process
    #     """
    #     # self.assign_length()
    #     # self.assign_end_node()
    #     # self.assign_flows()

    @property
    def length_km(self) -> float:
        """
        Calculate total length of pipe in segment [km]
        """
        return round(sum([pipe.length_km for pipe in self.pipes]), 12)

    def check_p_violations(self) -> None:
        """
        Determine if any pressure violations within pipe segment
        """
        self.pressure_violation = True in [pipe.design_violation for pipe in self.pipes]

    @property
    def mdot_in(self) -> float:
        """
        Inlet flow rate based on first pipe
        """
        return self.pipes[0].m_dot

    @property
    def mdot_out(self) -> float:
        """
        Outlet flow rate based on first pipe
        """
        return self.offtake_mdots[-1]

    @property
    def end_node(self) -> Node:
        """
        Determine last node in the segment by taking the last node in the node list. Used when determined segment outlet pressure
        """
        return self.nodes[-1]

    @property
    def offtake_mdots(self) -> list:
        """
        Get the list of offtakes flow rates. Summing up stacked demands
        """
        offtake_mdots = []

        # Sum up flow rates for stacked node (multiple demands at same node)
        # Need to maintain order, so don't use a set
        nodes = list(dict.fromkeys([dn.node.name for dn in self.demand_nodes]).keys())
        for node in nodes:
            offtake_mdots.append(
                np.sum(
                    [
                        d_n.flowrate_mdot
                        for d_n in self.demand_nodes
                        if d_n.node.name == node
                    ]
                )
            )
        return offtake_mdots

    @property
    def HHV(self) -> float:
        """
        Get the heating value of the segment based on the composition at the start node
        """
        return self.start_node.heating_value()

    def get_DNs(self, max_number=5) -> tuple:
        """
        Get the DN options equal or larger than the existing pipe. Use in PL, DR method
        """
        pick_ODs = [dn for dn in SCHEDULES_STEEL_DN if dn >= self.DN]
        pick_DNs = list(itemgetter(*pick_ODs)(SCHEDULES_STEEL_DN))
        dn_options = pick_DNs[:max_number]
        od_options = pick_ODs[:max_number]
        return dn_options, od_options

    def get_viable_schedules(
        self,
        design_option: bp_pa._DESIGN_OPTIONS,
        ASME_params: bp_pa.ASME_consts,
        grade: str,
        ASME_pressure_flag: bool = False,
        DN: float = None,
        return_all: bool = False,
    ) -> tuple:
        """
        Get viable schedules for the pipesegment
        """

        if DN is None:
            check_DN = self.DN
        else:
            check_DN = DN
        sch_list = SCHEDULES_STEEL_SCH[check_DN]

        # by default use the rating of original pipe. If set to none, the we are using
        # DR method, and don't know what the pipe pressure will be
        design_pressure = self.design_pressure_MPa if ASME_pressure_flag else None

        (th, schedule, pressure, index) = bp_pa.get_viable_schedules(
            sch_list,
            design_option,
            ASME_params,
            grade,
            design_pressure,
            self.design_pressure_MPa,
            check_DN,
        )
        if return_all:
            _, unique_indexes = np.unique(th, return_index=True)
            return (
                [th[i] for i in unique_indexes],
                [schedule[i] for i in unique_indexes],
                [pressure[i] for i in unique_indexes],
            )
        else:
            if index == -1:
                return (None, np.nan, None)
            return (th[index], schedule[index], pressure[index])

    @property
    def parent_pipes(self) -> dict:
        return parent_pipe_helper(self.pipes)
