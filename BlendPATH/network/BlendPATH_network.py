import logging
from collections import defaultdict, deque
from os.path import isfile
from pathlib import Path
from typing import Literal, Union, get_args

import cantera as ct
import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg

import BlendPATH.file_writing.file_writing as bp_file_w
import BlendPATH.file_writing.network_file_out_util as bp_network_file
import BlendPATH.Global as gl
import BlendPATH.network.network_import as bp_ni
import BlendPATH.network.network_solving_util as network_util
import BlendPATH.network.pipeline_components.cantera_util as ctu
from BlendPATH.costing.costing import Costing_params
from BlendPATH.network import pipeline_components as plc
from BlendPATH.network.pipeline_components import Pipe
from BlendPATH.network.pipeline_components.eos import _EOS_OPTIONS
from BlendPATH.network.pipeline_components.friction_factor import FF_TYPES
from BlendPATH.scenario_helper import (
    SCENARIO_VALUES,
    Design_params,
    Scenario_type,
)
from BlendPATH.util.pipe_assessment import PIPE_STR, ASME_consts
from BlendPATH.util.pipe_helper import parent_pipe_helper

logger = logging.getLogger(__name__)


class BlendPATH_network:
    """
    Hydraulic network
    """

    def __init__(
        self,
        composition: plc.Composition,
        name: str = "",
        pipes: dict | None = None,
        nodes: dict | None = None,
        demand_nodes: dict | None = None,
        supply_nodes: dict | None = None,
        compressors: dict | None = None,
        regulators: dict | None = None,
        eos: _EOS_OPTIONS = "rk",
        thermo_curvefit: bool = True,
        composition_tracking: bool = False,
        scenario_type: Scenario_type = Scenario_type.TRANSMISSION,
        ff_type: FF_TYPES = "hofer",
        validate_connectivity: bool = True,
    ) -> None:
        """
        Initialize a network
        """
        # Assign defaults
        self.name = name
        self.pipes = pipes if pipes is not None else {}
        self.update_pipe_designation()
        self.nodes = nodes if nodes is not None else {}
        self.demand_nodes = demand_nodes if demand_nodes is not None else {}
        self.supply_nodes = supply_nodes if supply_nodes is not None else {}
        self.compressors = compressors if compressors is not None else {}
        self.regulators = regulators if regulators is not None else {}
        self.composition = composition
        self.set_thermo_curvefit(thermo_curvefit)
        self.eos = eos
        self.set_composition_tracking(composition_tracking)
        self.ff_type = ff_type
        self.pipe_segments = []

        self.check_supply_demand_overlap()

        if scenario_type not in Scenario_type.options:
            raise ValueError(
                f"scenario_type must be one of {[scen_type for scen_type in Scenario_type.options]}, got {scenario_type}"
            )
        self.scenario_type = scenario_type

        self.assign_nodes()
        self.assign_connections()

        # Check if need more segments if some pipes are too long
        self.check_segmentation()

        if validate_connectivity:
            self.assert_reachable_directed(
                require="all_nodes",
            )

        self.unique_demand_nodes = list(
            set([dn.node.name for dn in self.demand_nodes.values()])
        )

    def update_pipe_designation(self) -> None:
        pipes_temp = {}
        for pipe in self.pipes.values():
            if type(pipe) is plc.Pipe:
                pipes_temp[pipe.name] = plc.Steel_pipe(
                    from_node=pipe.from_node,
                    to_node=pipe.to_node,
                    name=pipe.name,
                    diameter_mm=pipe.diameter_mm,
                    length_km=pipe.length_km,
                    roughness_mm=pipe.roughness_mm,
                    thickness_mm=pipe.thickness_mm,
                    rating_code=pipe.rating_code,
                    p_max_mpa_g=pipe.p_max_mpa_g,
                    ff_type=pipe.ff_type,
                )
                pipes_temp[pipe.name].m_dot = pipe.m_dot
            else:
                pipes_temp[pipe.name] = pipe
        self.pipes = pipes_temp

    def assign_connections(self) -> None:
        """
        Add connections for nodes to pipes/comps
        """
        for node in self.nodes.values():
            node.clear_connections()
        for pipe in self.pipes.values():
            pipe.from_node.add_connection(pipe)
            pipe.to_node.add_connection(pipe)
        for comp in self.compressors.values():
            comp.from_node.add_connection(comp)
            comp.to_node.add_connection(comp)
        for reg in self.regulators.values():
            reg.from_node.add_connection(reg)
            reg.to_node.add_connection(reg)

    def assign_nodes(self) -> None:
        node_index = 0
        for node in self.nodes.values():
            node.index = node_index
            node_index += 1

        self.ignore_nodes = []
        for sn in self.supply_nodes.values():
            if not sn.is_pressure_supply:
                continue
            self.ignore_nodes.append(sn.node.index)
        for comp in self.compressors.values():
            self.ignore_nodes.append(comp.to_node.index)
        for reg in self.regulators.values():
            self.ignore_nodes.append(reg.to_node.index)
        self.ignore_nodes.sort()

        self.nodes_clean = []
        index = 0
        for node in self.nodes.values():
            if node.index in self.ignore_nodes:
                self.nodes_clean.append(None)
                node.index_adj = None
            else:
                self.nodes_clean.append(index)
                node.index_adj = index
                index += 1

    @classmethod
    def import_from_file(
        cls,
        name: str,
        scenario_type: Scenario_type = Scenario_type.TRANSMISSION,
        composition_tracking: bool = False,
        thermo_curvefit: bool = True,
        eos: _EOS_OPTIONS = "rk",
        ff_type: FF_TYPES = "hofer",
        validate_connectivity: bool = True,
        blend: float | None = None,
    ):
        """
        Make network from file
        """
        filename_full_path = Path(name).absolute()
        logger.info(f"Importing file: {filename_full_path}")

        def check_node(node_list: dict, check_node: str) -> None:
            if check_node not in node_list:  # Check that the node exists
                raise ValueError(f"{check_node} is not in the node list")

        # Read in input from excel file by sheet
        if not isfile(filename_full_path):
            raise ValueError(
                f"Could not find file with file name: {filename_full_path}"
            )

        file_raw = bp_ni.read_workbook(filename_full_path)

        # Load in composition
        # If distribution network, get upper bound of pressure from supply nodes, compressors, and regulators,
        # otherwise if it is a transmission network, use the max pressure
        max_pressure = SCENARIO_VALUES[scenario_type].max_pressure_pa

        # Load composition
        composition = plc.Composition(
            pure_x=dict(
                zip(
                    file_raw[bp_ni.SheetName.COMPOSITION]["SPECIES"],
                    file_raw[bp_ni.SheetName.COMPOSITION]["X"],
                )
            ),
            min_pres=SCENARIO_VALUES[scenario_type].min_pressure_pa,
            max_pres=max_pressure,
            composition_tracking=composition_tracking,
            thermo_curvefit=thermo_curvefit,
            eos_type=eos,
            blend=blend,
        )

        # Create nodes
        nodes = {}
        for i, node_name in enumerate(file_raw[bp_ni.SheetName.NODES]["node_name"]):
            nodes[node_name] = plc.Node(
                name=node_name,
                p_max_mpa_g=float(file_raw[bp_ni.SheetName.NODES]["p_max_mpa_g"][i]),
                composition=composition,
                plot_xy=(
                    file_raw[bp_ni.SheetName.NODES]["plot_x"][i],
                    file_raw[bp_ni.SheetName.NODES]["plot_y"][i],
                ),
            )

        # Create supply nodes
        supply_nodes = {}
        for i, supply_name in enumerate(
            file_raw[bp_ni.SheetName.SUPPLY]["supply_name"]
        ):
            check_node(nodes, file_raw[bp_ni.SheetName.SUPPLY]["node_name"][i])
            supply_nodes[supply_name] = plc.Supply_node(
                name=supply_name,
                node=nodes[file_raw[bp_ni.SheetName.SUPPLY]["node_name"][i]],
                pressure_mpa=float(
                    file_raw[bp_ni.SheetName.SUPPLY]["pressure_mpa_g"][i]
                ),
                flowrate_MW=file_raw[bp_ni.SheetName.SUPPLY]["flowrate_MW"][i],
                blend=float(file_raw[bp_ni.SheetName.SUPPLY]["blend"][i]),
            )

        # Create demand nodes
        demand_nodes = {}
        for i, demand_name in enumerate(
            file_raw[bp_ni.SheetName.DEMAND]["demand_name"]
        ):
            check_node(nodes, file_raw[bp_ni.SheetName.DEMAND]["node_name"][i])
            demand_nodes[demand_name] = plc.Demand_node(
                name=demand_name,
                node=nodes[file_raw[bp_ni.SheetName.DEMAND]["node_name"][i]],
                flowrate_MW=float(file_raw[bp_ni.SheetName.DEMAND]["flowrate_MW"][i]),
                min_pressure_mpa_g=float(
                    file_raw[bp_ni.SheetName.DEMAND]["min_pressure_mpa_g"][i]
                ),
            )

        # Create pipes
        pipes = {}
        for i, pipe_name in enumerate(file_raw[bp_ni.SheetName.PIPES]["pipe_name"]):
            from_node = file_raw[bp_ni.SheetName.PIPES]["from_node"][i]
            to_node = file_raw[bp_ni.SheetName.PIPES]["to_node"][i]
            if from_node == to_node:
                continue
            check_node(nodes, from_node)
            check_node(nodes, to_node)
            max_pressure_mpa = min(
                [nodes[from_node].p_max_mpa_g, nodes[to_node].p_max_mpa_g]
            )

            rating_code = file_raw[bp_ni.SheetName.PIPES]["rating_code"][i].upper()
            if rating_code in PIPE_STR:
                pipes[pipe_name] = plc.Steel_pipe(
                    name=pipe_name,
                    from_node=nodes[from_node],
                    to_node=nodes[to_node],
                    diameter_mm=float(
                        file_raw[bp_ni.SheetName.PIPES]["diameter_mm"][i]
                    ),
                    length_km=float(file_raw[bp_ni.SheetName.PIPES]["length_km"][i]),
                    roughness_mm=float(
                        file_raw[bp_ni.SheetName.PIPES]["roughness_mm"][i]
                    ),
                    thickness_mm=float(
                        file_raw[bp_ni.SheetName.PIPES]["thickness_mm"][i]
                    ),
                    rating_code=rating_code,
                    p_max_mpa_g=max_pressure_mpa,
                )
            else:
                pipes[pipe_name] = plc.Pipe(
                    name=pipe_name,
                    from_node=nodes[from_node],
                    to_node=nodes[to_node],
                    diameter_mm=float(
                        file_raw[bp_ni.SheetName.PIPES]["diameter_mm"][i]
                    ),
                    length_km=float(file_raw[bp_ni.SheetName.PIPES]["length_km"][i]),
                    roughness_mm=float(
                        file_raw[bp_ni.SheetName.PIPES]["roughness_mm"][i]
                    ),
                    thickness_mm=float(
                        file_raw[bp_ni.SheetName.PIPES]["thickness_mm"][i]
                    ),
                    rating_code=rating_code,
                    p_max_mpa_g=max_pressure_mpa,
                )

        # Compressors
        compressors = {}
        for i, compressor_name in enumerate(
            file_raw[bp_ni.SheetName.COMPRESSORS]["compressor_name"]
        ):
            from_node = file_raw[bp_ni.SheetName.COMPRESSORS]["from_node"][i]
            to_node = file_raw[bp_ni.SheetName.COMPRESSORS]["to_node"][i]
            check_node(nodes, from_node)
            check_node(nodes, to_node)
            compressors[compressor_name] = plc.Compressor(
                name=compressor_name,
                from_node=nodes[from_node],
                to_node=nodes[to_node],
                pressure_out_mpa_g=float(
                    file_raw[bp_ni.SheetName.COMPRESSORS]["pressure_out_mpa_g"][i]
                ),
                original_rating_MW=float(
                    file_raw[bp_ni.SheetName.COMPRESSORS]["rating_MW"][i]
                ),
                fuel_extract=file_raw[bp_ni.SheetName.COMPRESSORS]["extract_fuel"][i],
            )

            if (
                eta_s_temp := file_raw[bp_ni.SheetName.COMPRESSORS]["eta_s"][i]
            ) is not None:
                if file_raw[bp_ni.SheetName.COMPRESSORS]["extract_fuel"][i]:
                    compressors[compressor_name].eta_comp_s = float(eta_s_temp)
                else:
                    compressors[compressor_name].eta_comp_s_elec = float(eta_s_temp)
            if (
                eta_driver_temp := file_raw[bp_ni.SheetName.COMPRESSORS]["eta_driver"][
                    i
                ]
            ) is not None:
                if file_raw[bp_ni.SheetName.COMPRESSORS]["extract_fuel"][i]:
                    compressors[compressor_name].eta_driver = float(eta_driver_temp)
                else:
                    compressors[compressor_name].eta_driver_elec = float(
                        eta_driver_temp
                    )

        # Regulators
        regulators = {}
        for i, regulator_name in enumerate(
            file_raw[bp_ni.SheetName.REGULATORS]["regulator_name"]
        ):
            from_node = file_raw[bp_ni.SheetName.REGULATORS]["from_node"][i]
            to_node = file_raw[bp_ni.SheetName.REGULATORS]["to_node"][i]
            check_node(nodes, from_node)
            check_node(nodes, to_node)

            regulators[regulator_name] = plc.Regulator(
                name=regulator_name,
                from_node=nodes[from_node],
                to_node=nodes[to_node],
                pressure_out_mpa_g=float(
                    file_raw[bp_ni.SheetName.REGULATORS]["pressure_out_mpa_g"][i]
                ),
            )

        return BlendPATH_network(
            name=name,
            pipes=pipes,
            nodes=nodes,
            demand_nodes=demand_nodes,
            supply_nodes=supply_nodes,
            compressors=compressors,
            regulators=regulators,
            composition=composition,
            composition_tracking=composition_tracking,
            thermo_curvefit=thermo_curvefit,
            scenario_type=scenario_type,
            eos=eos,
            ff_type=ff_type,
            validate_connectivity=validate_connectivity,
        )

    def blendH2(self, blend: float) -> None:
        """
        Blend amount of H2. Reassigns composition and recalculated flow rates
        """
        blend = float(blend)
        self.composition.blendH2(blend)

        for n in self.nodes.values():
            n.x_h2 = blend

        for sn in self.supply_nodes.values():
            sn.blend = blend
            sn.blendH2()

        # Update demand node conversion from MW to kg/s
        for dn in self.demand_nodes.values():
            dn.recalc_mdot()

    def solve(
        self,
        c_relax: float | None = None,
        cr_max: float = 1.5,
        low_p_buffer: float = 0.00,
        initializer: Union[int, dict] = 0,
        tol_abs: float = gl.SOLVER_TOL_ABS,
        tol_rel: float = gl.SOLVER_TOL_REL,
    ) -> None:
        """
        Solve network pressures
        """
        # Check if we need linear interpolation on composition
        if self.thermo_curvefit and not hasattr(self.composition, "curve_fit_rho"):
            self.composition.make_interps(
                self.min_pressure_bound, self.max_pressure_bound
            )

        self.check_supply_demand_overlap()

        # If not specificed, assign relaxation constant based on scenario type
        if c_relax is None:
            c_relax = SCENARIO_VALUES[self.scenario_type].c_relax

        # Stack up demands
        dn_node_vals = network_util.get_stacked_demand_mdot(network=self)

        n_adj = self.n_nodes - len(self.ignore_nodes)

        # Initialize pressure and set the state of each node
        p_init = network_util.initialize_handler(
            initializer=initializer, network=self, cr_max=cr_max
        )

        for node in self.nodes.values():
            if node.index in self.ignore_nodes:
                continue
            node.pressure = p_init[node.index_adj]

        # Get initial mass flow rate targets
        m_dot_target, dn_nodes, dn_node_adj = network_util.get_flow_targets(
            n_adj=n_adj, network=self
        )

        # Get static properties (e.g., diameter, length)
        pipe_static_props = np.array(
            [pipe.get_pipe_static_properties() for pipe in self.pipes.values()]
        )

        # Make a mask to collect only not ignore nodes
        mask = np.ones(self.n_nodes, dtype=bool)
        mask[self.ignore_nodes] = False

        # Pre-determine connections
        cxns = network_util.make_connections(
            self, shape_pn=(self.n_nodes, len(self.pipes))
        )
        jacobian = scipy.sparse.csr_matrix(
            ([0.0] * len(cxns[0]), (cxns[0], cxns[1])), shape=(n_adj, n_adj)
        )
        # Get all the indices from CSR format
        j_inds = [
            (i, j)
            for i in range(n_adj)
            for j in jacobian.indices[jacobian.indptr[i] : jacobian.indptr[i + 1]]
        ]

        m_dot_prev = [np.nan for _ in self.pipes]

        # Loop
        p_solving = p_init.copy()
        n_iter = 0
        err_abs, err_rel = np.inf, np.inf
        while err_abs > tol_abs or err_rel > tol_rel:
            # Make jacobian and nodal flow based connectivity and flow
            nodal_flow, x_out, m_dot_prev = self.make_jacobian(
                pipe_static_props=pipe_static_props,
                p_solving=p_solving,
                cxns=cxns,
                j_inds=j_inds,
                jacobian_data=jacobian.data,
                m_dot_prev=m_dot_prev,
            )
            # Get difference in flow rate from target
            delta_flow = m_dot_target - nodal_flow[mask]

            if n_iter % 10 == 0:
                j_max = int(np.argmax(np.abs(delta_flow)))
                node_name_dbg = [
                    n for n in self.nodes.values() if n.index_adj == j_max
                ][0].name
                logger.debug(
                    f"[it{n_iter:03}] max|Δṁ| = {delta_flow[j_max]:.4f} kg/s @ {node_name_dbg}"
                )

            # Solve for the change in pressure
            # delta_p = np.linalg.solve(a=jacobian, b=delta_flow)
            delta_p = scipy.sparse.linalg.spsolve(A=jacobian, b=delta_flow)

            # Apply relaxation factor
            p_solving += delta_p / c_relax

            # Check if any values are NAN
            if np.any(np.isnan(p_solving)):
                raise ValueError("NAN pressure")
            # Check if negative pressure. Apply correction to see if it resolves
            if np.any(p_solving < 0):
                ind = np.where(p_solving < 0)
                p_solving[ind] -= (delta_p[ind] / c_relax) * 0.99
                if np.any(p_solving < 0):
                    raise ValueError("Negative pressure")

            # Assign pressures at each node. Skip the ignored nodes
            for node in self.nodes.values():
                if node.index in self.ignore_nodes:
                    continue
                node.pressure = p_solving[node.index_adj]

            n_iter += 1

            if self.composition_tracking:
                x_out = np.clip(x_out, 0, 1)
                # Reassign compositions
                for pipe in self.pipes.values():
                    # Flow is from_node->to_node
                    if pipe.direction == 1:
                        pipe.x_h2 = x_out[pipe.from_node.index]
                    else:
                        pipe.x_h2 = x_out[pipe.to_node.index]
                for node in self.nodes.values():
                    node.x_h2 = x_out[node.index]

                # Update demand nodes mass flow rate based on blend
                m_dot_target = np.zeros(n_adj)
                for dn in self.demand_nodes.values():
                    if dn.node.index in self.ignore_nodes:
                        continue
                    hhv = self.composition.get_hhv(x_out[dn.node.index])
                    m_dot_target[dn.node.index_adj] += dn.recalc_mdot(hhv)
                for sn in self.supply_nodes.values():
                    if not sn.is_pressure_supply:
                        if sn.node.index in self.ignore_nodes:
                            continue
                        # hhv = self.composition.get_curvefit_hhv(x_out[sn.node.index])
                        m_dot_target[sn.node.index_adj] -= sn.flowrate_mdot

                dn_node_vals = network_util.get_stacked_demand_mdot(network=self)

            if self.compressors:
                # Evaluate compressor thermo as vector
                comp_p = network_util.get_compressor_p_and_x(network=self, x_out=x_out)
                comp_h_1, comp_s_1, comp_h_2_s = self.get_h_s_s(
                    x=comp_p[:, 0], p1=comp_p[:, 1], p2=comp_p[:, 2]
                )

                for c_i, comp in enumerate(self.compressors.values()):
                    if comp.from_node.is_supply:
                        fuel_use = comp.get_fuel_use(
                            comp_h_1[c_i],
                            comp_s_1[c_i],
                            comp_h_2_s[c_i],
                            -1 * nodal_flow[comp.to_node.index],
                        )
                        continue
                    from_c = comp.from_node.index_adj
                    comp_flow = -1 * nodal_flow[comp.to_node.index]
                    if not comp.to_node.connections["Pipe"]:
                        # If there is a compressor at the end of the segment
                        comp_flow = dn_node_vals[comp.to_node.name]
                        nodal_flow[comp.to_node.index] = comp_flow
                    elif comp.to_node.is_demand:
                        comp_flow += dn_node_vals[comp.to_node.name]
                        nodal_flow[comp.to_node.index] += comp_flow
                    m_dot_target[from_c] = comp_flow
                    # If using fuel extraction, then reduce mdot after comp
                    fuel_use = comp.get_fuel_use(
                        comp_h_1[c_i], comp_s_1[c_i], comp_h_2_s[c_i], comp_flow
                    )
                    if not comp.fuel_extract:
                        continue
                    m_dot_target[from_c] += fuel_use
                    # m_dot_target[supply_node.node.index] -= fuel_use

            for reg in self.regulators.values():
                from_r = reg.from_node.index_adj
                reg_flow = -1 * nodal_flow[reg.to_node.index]
                if not reg.to_node.connections["Pipe"]:
                    # If there is a regulator at the end of the segment
                    reg_flow = dn_node_vals[reg.to_node.name]
                    nodal_flow[reg.to_node.index] = reg_flow
                # m_dot_target[from_r] = (
                #     m_dot_target[from_r]
                #     + (reg_flow - m_dot_target[from_r]) / c_relax / 5
                # )
                m_dot_target[from_r] = reg_flow
                reg.apply_regulation()

            if n_iter > gl.MAX_ITER:
                raise ValueError(f"Could not converge in {gl.MAX_ITER} iterations")

            err_abs = np.max(np.absolute(delta_flow))
            err_rel = np.max(
                np.absolute(
                    (m_dot_target[dn_node_adj] - nodal_flow[dn_nodes])
                    / m_dot_target[dn_node_adj]
                )
            )

        # Update pipe mass balances
        for p_i, pipe in enumerate(self.pipes.values()):
            pipe.m_dot = m_dot_prev[p_i]

        # Check H2 mass balance
        network_util.check_h2_balance(network=self)

        # Check for any values below minimum pressure
        if np.any(
            self.min_pressure_bound - p_solving
            >= self.min_pressure_bound * low_p_buffer
        ):
            raise ValueError("Pressure below threshold")

        # Check minimum pressure at demands
        for dn in self.demand_nodes.values():
            actual_pa = dn.node.pressure
            required_pa = dn.min_pressure_mpa_g * gl.MPA2PA

            if actual_pa < required_pa:
                raise ValueError(
                    f"Demand node '{dn.name}' at node '{dn.node.name}' "
                    f"has pressure {actual_pa / gl.BAR2PA:.2f} bar, "
                    f"below required minimum of {dn.min_pressure_mpa_g} MPa-g"
                )

        # Assign demand flow rates
        # Some demands stack at a single node, so use same ratio of setpoints
        # to distribute calculated demand flow rate
        stacking_demand = {}
        for dnode in self.demand_nodes.values():
            if dnode.node.index not in stacking_demand.keys():
                stacking_demand[dnode.node.index] = {dnode.name: dnode.flowrate_mdot}
            else:
                stacking_demand[dnode.node.index][dnode.name] = dnode.flowrate_mdot
        for dnode in self.demand_nodes.values():
            dnode.flowrate_mdot_sim = (
                nodal_flow[dnode.node.index]
                * stacking_demand[dnode.node.index][dnode.name]
                / sum(stacking_demand[dnode.node.index].values())
            )

    def make_jacobian(
        self,
        pipe_static_props: npt.NDArray[np.float64],
        p_solving: npt.NDArray[np.float64],
        cxns: tuple,
        j_inds: list[str],
        jacobian_data: npt.NDArray[np.float64],
        m_dot_prev: npt.NDArray[np.float64],
    ) -> tuple:
        """
        Make jacobian matrix for solver
        """
        len_all_nodes = len(self.nodes)
        len_p_supply_nodes = 0
        len_comps = 0

        # Only need to solve to unknown pressures
        # P supplies are known and pressure outlet of compressors is known
        n_nodes = len_all_nodes - len_p_supply_nodes - len_comps
        # n_adj = n_nodes - len(self.ignore_nodes)

        if self.composition_tracking:
            n_nodal = np.zeros(n_nodes, dtype=float)
            x_nodal = np.zeros(n_nodes, dtype=float)

        pipe_dynamic_props = np.array(
            [pipe.get_pipe_dynamic_properties() for pipe in self.pipes.values()]
        )
        p_avg = Pipe.p_avg_vector(
            pipe_dynamic_props[:, Pipe.DYN_PROPS.P_FROM],
            pipe_dynamic_props[:, Pipe.DYN_PROPS.P_TO],
        )

        _, z_avgs, mu, mw = self.get_rho_z_mu(
            pipe_avgs_gauge=p_avg, x_vals=pipe_dynamic_props[:, Pipe.DYN_PROPS.X_H2]
        )

        dm_dp, m_dot_pipe_v = Pipe.get_jacobian_components(
            dyn_props=pipe_dynamic_props,
            static_props=pipe_static_props,
            z_avgs=z_avgs,
            mw=mw,
            mu=mu,
            ff_type=self.ff_type,
            m_dot_prev=m_dot_prev,
        )

        # Generate jacobian
        rows, cols, p_index, coefs, pipe_node_cxns = cxns
        data = coefs * dm_dp[p_index] * p_solving[cols]
        # Triplet → CSR
        # jacobian = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n_adj, n_adj))

        jd = defaultdict(float)
        # Add up duplicates and index by row_col
        for r_i, r in enumerate(rows):
            jd[(r, cols[r_i])] += data[r_i]

        # Swap in place jacobian data
        for i, val in enumerate(j_inds):
            jacobian_data[i] = jd[val]

        # Loop thru pipes
        # Assign nodal flows based on connections
        nodal_flow = pipe_node_cxns @ m_dot_pipe_v

        if self.composition_tracking:
            for p_i, pipe in enumerate(self.pipes.values()):
                pipe.m_dot = m_dot_pipe_v[p_i]

                # Assign nodal flows
                to_node_index = pipe.to_node.index
                from_node_index = pipe.from_node.index

                m_dot_pipe_mw_val = m_dot_pipe_v[p_i] / mw[p_i]
                n_nodal[to_node_index] += max(0, m_dot_pipe_mw_val)
                n_nodal[from_node_index] += max(0, -1 * m_dot_pipe_mw_val)
                x_nodal[to_node_index] += max(0, pipe.x_h2 * m_dot_pipe_mw_val)
                x_nodal[from_node_index] += max(0, -1 * pipe.x_h2 * m_dot_pipe_mw_val)

        if not self.composition_tracking:
            x_out = np.array([node.x_h2 for node in self.nodes.values()])
        else:
            # Get x_h2 out of nodes
            with np.errstate(invalid="ignore"):
                x_out = np.divide(x_nodal, n_nodal)
            for sn in self.supply_nodes.values():
                sn_mdot = sn.mdot
                if sn_mdot == 0:
                    sn_mdot = sum(
                        [dn.flowrate_mdot for dn in self.demand_nodes.values()]
                    )
                n_sn = sn_mdot / sn.mw
                x_existing = x_nodal[sn.node.index]
                if np.isnan(x_existing):
                    x_existing = 0
                x_out[sn.node.index] = (x_existing + sn.blend * n_sn) / (
                    n_nodal[sn.node.index] + n_sn
                )
            for comp in self.compressors.values():
                x_out[comp.to_node.index] = x_out[comp.from_node.index]
            # If there is not a flow into a node, it will register as nan. then overwrite
            # with the existing node value
            nans = np.where(np.isnan(x_out))
            x_h2_nodes = np.array([node.x_h2 for node in self.nodes.values()])
            x_out[nans] = x_h2_nodes[nans]

        return nodal_flow, x_out, m_dot_pipe_v

    def directional_reachability_check(
        self,
        treat_pipes_bidirectional: bool = True,
        require: Literal["demands", "all_nodes"] = "all_nodes",
        sample_n: int = 10,
    ) -> dict:
        """
        Directed reachability from the union of all supply nodes.
        - Pipes: by default treated as bidirectional (two arcs).
        - Compressors / Regulators: treated as ONE-WAY from_node -> to_node.
        """
        node_names = list(self.nodes.keys())
        adj: dict[str, set[str]] = {name: set() for name in node_names}

        for p in self.pipes.values():
            adj[p.from_node.name].add(p.to_node.name)
            adj[p.to_node.name].add(p.from_node.name)

        # Equipment as one-way edges.
        for c in self.compressors.values():
            adj[c.from_node.name].add(c.to_node.name)
        for r in self.regulators.values():
            adj[r.from_node.name].add(r.to_node.name)

        # Seeds: union of all supplies (pressure supplies first, then others).
        pressure_seeds = [
            sn.node.name for sn in self.supply_nodes.values() if sn.is_pressure_supply
        ]
        other_seeds = [
            sn.node.name
            for sn in self.supply_nodes.values()
            if not sn.is_pressure_supply
        ]
        seeds = pressure_seeds + other_seeds

        if not seeds:
            return {
                "ok": False,
                "reason": "no_supply",
                "seeds": [],
                "all_nodes_reachable": False,
                "all_demands_reachable": False,
                "unreachable_nodes_sample": sorted(node_names)[: max(0, sample_n)],
                "unreachable_demands": sorted(
                    [dn.name for dn in self.demand_nodes.values()]
                ),
                "require": require,
            }

        # Multi-source BFS.
        seen: set[str] = set(seeds)
        dq = deque(seeds)
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    dq.append(v)

        # Report for both demands and all nodes.
        unreachable_nodes = sorted([nm for nm in node_names if nm not in seen])
        unreachable_demands = sorted(
            [dn.name for dn in self.demand_nodes.values() if dn.node.name not in seen]
        )

        return {
            "ok": True,
            "all_nodes_reachable": len(unreachable_nodes) == 0,
            "all_demands_reachable": len(unreachable_demands) == 0,
            "unreachable_nodes_sample": unreachable_nodes[: max(0, sample_n)],
            "unreachable_demands": unreachable_demands,
            "require": require,
        }

    def assert_reachable_directed(
        self,
        treat_pipes_bidirectional: bool = True,
        require: Literal["demands", "all_nodes"] = "all_nodes",
    ) -> None:
        rpt = self.directional_reachability_check(
            treat_pipes_bidirectional=treat_pipes_bidirectional,
            require=require,
            sample_n=10,
        )
        if not rpt.get("ok") and rpt.get("reason") == "no_supply":
            raise ValueError(
                "Directed connectivity: no supply node found; cannot check reachability."
            )

        if require == "all_nodes":
            if not rpt["all_nodes_reachable"]:
                raise ValueError(
                    "Directed connectivity failure (all nodes): "
                    f"unreachable_nodes_sample={rpt['unreachable_nodes_sample']}"
                )
        elif require == "demands":
            if not rpt["all_demands_reachable"]:
                raise ValueError(
                    "Directed connectivity failure (demands): "
                    f"unreachable_demands={rpt['unreachable_demands'][:10]}"
                )
        else:
            raise ValueError(f"Unknown reachability requirement: {require}")

    def segment_pipe(self) -> list:
        """
        Segment pipeline network into segments based on compressors, branches, diameter change
        """
        p_supply_node_name = list(self.supply_nodes.values())[0].node.name
        pipe_segments = []
        start_nodes = [p_supply_node_name]

        visited = {n: None for n in self.nodes}

        for ps_i, s_node in enumerate(start_nodes):
            queue = deque([s_node])
            pipes = []
            diameter = None
            DN = None
            comps = []
            p_max_mpa_g = None
            design_pressure_MPa = None
            nodes = []
            demand_nodes = []
            offtake_lengths = [0]

            while queue:
                current_name = queue.popleft()
                current_node = self.nodes[current_name]
                nodes.append(current_node)
                if current_node.is_demand:
                    demand_nodes.extend(
                        [
                            dn
                            for dn in self.demand_nodes.values()
                            if dn.node is current_node
                        ]
                    )
                    offtake_lengths.append(0)

                # If pipe connections, add to segment
                for pipe_conn in current_node.connections["Pipe"]:
                    neighbor = (
                        pipe_conn.to_node
                        if pipe_conn.from_node == current_node
                        else pipe_conn.from_node
                    )
                    visited[current_name] = ps_i

                    if visited[neighbor.name] is None:
                        # Check if diameter changes add to start nodes
                        if diameter is not None and diameter != pipe_conn.diameter_mm:
                            start_nodes.append(neighbor.name)
                            continue
                        queue.append(neighbor.name)

                        # Add to pipe segment values
                        pipes.append(pipe_conn)
                        diameter = pipe_conn.diameter_mm
                        DN = pipe_conn.DN
                        p_max_mpa_g = pipe_conn.p_max_mpa_g
                        design_pressure_MPa = pipe_conn.design_pressure_MPa
                        offtake_lengths[-1] += pipe_conn.length_km

                # If compressor, add to start nodes
                for comp_conn in current_node.connections["Comp"]:
                    neighbor = (
                        comp_conn.to_node
                        if comp_conn.from_node == current_node
                        else comp_conn.from_node
                    )
                    if (
                        neighbor.name not in start_nodes
                        and visited[neighbor.name] is None
                    ):
                        start_nodes.append(neighbor.name)

                        # Add compressor at end of segment
                        comps.append(comp_conn)

            if offtake_lengths[0] == 0:
                offtake_lengths.pop(0)
            pipe_segments.append(
                plc.PipeSegment(
                    pipes=pipes,
                    diameter=diameter,
                    DN=DN,
                    comps=comps,
                    start_node=self.nodes[s_node],
                    p_max_mpa_g=p_max_mpa_g,
                    design_pressure_MPa=design_pressure_MPa,
                    nodes=nodes,
                    demand_nodes=demand_nodes,
                    offtake_lengths=offtake_lengths,
                )
            )

        return pipe_segments

    def pipe_assessment(
        self, ASME_params: ASME_consts, design_option: str | float = "b"
    ) -> None:
        """
        Assess pipe MAOP based on ASME B31.12
        """
        for pipe in self.pipes.values():
            pipe.pipe_assessment(
                design_option=design_option,
                location_class=ASME_params.location_class,
                joint_factor=ASME_params.joint_factor,
                T_derating_factor=ASME_params.T_rating,
            )

    def check_segmentation(self) -> None:
        """
        Unused function to check if a pipe needs further segmentation based on L/D ratio
        """
        seg_max = gl.SEG_MAX
        node_index = len(self.nodes.keys()) - 1
        all_pipe_names = list(self.pipes.keys())
        for pipe_name in all_pipe_names:
            pipe = self.pipes[pipe_name]
            if (pipe.length_km * gl.KM2M) / (pipe.diameter_mm * gl.MM2M) > seg_max:
                lenth_sub_segment = seg_max * (pipe.diameter_mm * gl.MM2M) / gl.KM2M
                n_nodes = int(np.floor(pipe.length_km / lenth_sub_segment))
                lenth_sub_segment = pipe.length_km / (n_nodes + 1)
                from_node = pipe.from_node
                from_node_name_fixed = from_node.name
                from_node.connections["Pipe"].remove(pipe)
                for subseg in range(n_nodes):
                    new_node_name = f"{from_node_name_fixed}_{subseg}"
                    # Keep addng underscript till it is a unique name
                    while new_node_name in self.nodes:
                        new_node_name = f"{new_node_name}_"
                    self.nodes[new_node_name] = plc.Node(
                        name=new_node_name,
                        p_max_mpa_g=from_node.p_max_mpa_g,
                        index=node_index + 1,
                        composition=from_node.composition,
                        _report_out=False,
                    )
                    node_index += 1
                    new_pipe_name = f"{pipe.name}_{subseg}"
                    while new_pipe_name in self.pipes:
                        new_pipe_name = f"{new_pipe_name}_"
                    to_node = self.nodes[new_node_name]
                    self.pipes[new_pipe_name] = plc.Steel_pipe(
                        name=new_pipe_name,
                        from_node=from_node,
                        to_node=to_node,
                        diameter_mm=pipe.diameter_mm,
                        length_km=lenth_sub_segment,
                        roughness_mm=pipe.roughness_mm,
                        thickness_mm=pipe.thickness_mm,
                        rating_code=pipe.rating_code,
                        p_max_mpa_g=pipe.p_max_mpa_g,
                    )
                    self.pipes[new_pipe_name]._parent_pipe = pipe.name

                    from_node = to_node

                length_remaining = lenth_sub_segment
                pipe.from_node = from_node
                pipe.length_km = length_remaining
        self.set_thermo_curvefit(self.thermo_curvefit)
        self.eos = self.eos
        self.assign_nodes()
        self.assign_connections()

    def to_file(self, filename: str) -> None:
        """
        Export hydraulic model results to file
        """

        # File setup
        filename = bp_file_w.check_filename_ext(filename=filename, ext="xlsx")
        workbook = bp_file_w.file_setup(filename=filename)

        bp_network_file.write_info_sheet(workbook=workbook, network=self)
        bp_network_file.write_nodes_sheet(workbook=workbook, network=self)
        bp_network_file.write_pipes_sheet(workbook=workbook, network=self)
        bp_network_file.write_comps_sheet(workbook=workbook, network=self)
        bp_network_file.write_composition_sheet(workbook=workbook, network=self)
        bp_network_file.write_regs_sheet(workbook=workbook, network=self)

        ### Closeout
        bp_file_w.file_closeout(workbook=workbook)

    def set_thermo_curvefit(self, thermo_curvefit: bool) -> None:
        """
        Set thermo curvefit for nodes
        """
        self.thermo_curvefit = thermo_curvefit
        self.composition.thermo_curvefit = thermo_curvefit

    def set_composition_tracking(self, composition_tracking: bool) -> None:
        self.composition_tracking = composition_tracking
        self.composition.set_composition_tracking(composition_tracking)

    def get_rho_z_mu(self, pipe_avgs_gauge: np.ndarray, x_vals: np.ndarray) -> tuple:
        if self.thermo_curvefit:
            mw = self.composition.get_curvefit_mw(x=x_vals)
            rho, z = self.composition.get_curvefit_rho_z(
                p_gauge_pa=pipe_avgs_gauge, x=x_vals, mw=mw
            )
            mu = self.composition.get_curvefit_mu(p_gauge_pa=pipe_avgs_gauge, x=x_vals)
        else:
            rho = np.zeros_like(pipe_avgs_gauge)
            z = np.zeros_like(pipe_avgs_gauge)
            mu = np.zeros_like(pipe_avgs_gauge)

            mw = np.array([pipe.from_node.mw for pipe in self.pipes.values()])
            mu = np.array([pipe.from_node.mu for pipe in self.pipes.values()])
            for p_i, p in enumerate(pipe_avgs_gauge):
                rho[p_i], z[p_i] = plc.eos.get_rz(
                    p_gauge=p,
                    T_K=gl.T_FIXED,
                    X=self.composition,
                    eos=self.eos,
                    mw=mw[p_i],
                )

        return rho, z, mu, mw

    def get_h_s_s(self, x: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple:
        if False:
            h = self.composition.get_curvefit_h(x=x, p_gauge_pa=p1)
            s = self.composition.get_curvefit_s(x=x, p_gauge_pa=p1)
            hs = self.composition.get_curvefit_hs(x=x, p_gauge_pa=p2, s=s)
        else:
            h = np.zeros_like(p1)
            s = np.zeros_like(p1)
            hs = np.zeros_like(p1)
            for i in range(len(h)):
                ctu.gas.TPX = gl.T_FIXED, p1[i] + ct.one_atm, self.composition.x
                h[i] = ctu.gas.h
                s[i] = ctu.gas.s
                try:
                    ctu.gas.SPX = s[i], p2[i] + ct.one_atm, self.composition.x
                    hs[i] = ctu.gas.h
                except ct.CanteraError:
                    ctu.gas.SPX = (
                        s[i],
                        p2[i] * 1.0001 + ct.one_atm,
                        self.composition.x,
                    )
                    hs1_temp = ctu.gas.h
                    ctu.gas.SPX = (
                        s[i],
                        p2[i] * 0.9999 + ct.one_atm,
                        self.composition.x,
                    )
                    hs2_temp = ctu.gas.h
                    hs[i] = (hs1_temp + hs2_temp) / 2

        return h, s, hs

    def check_supply_demand_overlap(self):
        """
        Check if a suply node is also a demand node
        if so, raise error
        """
        supplys = {sn.node.name for sn in self.supply_nodes.values()}
        demands = {dn.node.name for dn in self.demand_nodes.values()}
        if supplys & demands:
            raise NameError("A supply node is also a demand node")

    def get_n_adj(self, index) -> int:
        return self.nodes_clean[index]

    def get_compressor_agg_costs(
        self, design_params: Design_params, costing_params: Costing_params
    ) -> tuple:
        # Get fuel usage
        all_fuel_MW = 0
        all_fuel_elec_kW = 0
        supply_comp_fuel = {"gas": 0, "elec": 0}

        # Get compressor capex of added compressors
        # Get revamped compressor cost
        comp_capex = []
        revamped_comp_capex = []
        supply_comp_capex = 0

        for comp in self.compressors.values():
            # Separate out supply compressor if exists
            if comp.name in "Supply compressor":
                supply_comp_fuel = {
                    "gas": comp.fuel_w * gl.W2MW,
                    "elec": comp.fuel_electric_W / gl.KW2W,
                }
                supply_comp_capex += comp.get_cap_cost(
                    cp=costing_params,
                    to_electric=design_params.existing_comp_elec,
                )
                comp.revamp_cost = 0
                continue
            all_fuel_elec_kW += comp.fuel_electric_W / gl.KW2W
            all_fuel_MW += comp.fuel_w * gl.W2MW

            comp_capex.append(
                comp.get_cap_cost(
                    cp=costing_params,
                    to_electric=design_params.existing_comp_elec,
                )
            )
            revamped_comp_capex.append(
                comp.get_cap_cost(
                    cp=costing_params,
                    revamp=True,
                    to_electric=design_params.existing_comp_elec,
                )
            )
        return (
            all_fuel_MW,
            all_fuel_elec_kW,
            supply_comp_fuel,
            comp_capex,
            revamped_comp_capex,
            supply_comp_capex,
        )

    def get_comp_breakdown(self) -> tuple[list, int, float, float]:
        """
        Get compressor info for results file
        """
        # Format compressors
        comp_breakdown = []
        comp_cost_total = 0
        comp_addl_rating = 0
        supply_comp = 0
        for comp in self.compressors.values():
            comp_breakdown.append(
                (
                    comp.name,
                    comp.shaft_power_MW,
                    comp.original_rating_MW,
                    comp.addl_rating,
                    comp.cost,
                    comp.revamp_cost,
                    comp.fuel_use_MMBTU_hr,
                    comp.fuel_electric_W / gl.KW2W,
                )
            )
            comp_addl_rating += comp.addl_rating
            comp_cost_total += comp.cost
            if comp.name == "Supply compressor" and comp.original_rating_MW == 0:
                supply_comp = 1

        return comp_breakdown, supply_comp, comp_addl_rating, comp_cost_total

    @property
    def ff_type(self):
        return self._ff_type

    @ff_type.setter
    def ff_type(self, ff_type: FF_TYPES = "hofer"):
        ff_lc = ff_type.lower()
        if ff_lc not in get_args(FF_TYPES):
            raise ValueError(
                f"{ff_type} is not a valid friction factor correlation (must be one of {list(get_args(FF_TYPES))})"
            )
        self._ff_type = ff_lc
        for pipe in self.pipes.values():
            pipe.ff_type = ff_lc

    @property
    def parent_pipes(self) -> dict:
        return parent_pipe_helper(self.pipes.values())

    @property
    def eos(self):
        return self._eos

    @eos.setter
    def eos(self, eos: _EOS_OPTIONS = "rk"):
        eos_lc = eos.lower()
        if eos_lc not in get_args(_EOS_OPTIONS):
            raise ValueError(
                f"{eos} is not a valid design option (must be one of {list(get_args(_EOS_OPTIONS))})"
            )
        self._eos = eos
        for node in self.nodes.values():
            node.eos_type = eos
        self.composition.eos_type = eos

    @property
    def capacity_MMBTU_day(self):
        """
        Network capacity, as sum of demands, in MMBTU/day
        """
        return sum([d.flowrate_MMBTU_day for d in self.demand_nodes.values()])

    @property
    def min_pressure_bound(self):
        return SCENARIO_VALUES[self.scenario_type.lower()].min_pressure_pa

    @property
    def max_pressure_bound(self):
        return SCENARIO_VALUES[self.scenario_type.lower()].max_pressure_pa

    @property
    def blend_ratio_energy(self) -> float:
        # Get energy ratio of H2
        _, GCV_H2_MJpsm3 = self.composition.pure_h2_hhv_gcv()
        _, GCV_NG_MJpsm3 = self.composition.pure_ng_hhv_gcv()
        blend = self.composition.x["H2"]

        return (blend * GCV_H2_MJpsm3) / (
            (blend * GCV_H2_MJpsm3) + (1 - blend) * GCV_NG_MJpsm3
        )

    @property
    def velocity_violations(self):
        flag_mach = False
        flag_erosional = False
        for pipe in self.pipes.values():
            if not flag_mach and pipe.mach_number >= 1:
                flag_mach = True
            if not flag_erosional and pipe.erosional_velocity_ASME <= pipe.v_max:
                flag_erosional = True
        return flag_mach, flag_erosional

    @property
    def pressure_init_out(self) -> dict[str, tuple[float, float]]:
        return {node.name: (node.pressure, node.x_h2) for node in self.nodes.values()}

    @property
    def n_nodes(self):
        return len(self.nodes)

    def plot(
        self,
        fig=None,
        ax=None,
        pipe_property="v_avg",
        property_range=None,
        show_demands=True,
        savename=None,
        cbar_range=None,
        pipe_limit_red=np.inf,
        show_plot=True,
        title="",
        pipe_selections=None,
    ):
        raise RuntimeError("Network plotting is not available in version 2.1")
