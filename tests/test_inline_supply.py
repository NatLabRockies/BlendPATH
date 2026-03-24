import numpy as np
import pytest

import BlendPATH
import BlendPATH.network.pipeline_components as bp_plc


@pytest.fixture(scope="module")
def setup_comp_tracking_network():
    composition = bp_plc.Composition(
        pure_x={"CH4": 1},
        composition_tracking=True,
        thermo_curvefit=True,
        eos_type="rk",
    )

    nodes = {}
    for i in range(3):
        nodes[f"n{i}"] = bp_plc.Node(name=f"n{i}", composition=composition)

    pipes = {}
    diam_mm = 882.65
    l_km = 20
    ro_mm = 0.006350
    th_mm = 15.875
    grade = "X70"
    pipes["p1"] = bp_plc.Pipe(
        from_node=nodes["n0"],
        to_node=nodes["n1"],
        name="p1",
        diameter_mm=diam_mm,
        length_km=l_km,
        roughness_mm=ro_mm,
        thickness_mm=th_mm,
        rating_code=grade,
    )
    pipes["p2"] = bp_plc.Pipe(
        from_node=nodes["n1"],
        to_node=nodes["n2"],
        name="p2",
        diameter_mm=diam_mm,
        length_km=l_km,
        roughness_mm=ro_mm,
        thickness_mm=th_mm,
        rating_code=grade,
    )

    demand_nodes = {
        "d1": bp_plc.Demand_node(name="d1", node=nodes["n2"], flowrate_MW=20000)
    }

    supply_nodes = {
        "s1": bp_plc.Supply_node(
            name="s1", node=nodes["n0"], pressure_mpa=13.34, blend=0.25
        ),
        "s2": bp_plc.Supply_node(
            name="s2", node=nodes["n1"], flowrate_MW=10000, blend=0.75
        ),
    }

    compressors = {}

    return BlendPATH.BlendPATH_network(
        name="comptracking",
        pipes=pipes,
        nodes=nodes,
        demand_nodes=demand_nodes,
        supply_nodes=supply_nodes,
        compressors=compressors,
        composition=composition,
        composition_tracking=True,
        thermo_curvefit=True,
        scenario_type="transmission",
        eos="rk",
    )


def test_inline_supply_values(setup_comp_tracking_network):
    setup_comp_tracking_network.solve()
    assert setup_comp_tracking_network.supply_nodes["s2"].mdot == pytest.approx(
        setup_comp_tracking_network.supply_nodes["s2"].flowrate_MW / 79.1470612507175
    ), "Supply node flow rate is incorrect"

    assert setup_comp_tracking_network.nodes["n2"].x_h2 == pytest.approx(
        np.float64(0.5643874114023298)
    ), "H2 concentation at mixed node is incorrect"
