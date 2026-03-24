import pytest

import BlendPATH
import BlendPATH.Global as gl
import BlendPATH.network.pipeline_components as bp_plc

REG_PRES_OUT = 5


@pytest.fixture(scope="module")
def setup_reg_network():
    composition = bp_plc.Composition(
        pure_x={"CH4": 1},
        composition_tracking=False,
        thermo_curvefit=True,
        eos_type="rk",
    )

    nodes = {}
    for i in range(4):
        nodes[f"n{i}"] = bp_plc.Node(
            name=f"n{i}",
            composition=composition,
        )

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
        from_node=nodes["n2"],
        to_node=nodes["n3"],
        name="p2",
        diameter_mm=diam_mm,
        length_km=l_km,
        roughness_mm=ro_mm,
        thickness_mm=th_mm,
        rating_code=grade,
    )

    demand_nodes = {
        "d1": bp_plc.Demand_node(
            name="d1", node=nodes["n3"], flowrate_MW=21743.1429436407
        )
    }

    supply_nodes = {
        "s1": bp_plc.Supply_node(
            name="s1", node=nodes["n0"], pressure_mpa=13.34, blend=0.25
        )
    }

    compressors = {}

    regulators = {
        "reg": bp_plc.Regulator(
            name="reg",
            from_node=nodes["n1"],
            to_node=nodes["n2"],
            pressure_out_mpa_g=REG_PRES_OUT,
        )
    }

    return BlendPATH.BlendPATH_network(
        name="comptracking",
        pipes=pipes,
        nodes=nodes,
        demand_nodes=demand_nodes,
        supply_nodes=supply_nodes,
        compressors=compressors,
        regulators=regulators,
        composition=composition,
        composition_tracking=False,
        thermo_curvefit=True,
        scenario_type="transmission",
        eos="rk",
    )


def test_pressure_at_reg(setup_reg_network):
    setup_reg_network.solve()
    assert setup_reg_network.nodes["n2"].pressure == pytest.approx(
        REG_PRES_OUT * gl.MPA2PA
    )


def test_pressure_at_reg_min(setup_reg_network):
    setup_reg_network.regulators["reg"].pressure_out_mpa_g = 13.34
    setup_reg_network.solve(c_relax=3)
    assert (
        setup_reg_network.nodes["n2"].pressure == setup_reg_network.nodes["n1"].pressure
    )
