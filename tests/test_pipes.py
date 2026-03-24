import numpy as np
import pytest

from BlendPATH.network import pipeline_components as bp_plc

x = bp_plc.Composition(
    pure_x={"CH4": 1},
    composition_tracking=False,
    thermo_curvefit=False,
    eos_type="rk",
)
node1 = bp_plc.Node(
    name="pipe_fm_node",
    composition=x,
)
node2 = bp_plc.Node(name="pipe_to_node", composition=x)


@pytest.mark.parametrize(
    "inputs,outputs",
    [
        ((13.7 - 2 * 1.65, 1.65), (8, 0.25, "S 10S")),
        ((48.3 - 2 * 1.65, 1.65), (40, 1.5, "S 5S")),
        ((355.6 - 2 * 4.78, 4.78), (350, 14, "S 10S")),
        ((1016 - 2 * 9.53, 9.53), (1000, 40, "S Std")),
    ],
)
def test_steel_pipe_geometry(inputs, outputs):
    pipe_diam_mm, pipe_th_mm = inputs
    dn, nps, sch = outputs
    pipe = bp_plc.Steel_pipe(
        from_node=node1,
        to_node=node2,
        rating_code="X70",
        diameter_mm=pipe_diam_mm,
        thickness_mm=pipe_th_mm,
    )

    assert pipe.diameter_mm == pipe_diam_mm, "Inner diameter"
    assert pipe.diameter_out_mm == pipe_diam_mm + 2 * pipe_th_mm, "Outer diameter"
    assert pipe.A_m2 == pytest.approx(np.pi * (pipe_diam_mm / 1000) ** 2 / 4), (
        "Cross section area"
    )
    assert pipe.dimension_ratio == round(
        (pipe_diam_mm + 2 * pipe_th_mm) / pipe_th_mm, 2
    ), "Dimension ratio"
    assert pipe.DN == dn, "DN"
    assert pipe.NPS == nps, "NPS"
    assert pipe.material == "steel", "Material"
    assert pipe.schedule == sch, "Schedule"


def test_steel_pipe_design_pressure():
    pipe = bp_plc.Steel_pipe(
        from_node=node1,
        to_node=node2,
        rating_code="X70",
        diameter_mm=1016,
        thickness_mm=10,
    )
    pipe.pipe_assessment(
        design_option=0.72, location_class=1, joint_factor=1, T_derating_factor=1
    )
    assert pipe.design_pressure_MPa == pytest.approx(6.979688)
