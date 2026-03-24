import pytest

from BlendPATH import Global as gl
from BlendPATH.network import pipeline_components as bp_plc

P_IN = 3000000
P_OUT = 6000000


@pytest.fixture()
def setup_compressor():
    x = bp_plc.Composition(
        pure_x={"CH4": 1},
        composition_tracking=False,
        thermo_curvefit=True,
        eos_type="rk",
    )

    node1 = bp_plc.Node(
        name="comp_fm_node",
        composition=x,
        pressure=P_IN,
    )
    node2 = bp_plc.Node(
        name="comp_to_node",
        composition=x,
    )
    return bp_plc.Compressor(
        from_node=node1, to_node=node2, pressure_out_mpa_g=P_OUT / gl.MPA2PA
    )


def test_compressor_pressures(setup_compressor):
    assert setup_compressor.from_node.pressure == P_IN
    assert setup_compressor.to_node.pressure == P_OUT
    assert setup_compressor.compression_ratio == P_OUT / P_IN
