import pytest

import BlendPATH

wangetal = BlendPATH.BlendPATH_scenario(
    casestudy_name="examples/wangetal2018",
)

INPUTS = {
    "blend": [
        ((-0.1, True), ValueError),
        ((0, False), 0.0),
        ((0.5, False), 0.5),
        ((1, False), 1.0),
        ((1.0, False), 1.0),
        ((1.5, True), ValueError),
    ],
    "location_class": [
        ((-0.1, True), ValueError),
        ((0, True), ValueError),
        ((1.1, False), 1),
        ((1, False), 1),
        ((4, False), 4),
        ((5, True), ValueError),
    ],
    "T_rating": [
        ((-0.1, True), ValueError),
        ((0, True), ValueError),
        ((0.5, False), 0.5),
        ((1, False), 1.0),
        ((1.0, False), 1.0),
        ((1.5, True), ValueError),
    ],
    "joint_factor": [
        ((-0.1, True), ValueError),
        ((0, True), ValueError),
        ((0.5, False), 0.5),
        ((1, False), 1.0),
        ((1.0, False), 1.0),
        ((1.5, True), ValueError),
    ],
    "design_option": [
        (("a", False), "a"),
        (("b", False), "b"),
        (("nfc", False), "nfc"),
        (("no fracture criterion", False), "no fracture criterion"),
        (("c", True), ValueError),
        ((0.3, False), 0.3),
        ((0.82, False), 0.82),
        ((1.1, True), ValueError),
        ((0, True), ValueError),
        ((-0.1, True), ValueError),
    ],
    "ng_price": [
        ((0, False), 0),
        ((5, False), 5),
        ((10, False), 10),
        (({"2015": 5, "2018": 10}, False), {"2015": 5, "2018": 10}),
    ],
    "h2_price": [
        ((0, False), 0),
        ((5, False), 5),
        ((10, False), 10),
        (({"2015": 5, "2018": 10}, False), {"2015": 5, "2018": 10}),
    ],
    "elec_price": [
        ((0, False), 0),
        ((5, False), 5),
        ((10, False), 10),
        (({"2015": 5, "2018": 10}, False), {"2015": 5, "2018": 10}),
    ],
    "region": [
        (("GP", False), "GP"),
        (("gp", False), "GP"),
        (("RM", False), "RM"),
        (("abc", True), ValueError),
    ],
    "design_CR": [
        (([1.2, 1.4, 1.6, 1.8, 2.0], False), [1.2, 1.4, 1.6, 1.8, 2.0]),
        (([1.5], False), [1.5]),
        ((["1.2", 1.3, "1.4"], False), [1.2, 1.3, 1.4]),
        (([-1.2, 1.4], True), ValueError),
        (([-1.2], True), ValueError),
        ((1.2, False), [1.2]),
        ((-1.2, True), ValueError),
    ],
    "final_outlet_pressure_mpa_g": [
        (("1.5", False), 1.5),
        ((3, False), 3),
        ((0, True), ValueError),
        ((-1, True), ValueError),
    ],
    "results_dir": [
        (("./", False), "./"),
    ],
    "eos": [
        (("rk", False), "rk"),
    ],
    "ili_interval": [
        (("3", False), 3),
        ((3, False), 3),
        ((0, True), ValueError),
        ((-1, True), ValueError),
    ],
    "original_pipeline_cost": [
        (("3", False), 3),
        ((3, False), 3),
        ((0, False), 0),
        ((-1, True), ValueError),
    ],
    "new_compressors_electric": [
        ((True, False), True),
        (("True", False), True),
        (("TRUE", False), True),
        (("true", False), True),
        ((1, False), True),
        ((0, False), False),
        (("YES", False), False),
    ],
    "existing_compressors_to_electric": [
        ((True, False), True),
        (("True", False), True),
        (("TRUE", False), True),
        (("true", False), True),
        ((1, False), True),
        ((0, False), False),
        (("YES", False), False),
    ],
    "new_comp_eta_s": [
        ((0.8, False), 0.8),
        (("0.8", False), 0.8),
        ((0, True), ValueError),
        ((-1, True), ValueError),
        ((1.2, True), ValueError),
    ],
    "new_comp_eta_s_elec": [
        ((0.8, False), 0.8),
        (("0.8", False), 0.8),
        ((0, True), ValueError),
        ((-1, True), ValueError),
        ((1.2, True), ValueError),
    ],
    "new_comp_eta_driver": [
        ((0.8, False), 0.8),
        (("0.8", False), 0.8),
        ((0, True), ValueError),
        ((-1, True), ValueError),
        ((1.2, True), ValueError),
    ],
    "new_comp_eta_driver_elec": [
        ((0.8, False), 0.8),
        (("0.8", False), 0.8),
        ((0, True), ValueError),
        ((-1, True), ValueError),
        ((1.2, True), ValueError),
    ],
    "pipe_markup": [
        ((0.8, False), 0.8),
        (("0.8", False), 0.8),
        ((0, False), 0),
        ((-1, True), ValueError),
    ],
    "compressor_markup": [
        ((0.8, False), 0.8),
        (("0.8", False), 0.8),
        ((0, False), 0),
        ((-1, True), ValueError),
    ],
    "thermo_curvefit": [
        ((True, False), True),
        (("True", False), True),
        (("TRUE", False), True),
        (("true", False), True),
        ((1, False), True),
        ((0, False), False),
        (("YES", False), False),
    ],
    "composition_tracking": [
        ((True, False), True),
        (("YES", False), False),
    ],
    "scenario_type": [
        (("transmission", False), "transmission"),
        (("TransmissioN", False), "transmission"),
        (("asff", True), ValueError),
    ],
}


@pytest.mark.parametrize(
    "inputs",
    list(INPUTS.keys()),
)
def test_input_params(inputs):
    param_name = inputs

    for test_case in INPUTS[param_name]:
        val, catch_error = test_case[0]
        outputs = test_case[1]

        params = {param_name: val}

        for i in [True, False]:
            # Check setting through new scenario
            if i:
                if catch_error:
                    with pytest.raises(outputs):
                        BlendPATH.BlendPATH_scenario(
                            casestudy_name="examples/wangetal2018",
                            **params,
                        )
                else:
                    scenario = BlendPATH.BlendPATH_scenario(
                        casestudy_name="examples/wangetal2018", **params
                    )
                    assert getattr(scenario, param_name) == outputs, (
                        f"Setting parameter {param_name} to {val} failed"
                    )
            # Check setting through properties
            else:
                if catch_error:
                    with pytest.raises(outputs):
                        setattr(wangetal, param_name, val)
                else:
                    setattr(wangetal, param_name, val)
                    assert getattr(wangetal, param_name) == outputs, (
                        f"Setting parameter {param_name} to {val} failed"
                    )
