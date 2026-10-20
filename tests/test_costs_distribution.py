import numpy as np
import pytest

import BlendPATH.costing.distribution_network_nonpipe_costs as bp_dist_np_costs
import BlendPATH.costing.pipe_costs.polymer_pipe_costs as bp_dist_polymer_costs


@pytest.mark.parametrize(
    "inputs, outputs",
    [
        ((275, 2), 280.7),
        ((276, 25), 700.73),
        ((277, 59), 700.73),
        ((940, 61), 2_526.12),
        ((941, 65), 4_847.83),
        ((50_000, 100), 29_165.57),
    ],
)
def test_msa_cost(inputs, outputs):
    flowrate_cfh, pressure_psig = inputs
    assert (
        bp_dist_np_costs.get_msa_cost(
            flowrate_cfh=np.asarray([flowrate_cfh]),
            pressure_psig=np.asarray([pressure_psig]),
        )
        == outputs
    )


nps_diam_in_test_value = 2


@pytest.mark.parametrize(
    "inputs, outputs",
    [
        (
            ("socal", nps_diam_in_test_value, False),
            bp_dist_polymer_costs.polymer_pipe_cost_socalgas(
                nps_diam_in=nps_diam_in_test_value
            ),
        ),
        (
            ("nexant_rural", nps_diam_in_test_value, False),
            bp_dist_polymer_costs.polymer_pipe_cost_nexant_rural(
                nps_diam_in=nps_diam_in_test_value
            ),
        ),
        (
            ("nexant_suburban", nps_diam_in_test_value, False),
            bp_dist_polymer_costs.polymer_pipe_cost_nexant_suburban(
                nps_diam_in=nps_diam_in_test_value
            ),
        ),
        (
            ("nexant_urban", nps_diam_in_test_value, False),
            bp_dist_polymer_costs.polymer_pipe_cost_nexant_urban(
                nps_diam_in=nps_diam_in_test_value
            ),
        ),
        (
            ("nexant_downtown", nps_diam_in_test_value, False),
            bp_dist_polymer_costs.polymer_pipe_cost_nexant_downtown(
                nps_diam_in=nps_diam_in_test_value
            ),
        ),
        (("unknown", 2, True), ValueError),
        (("unknown", -1, True), ValueError),
    ],
)
def test_polymer_pipe_costs(inputs, outputs):
    cost_type, nps_diam_in, throwerror = inputs
    if throwerror:
        with pytest.raises(ValueError):
            bp_dist_polymer_costs.get_polymer_pipe_cost_per_ft(
                cost_type=cost_type, nps_diam_in=nps_diam_in
            )
    else:
        assert (
            bp_dist_polymer_costs.get_polymer_pipe_cost_per_ft(
                cost_type=cost_type, nps_diam_in=nps_diam_in
            )
            == outputs
        )
