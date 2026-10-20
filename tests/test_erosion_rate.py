import pytest

import BlendPATH.util.erosion_rate as bp_erosion


def test_velocity_from_erosion():
    assert bp_erosion.erosion_rate_velocity(
        diameter_mm=569,
        thickness_mm=31,
        fraction_of_thickness=0.1,
        years=50,
        rho_g=56.93,
        material="steel",
        loading_ppm_w=35.6753908,
    ) == pytest.approx(52.469061844247484)


@pytest.mark.parametrize(
    "inputs, outputs",
    [
        ((-45, True), ValueError),
        ((5, False), 0.559892101),
        ((30, False), 0.992543276),
        ((60, False), 0.893678595),
        ((90, False), 0.568510726),
        ((180, True), ValueError),
    ],
)
def test_get_F_ductility(inputs, outputs):
    alpha, catch_error = inputs
    if catch_error:
        with pytest.raises(ValueError):
            bp_erosion.get_F_ductility(alpha=alpha)
    else:
        assert bp_erosion.get_F_ductility(alpha=alpha) == pytest.approx(outputs)


@pytest.mark.parametrize(
    "inputs, outputs",
    [
        ((147, 0.100, 0.03, 0.04, "steel", 30), 0.0000000988601),
        ((200, 0.500, 0.03, 0.1, "steel", 15), 0.0000000002448),
        ((200, 0.500, 0.03, 0.1, "polymer", 15), 0.0000000004851),
    ],
)
def test_erosion_rate(inputs, outputs):
    diameter_mm, length_km, mixture_kg_s, rho_g, material, alpha = inputs
    assert bp_erosion.erosion_rate(
        diameter_mm=diameter_mm,
        length_km=length_km,
        mixture_kg_s=mixture_kg_s,
        rho_g=rho_g,
        material=material,
        alpha=alpha,
    ) == pytest.approx(outputs)
