import pytest

import BlendPATH

PL_LCOT = 0.40540469765775644
AC_LCOT = 0.7064499137683713
DR_LCOT = 0.44095529245361353
NEWH2_LCOT = 0.716540816801485


@pytest.fixture(scope="module")
def setup_wang_cases():
    wangetal = BlendPATH.BlendPATH_scenario(
        casestudy_name="examples/wangetal2018",
        thermo_curvefit=True,
        composition_tracking=True,
        ng_price=7.39,
        elec_price=0.07,
    )
    wangetal.update_design_option(design_option="nfc")
    wangetal.blendH2(blend=0.5)
    return wangetal


def test_wang_cases_TC_true_CT_true():
    wangetal = BlendPATH.BlendPATH_scenario(
        casestudy_name="examples/wangetal2018",
        thermo_curvefit=True,
        composition_tracking=True,
        ng_price=7.39,
        elec_price=0.07,
    )
    wangetal.update_design_option(design_option="nfc")
    wangetal.blendH2(blend=0.5)
    assert wangetal.run_mod("pl", allow_compressor_bypass=False) == pytest.approx(
        PL_LCOT, rel=1e-4
    ), "PL method is incorrect"
    assert wangetal.run_mod("ac", allow_compressor_bypass=False) == pytest.approx(
        AC_LCOT, rel=1e-4
    ), "AC method is incorrect"
    assert wangetal.run_mod("dr", allow_compressor_bypass=False) == pytest.approx(
        DR_LCOT, rel=1e-4
    ), "DR method is incorrect"
    assert wangetal.run_mod("newh2") == pytest.approx(NEWH2_LCOT, rel=1e-4), (
        "New h2 method is incorrect"
    )


def test_wang_cases_TC_false_CT_true():
    wangetal = BlendPATH.BlendPATH_scenario(
        casestudy_name="examples/wangetal2018",
        thermo_curvefit=False,
        composition_tracking=True,
        ng_price=7.39,
        elec_price=0.07,
    )
    wangetal.update_design_option(design_option="nfc")
    wangetal.blendH2(blend=0.5)
    assert wangetal.run_mod("pl", allow_compressor_bypass=False) == pytest.approx(
        PL_LCOT, rel=1e-4
    ), "PL method is incorrect"
    assert wangetal.run_mod("ac", allow_compressor_bypass=False) == pytest.approx(
        AC_LCOT, rel=1e-4
    ), "AC method is incorrect"
    assert wangetal.run_mod("dr", allow_compressor_bypass=False) == pytest.approx(
        DR_LCOT, rel=1e-4
    ), "DR method is incorrect"
    assert wangetal.run_mod("newh2") == pytest.approx(NEWH2_LCOT, rel=1e-4), (
        "New h2 method is incorrect"
    )


def test_wang_cases_TC_false_CT_false():
    wangetal = BlendPATH.BlendPATH_scenario(
        casestudy_name="examples/wangetal2018",
        thermo_curvefit=False,
        composition_tracking=False,
        ng_price=7.39,
        elec_price=0.07,
    )
    wangetal.update_design_option(design_option="nfc")
    wangetal.blendH2(blend=0.5)
    assert wangetal.run_mod("pl", allow_compressor_bypass=False) == pytest.approx(
        PL_LCOT, rel=1e-4
    ), "PL method is incorrect"
    assert wangetal.run_mod("ac", allow_compressor_bypass=False) == pytest.approx(
        AC_LCOT, rel=1e-4
    ), "AC method is incorrect"
    assert wangetal.run_mod("dr", allow_compressor_bypass=False) == pytest.approx(
        DR_LCOT, rel=1e-4
    ), "DR method is incorrect"
    assert wangetal.run_mod("newh2") == pytest.approx(NEWH2_LCOT, rel=1e-4), (
        "New h2 method is incorrect"
    )


def test_wang_cases_TC_true_CT_false():
    wangetal = BlendPATH.BlendPATH_scenario(
        casestudy_name="examples/wangetal2018",
        thermo_curvefit=True,
        composition_tracking=False,
        ng_price=7.39,
        elec_price=0.07,
    )
    wangetal.update_design_option(design_option="nfc")
    wangetal.blendH2(blend=0.5)
    assert wangetal.run_mod("pl", allow_compressor_bypass=False) == pytest.approx(
        PL_LCOT, rel=1e-4
    ), "PL method is incorrect"
    assert wangetal.run_mod("ac", allow_compressor_bypass=False) == pytest.approx(
        AC_LCOT, rel=1e-4
    ), "AC method is incorrect"
    assert wangetal.run_mod("dr", allow_compressor_bypass=False) == pytest.approx(
        DR_LCOT, rel=1e-4
    ), "DR method is incorrect"
    assert wangetal.run_mod("newh2") == pytest.approx(NEWH2_LCOT, rel=1e-4), (
        "New h2 method is incorrect"
    )


@pytest.mark.parametrize(
    "inputs",
    [50, -1],
)
def test_blendH2_values(setup_wang_cases, inputs):
    with pytest.raises(ValueError):
        setup_wang_cases.blendH2(inputs)


def test_electricity_dict():
    wangetal_2 = BlendPATH.BlendPATH_scenario(
        casestudy_name="examples/wangetal2018",
        new_compressors_electric=False,
        existing_compressors_to_electric=False,
        thermo_curvefit=True,
        composition_tracking=False,
        ng_price=7.39,
        elec_price={
            x: 0.07 * 1.025 ** (i - 1) for i, x in enumerate(range(2019, 2020 + 70))
        },
    )
    wangetal_2.update_design_option(design_option="nfc")
    wangetal_2.blendH2(blend=0.5)

    lcot_2 = wangetal_2.run_mod("pl", allow_compressor_bypass=False)
    assert PL_LCOT == pytest.approx(lcot_2, rel=1e-5)


def test_ng_dict():
    wangetal_2 = BlendPATH.BlendPATH_scenario(
        casestudy_name="examples/wangetal2018",
        new_compressors_electric=False,
        existing_compressors_to_electric=False,
        thermo_curvefit=True,
        composition_tracking=False,
        elec_price=0.07,
        ng_price={
            x: 7.39 * 1.025 ** (i - 1) for i, x in enumerate(range(2019, 2020 + 70))
        },
    )
    wangetal_2.update_design_option(design_option="nfc")
    wangetal_2.blendH2(blend=0.5)

    lcot_2 = wangetal_2.run_mod("pl", allow_compressor_bypass=False)
    assert PL_LCOT == pytest.approx(lcot_2, rel=1e-5)


def test_d2e():
    wangetal = BlendPATH.BlendPATH_scenario(
        casestudy_name="examples/wangetal2018",
        financial_overrides={"debt equity ratio of initial financing": 1.5},
        design_option="nfc",
        blend=0.5,
    )
    assert wangetal.run_mod("pl", allow_compressor_bypass=False) == pytest.approx(
        0.36066579174158797, rel=1e-5
    )


def test_invalid_financial_override_name():
    with pytest.raises(ValueError):
        wangetal = BlendPATH.BlendPATH_scenario(
            casestudy_name="examples/wangetal2018",
            financial_overrides={"asdfasdf": 1.5},
            design_option="nfc",
            blend=0.5,
        )
        wangetal.run_mod("pl", allow_compressor_bypass=False)


@pytest.mark.parametrize(
    "inputs",
    ["abcd", 50],
)
def test_design_option_entries(setup_wang_cases, inputs):
    with pytest.raises(ValueError):
        setup_wang_cases.update_design_option(design_option=inputs)


def test_design_option_match_b(setup_wang_cases):
    setup_wang_cases.update_design_option(design_option="b")
    lcot_b = setup_wang_cases.run_mod("pl", allow_compressor_bypass=False)
    setup_wang_cases.update_design_option(design_option=0.72)
    lcot_72 = setup_wang_cases.run_mod("pl", allow_compressor_bypass=False)
    assert lcot_b == pytest.approx(lcot_72)
