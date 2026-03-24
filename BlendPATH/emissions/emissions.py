from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import cantera as ct
import numpy as np

from BlendPATH.emissions.emissions_params import EmissionsParams
from BlendPATH.network.pipeline_components import Composition
from BlendPATH.network.pipeline_components.eos import get_rz

if TYPE_CHECKING:
    from BlendPATH.network import BlendPATH_network

logger = logging.getLogger(__name__)

_SEC_PER_YEAR = 60 * 60 * 24 * 365
_G2MMT = 1e-12
_KG_PER_MMT = 1e9
_MW_H2 = 2.016
_HHV_H2 = 141.818  # MJ/kg


def _dbg(debug: bool, msg: str) -> None:
    logger.debug(msg)


def _to_float_scalar(v: object) -> float:
    arr = np.asarray(v)
    return float(arr.reshape(-1)[0])


def _eval_curvefit(val: object, x: float) -> float:
    try:
        return _to_float_scalar(val(x)) if callable(val) else _to_float_scalar(val)
    except Exception:
        return 0.0


def _get_hhv_mj_per_kg(network) -> float:
    comp = network.composition
    x_h2 = float(comp.x.get("H2", 0.0))
    return _to_float_scalar(comp.get_hhv(x=np.array([x_h2])))


def _get_mw_kg_per_kmol(network) -> float:
    comp = network.composition
    x_h2 = float(comp.x.get("H2", 0.0))
    return _to_float_scalar(comp.get_mw(p_gauge_pa=101325, x=np.array([x_h2])))


def _safe_sum_supply_mdot_kg_s(network: "BlendPATH_network") -> float:
    mdot = 0.0
    for sn in network.supply_nodes.values():
        md = getattr(sn, "mdot", 0.0) or 0.0
        mdot += abs(float(md))
    return mdot


def _safe_total_pipe_length_km(network: "BlendPATH_network") -> float:
    return float(
        sum(float(getattr(p, "length_km", 0.0) or 0.0) for p in network.pipes.values())
    )


def _safe_sum_electric_compressor_kw(network: "BlendPATH_network") -> float:
    return (
        float(
            sum(
                float(getattr(c, "fuel_electric_W", 0.0) or 0.0)
                for c in network.compressors.values()
            )
        )
        / 1000.0
    )


def _safe_sum_compressor_fuel_w(network: "BlendPATH_network") -> float:
    return float(
        sum(
            float(getattr(c, "fuel_w", 0.0) or 0.0)
            for c in network.compressors.values()
        )
    )


def _is_distribution_network(network: "BlendPATH_network") -> bool:
    scen = getattr(network, "scenario_type", None)
    if scen is None:
        return False
    scen_val = getattr(scen, "value", scen)
    return "distribution" in str(scen_val).lower()


def _normalize_case_study_path(case_study: str | None) -> str | None:
    if not case_study:
        return None

    case_study = str(case_study)

    candidates = [
        case_study,
        os.path.join("case_studies", case_study),
    ]

    for cand in candidates:
        if os.path.exists(os.path.join(cand, "network_design.xlsx")):
            return cand

    return case_study


def _resolve_case_study(
    network: "BlendPATH_network",
    case_study: str | None,
) -> str | None:
    if case_study:
        return _normalize_case_study_path(case_study)

    for attr in ("casestudy_name", "case_study", "network_name"):
        val = getattr(network, attr, None)
        if val:
            return _normalize_case_study_path(str(val))

    parent = getattr(network, "parent", None)
    if parent is not None:
        for attr in ("casestudy_name", "case_study", "network_name"):
            val = getattr(parent, attr, None)
            if val:
                return _normalize_case_study_path(str(val))

    return None


def _resolve_blend(
    network: "BlendPATH_network",
    blend_override: float | None,
) -> float:
    if blend_override is not None:
        return max(0.0, min(1.0, float(blend_override)))
    return max(
        0.0, min(1.0, float(getattr(network.composition, "x", {}).get("H2", 0.0)))
    )


def _get_pure_ng_props(
    comp,
    case_study: str | None = None,
) -> tuple[float | None, float | None, float | None]:
    """
    Keep the direct curve_fit_* access here.
    This path is used for upstream/combustion parity.
    """
    try:
        case_study = _normalize_case_study_path(case_study)

        if case_study:
            import co2_reduction_analysis

            comp_dict = co2_reduction_analysis.read_file_calamine_composition(
                f"{case_study}/network_design.xlsx",
                "COMPOSITION",
            )
            comp_dict = {k: v for k, v in comp_dict.items() if k != "H2"}
        else:
            x_no_h2 = getattr(comp, "x_no_h2", None)
            if not x_no_h2:
                return None, None, None
            comp_dict = {k: v for k, v in x_no_h2.items() if k != "H2"}
            if (
                not comp_dict
                or sum([molefrac for molefrac in comp_dict.values()]) == 0.0
            ):
                comp_dict = x_no_h2

        comp_pure = Composition(pure_x=comp_dict, composition_tracking=False)

        hhv_pure = float(comp_pure.curve_fit_hhv)
        mw_pure = float(comp_pure.curve_fit_mw)

        rho_pure, _ = get_rz(
            p_gauge=0,
            T_K=298.15,
            X=comp_pure,
            eos="rk",
            mw=mw_pure,
        )
        return hhv_pure, mw_pure, float(rho_pure)

    except Exception as e:
        print(f"PURE_NG_FAILED: {e}")
        return None, None, None


def _build_fugitive_basis(
    network: "BlendPATH_network",
    case_study: str | None = None,
    blend_override: float | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Important:
    fugitive should not use network.composition directly for rho.
    Rebuild a non-tracking composition from the workbook/x_no_h2 path.
    """
    case_study = _resolve_case_study(network, case_study)
    blend = _resolve_blend(network, blend_override)

    comp_dict: dict[str, float] | None = None
    source = None

    if case_study:
        import co2_reduction_analysis

        comp_dict = co2_reduction_analysis.read_file_calamine_composition(
            f"{case_study}/network_design.xlsx",
            "COMPOSITION",
        )
        source = f"{case_study}/network_design.xlsx::COMPOSITION"

    if not comp_dict:
        x_no_h2 = getattr(getattr(network, "composition", None), "x_no_h2", None)
        if x_no_h2:
            comp_dict = dict(x_no_h2)
            source = "network.composition.x_no_h2"

    if not comp_dict:
        raise ValueError(
            "Could not build fugitive composition basis. "
            "Pass case_study or ensure network.composition.x_no_h2 exists."
        )

    comp_dict = {k: v for k, v in comp_dict.items() if k != "H2"}
    if not comp_dict or sum([molefrac for molefrac in comp_dict.values()]) == 0.0:
        comp_dict = x_no_h2

    comp_pure = Composition(pure_x=comp_dict, composition_tracking=False)
    hhv_pure = float(comp_pure.curve_fit_hhv)
    mw_pure = float(comp_pure.curve_fit_mw)

    rho_pure, _ = get_rz(
        p_gauge=0,
        T_K=298.15,
        X=comp_pure,
        eos="rk",
        mw=mw_pure,
    )
    rho_pure = float(rho_pure)

    comp_blend = Composition(pure_x=comp_dict, composition_tracking=False)
    comp_blend.blendH2(blend)

    # Do not swap this to get_curvefit_*.
    # The direct attribute behavior is what matches legacy fugitive.
    hhv_blend = float(comp_blend.curve_fit_hhv)
    mw_blend = float(comp_blend.curve_fit_mw)

    rho_blend, _ = get_rz(
        p_gauge=0,
        T_K=298.15,
        X=comp_blend,
        eos="RK",
        mw=mw_blend,
    )
    rho_blend = float(rho_blend)

    blend_actual = float(comp_blend.x.get("H2", blend))

    _dbg(
        debug,
        (
            "EMISSIONS_FUG_COMP_DEBUG | "
            f"case_study={case_study} | "
            f"source={source} | "
            f"blend_requested={blend:.12f} | "
            f"blend_actual={blend_actual:.12f} | "
            f"hhv_pure={hhv_pure:.12f} | "
            f"mw_pure={mw_pure:.12f} | "
            f"rho_pure={rho_pure:.12f} | "
            f"hhv_blend={hhv_blend:.12f} | "
            f"mw_blend={mw_blend:.12f} | "
            f"rho_blend={rho_blend:.12f}"
        ),
    )

    return {
        "case_study": case_study,
        "source": source,
        "blend_requested": blend,
        "blend_actual": blend_actual,
        "composition": comp_blend,
        "hhv_pure_MJ_per_kg": hhv_pure,
        "mw_pure_kg_per_kmol": mw_pure,
        "rho_pure_kg_per_m3": rho_pure,
        "hhv_blend_MJ_per_kg": hhv_blend,
        "mw_blend_kg_per_kmol": mw_blend,
        "rho_blend_kg_per_m3": rho_blend,
    }


def _calc_fugitive_details(
    network: "BlendPATH_network",
    p: EmissionsParams,
    mdot_supply_kg_s: float,
    km_pipe_override: float | None = None,
    case_study: str | None = None,
    blend_override: float | None = None,
    debug: bool = False,
) -> tuple[float, dict[str, Any]]:
    km_pipe = (
        float(km_pipe_override)
        if km_pipe_override is not None
        else _safe_total_pipe_length_km(network)
    )

    if km_pipe <= 0 or mdot_supply_kg_s <= 0:
        return 0.0, {
            "is_distribution": _is_distribution_network(network),
            "leak_basis_mode": None,
            "leak_ch4_base": 0.0,
            "leak_factor": 0.0,
            "vol_nm3_s": 0.0,
            "mass_frac_h2": 0.0,
            "leak_rate_m3_s": 0.0,
            "leak_rate_kg_s": 0.0,
            "leak_h2_g_s": 0.0,
            "leak_ng_g_s": 0.0,
        }

    basis = _build_fugitive_basis(
        network=network,
        case_study=case_study,
        blend_override=blend_override,
        debug=debug,
    )

    composition = basis["composition"]
    mw_blend = float(basis["mw_blend_kg_per_kmol"])
    rho_pure = float(basis["rho_pure_kg_per_m3"])
    rho_blend = float(basis["rho_blend_kg_per_m3"])

    is_dist = _is_distribution_network(network)
    if is_dist and getattr(p, "leak_dist_CH4_m3_per_m3", None) is not None:
        leak_ch4_base = float(p.leak_dist_CH4_m3_per_m3)
        leak_basis_mode = "distribution"
    else:
        leak_ch4_base = float(p.leak_trans_CH4_m3_per_m3_per_km) * km_pipe
        leak_basis_mode = "transmission"

    leak_factor = (
        leak_ch4_base * (rho_pure / rho_blend) ** 0.5
        if rho_blend > 0
        else leak_ch4_base
    )

    vol_nm3_s = mdot_supply_kg_s * ct.gas_constant * 273.15 / ct.one_atm / mw_blend
    leak_rate_m3_s = leak_factor * vol_nm3_s
    leak_rate_kg_s = leak_rate_m3_s * rho_blend

    mole_frac_h2 = max(
        0.0, min(1.0, float(composition.x.get("H2", basis["blend_actual"])))
    )
    mass_frac_h2 = mole_frac_h2 * _MW_H2 / mw_blend

    leak_h2_g_s = leak_rate_kg_s * mass_frac_h2 * 1000.0
    leak_ng_g_s = leak_rate_kg_s * (1.0 - mass_frac_h2) * 1000.0

    gco2e_yr = (
        leak_h2_g_s * float(p.EF_FUG_H2_gCO2e_per_gH2)
        + leak_ng_g_s * float(p.EF_FUG_NG_gCO2e_per_gCH4)
    ) * _SEC_PER_YEAR

    fugitive_mmt_yr = gco2e_yr * _G2MMT

    fug_debug = {
        "is_distribution": is_dist,
        "leak_basis_mode": leak_basis_mode,
        "leak_ch4_base": leak_ch4_base,
        "leak_factor": leak_factor,
        "vol_nm3_s": vol_nm3_s,
        "mass_frac_h2": mass_frac_h2,
        "leak_rate_m3_s": leak_rate_m3_s,
        "leak_rate_kg_s": leak_rate_kg_s,
        "leak_h2_g_s": leak_h2_g_s,
        "leak_ng_g_s": leak_ng_g_s,
        "fug_case_study": basis["case_study"],
        "fug_source": basis["source"],
        "fug_blend_requested": basis["blend_requested"],
        "fug_blend_actual": basis["blend_actual"],
        "fug_hhv_pure_MJ_per_kg": basis["hhv_pure_MJ_per_kg"],
        "fug_mw_pure_kg_per_kmol": basis["mw_pure_kg_per_kmol"],
        "fug_rho_pure_kg_per_m3": basis["rho_pure_kg_per_m3"],
        "fug_hhv_blend_MJ_per_kg": basis["hhv_blend_MJ_per_kg"],
        "fug_mw_blend_kg_per_kmol": basis["mw_blend_kg_per_kmol"],
        "fug_rho_blend_kg_per_m3": basis["rho_blend_kg_per_m3"],
    }

    _dbg(
        debug,
        (
            "EMISSIONS_FUG_DEBUG | "
            f"is_distribution={is_dist} | "
            f"leak_basis_mode={leak_basis_mode} | "
            f"km_pipe={km_pipe:.12f} | "
            f"mdot_supply_kg_s={mdot_supply_kg_s:.12f} | "
            f"rho_pure={rho_pure:.12f} | "
            f"rho_blend={rho_blend:.12f} | "
            f"leak_ch4_base={leak_ch4_base:.12e} | "
            f"leak_factor={leak_factor:.12e} | "
            f"vol_nm3_s={vol_nm3_s:.12f} | "
            f"mass_frac_h2={mass_frac_h2:.12f} | "
            f"fugitive_mmt_yr={fugitive_mmt_yr:.12f}"
        ),
    )

    return fugitive_mmt_yr, fug_debug


def _calc_fugitive_mmt_per_yr(
    network: "BlendPATH_network",
    p: EmissionsParams,
    mdot_supply_kg_s: float,
    km_pipe_override: float | None = None,
    case_study: str | None = None,
    blend_override: float | None = None,
    debug: bool = False,
) -> float:
    fugitive_mmt_yr, _ = _calc_fugitive_details(
        network=network,
        p=p,
        mdot_supply_kg_s=mdot_supply_kg_s,
        km_pipe_override=km_pipe_override,
        case_study=case_study,
        blend_override=blend_override,
        debug=debug,
    )
    return fugitive_mmt_yr


def calc_emissions_intensity(
    network: "BlendPATH_network",
    emissions_params: EmissionsParams,
    mdot_override: float | None = None,
    km_pipe_override: float | None = None,
    case_study: str | None = None,
    blend_override: float | None = None,
    elec_comp_override_kw: float | None = None,
    delivered_mmbtu_override: float | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Transmission parity is taken from the legacy path.
    Distribution keeps the separate leak factor branch.
    """
    p = emissions_params
    case_study = _resolve_case_study(network, case_study)

    delivered_mmbtu_yr = (
        float(delivered_mmbtu_override)
        if delivered_mmbtu_override is not None
        else float(getattr(network, "capacity_MMBTU_day", 0.0) or 0.0) * 365.0
    )

    mdot_supply_kg_s = (
        float(mdot_override)
        if mdot_override is not None
        else _safe_sum_supply_mdot_kg_s(network)
    )

    x_h2 = _resolve_blend(network, blend_override)
    x_ng = 1.0 - x_h2

    # Keep the original network-based path here.
    # This is the part that matched legacy for upstream/combustion.
    comp = network.composition
    mw_blend = _get_mw_kg_per_kmol(network)
    hhv_blend_mj_per_kg = _get_hhv_mj_per_kg(network)

    hhv_pure, mw_pure, _ = _get_pure_ng_props(comp, case_study=case_study)
    if hhv_pure is None:
        hhv_pure = hhv_blend_mj_per_kg
        mw_pure = mw_blend

    ef_upstream_g_per_mj = (
        x_ng * float(p.EF_NG_UPSTREAM_gCO2_per_MJ) * hhv_pure * mw_pure
        + x_h2 * float(p.EF_H2_UPSTREAM_gCO2_per_MJ) * _HHV_H2 * _MW_H2
    ) / (hhv_blend_mj_per_kg * mw_blend)

    hhv_ratio = hhv_pure * mw_pure / (hhv_blend_mj_per_kg * mw_blend)

    ef_comb_g_per_mj = (
        x_ng * float(p.EF_NG_COMB_gCO2_per_MJ) * hhv_ratio
        if p.include_combustion
        else 0.0
    )

    supply_mj_yr = mdot_supply_kg_s * hhv_blend_mj_per_kg * _SEC_PER_YEAR

    upstream_mmt_yr = supply_mj_yr * ef_upstream_g_per_mj * _G2MMT
    combustion_mmt_yr = supply_mj_yr * ef_comb_g_per_mj * _G2MMT

    # Excel override is in kW. Native network values are in W.
    if elec_comp_override_kw is not None:
        elec_usage_kw = float(elec_comp_override_kw)
        elec_mj_yr = elec_usage_kw * _SEC_PER_YEAR / 1000.0
    else:
        elec_w = float(
            sum(
                getattr(c, "fuel_electric_W", 0.0) or 0.0
                for c in network.compressors.values()
            )
        )
        elec_usage_kw = elec_w / 1000.0
        elec_mj_yr = elec_w * _SEC_PER_YEAR / 1e6

    electricity_mmt_yr = elec_mj_yr * float(p.EF_ELEC_gCO2_per_MJ) * _G2MMT

    fuel_w = _safe_sum_compressor_fuel_w(network)
    fuel_mj_yr = fuel_w * _SEC_PER_YEAR / 1e6
    compressor_combustion_mmt_yr = fuel_mj_yr * ef_comb_g_per_mj * _G2MMT
    downstream_combustion_mmt_yr = combustion_mmt_yr - compressor_combustion_mmt_yr

    fugitive_mmt_yr, fug_debug = _calc_fugitive_details(
        network=network,
        p=p,
        mdot_supply_kg_s=mdot_supply_kg_s,
        km_pipe_override=km_pipe_override,
        case_study=case_study,
        blend_override=blend_override,
        debug=debug,
    )

    total_mmt_yr = (
        upstream_mmt_yr + combustion_mmt_yr + electricity_mmt_yr + fugitive_mmt_yr
    )

    def to_intensity(mmt_yr: float) -> float | None:
        if delivered_mmbtu_yr <= 0:
            return None
        return (mmt_yr * _KG_PER_MMT) / delivered_mmbtu_yr

    _dbg(
        debug,
        (
            "EMISSIONS_UPCOMB_DEBUG | "
            f"x_h2={x_h2:.12f} | "
            f"hhv_pure={hhv_pure:.12f} | "
            f"mw_pure={mw_pure:.12f} | "
            f"hhv_blend={hhv_blend_mj_per_kg:.12f} | "
            f"mw_blend={mw_blend:.12f} | "
            f"hhv_ratio={hhv_ratio:.12f} | "
            f"ef_upstream_g_per_mj={ef_upstream_g_per_mj:.12f} | "
            f"ef_comb_g_per_mj={ef_comb_g_per_mj:.12f}"
        ),
    )

    _dbg(
        debug,
        (
            "EMISSIONS_TOTAL_DEBUG | "
            f"mdot_supply_kg_s={mdot_supply_kg_s:.12f} | "
            f"elec_usage_kw={elec_usage_kw:.12f} | "
            f"upstream_mmt_yr={upstream_mmt_yr:.12f} | "
            f"combustion_mmt_yr={combustion_mmt_yr:.12f} | "
            f"electricity_mmt_yr={electricity_mmt_yr:.12f} | "
            f"fugitive_mmt_yr={fugitive_mmt_yr:.12f} | "
            f"total_mmt_yr={total_mmt_yr:.12f}"
        ),
    )

    return {
        "intensity_kgCO2e_per_MMBTU": to_intensity(total_mmt_yr),
        "annual_MMTCO2e_per_yr": total_mmt_yr,
        "breakdown": {
            "upstream": {
                "intensity_kgCO2e_per_MMBTU": to_intensity(upstream_mmt_yr),
                "annual_MMTCO2e_per_yr": upstream_mmt_yr,
            },
            "combustion": {
                "intensity_kgCO2e_per_MMBTU": to_intensity(combustion_mmt_yr),
                "annual_MMTCO2e_per_yr": combustion_mmt_yr,
            },
            "combustion_compressor": {
                "intensity_kgCO2e_per_MMBTU": to_intensity(
                    compressor_combustion_mmt_yr
                ),
                "annual_MMTCO2e_per_yr": compressor_combustion_mmt_yr,
            },
            "combustion_downstream": {
                "intensity_kgCO2e_per_MMBTU": to_intensity(
                    downstream_combustion_mmt_yr
                ),
                "annual_MMTCO2e_per_yr": downstream_combustion_mmt_yr,
            },
            "electricity": {
                "intensity_kgCO2e_per_MMBTU": to_intensity(electricity_mmt_yr),
                "annual_MMTCO2e_per_yr": electricity_mmt_yr,
            },
            "fugitive": {
                "intensity_kgCO2e_per_MMBTU": to_intensity(fugitive_mmt_yr),
                "annual_MMTCO2e_per_yr": fugitive_mmt_yr,
            },
        },
        "delivered_MMBTU_per_yr": delivered_mmbtu_yr,
        "supply_MJ_per_yr": supply_mj_yr,
        "emissions_debug": {
            "upcomb_x_h2": x_h2,
            "upcomb_hhv_pure_MJ_per_kg": hhv_pure,
            "upcomb_mw_pure_kg_per_kmol": mw_pure,
            "upcomb_hhv_blend_MJ_per_kg": hhv_blend_mj_per_kg,
            "upcomb_mw_blend_kg_per_kmol": mw_blend,
            "upcomb_hhv_ratio": hhv_ratio,
            "upcomb_ef_upstream_g_per_mj": ef_upstream_g_per_mj,
            "upcomb_ef_comb_g_per_mj": ef_comb_g_per_mj,
            "mdot_supply_kg_s": mdot_supply_kg_s,
            "elec_usage_kw": elec_usage_kw,
            "elec_mj_yr": elec_mj_yr,
            "fuel_w_network": fuel_w,
            "fuel_mj_yr_network": fuel_mj_yr,
            "supply_mj_yr": supply_mj_yr,
            **fug_debug,
        },
    }
