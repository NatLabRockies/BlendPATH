import os.path
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import importlib_resources
from ProFAST import ProFAST

if TYPE_CHECKING:
    from BlendPATH.costing.costing import Costing_params
    from BlendPATH.modifications.mod_util import (
        Mod_capital,
        Mod_costing_params,
        Mod_feedstocks,
        Mod_fixed_costs,
    )


def profast_init(costing_params: "Costing_params") -> ProFAST:
    """Setup a ProFAST

    Args:
        costing_params (Costing_params): Costing parameters

    Returns:
        ProFAST: Return ProFAST instance
    """
    jsonfilename = f"{costing_params.casestudy_name}/financial_params.json"
    if not os.path.exists(jsonfilename):
        jsonfilename = importlib_resources.files("BlendPATH.costing").joinpath(
            "default_financial_params.json"
        )

    return ProFAST(jsonfilename)


def profast_overrides(pf: ProFAST, costing_params: "Costing_params") -> None:
    """Overwrite any parameters in ProFAST with set_params

    Args:
        pf (ProFAST): ProFAST instance
        costing_params (Costing_params): Costing parameters
    """

    if costing_params.financial_overrides is not None:
        for i, v in costing_params.financial_overrides.items():
            pf.set_params(i, v)


def set_capacity(pf: ProFAST, mod_costing_params: "Mod_costing_params") -> None:
    """Update capacity and commodity

    Args:
        pf (ProFAST): ProFAST instance
        mod_costing_params (Mod_costing_params): Modification costing parameters
    """
    pf.set_params("capacity", mod_costing_params.capacity)  # MMBTU/day
    pf.set_params(
        "commodity",
        {
            "name": "Natural Gas-H2 Blend",
            "unit": "MMBTU",
            "initial price": 0.25,
            "escalation": pf.vals["general inflation rate"],
        },
    )


class ProFAST_name_capital(StrEnum):
    NEW_PIPE = "New pipe CapEx"
    NEW_COMPR = "New compression capacity CapEx"
    REFURB_COMPR = "Refurbished compressor capacity CapEx"
    SUPPLY_COMPR = "Supply compressor CapEx"
    METER_TRANS = "Meter & regulator station modification CapEx"
    VALVE = "Valve replacement CapEx"
    ORIG_NETWORK = "Original network residual value"
    MSA_DIST = "Meter set assembly"
    REG_REFURB = "Gas pressure regulation and metering station refurbishment"
    SUPPLY_REG = "Regulator at supply node"


class ProFAST_name_feedstocks(StrEnum):
    GAS_COMPR_FUEL = "Compressor fuel"
    ELEC_COMPR_FUEL = "Compressor fuel (electric)"
    GAS_SUPPLY_COMPR_FUEL = "Supply commpressor fuel"
    ELEC_SUPPLY_COMPR_FUEL = "Supply commpressor fuel (electric)"


class ProFAST_name_fixedom(StrEnum):
    ILI = "In-line inspection"
    ODOR = "Odorization"
    REG_MAINT = "Gas pressure regulation and metering station maintenance"
    LEAK_SURVEY = "Leak survey"


ProFAST_name = StrEnum(
    "ProFAST_name",
    {
        m.name: m.value
        for m in list(ProFAST_name_capital)
        + list(ProFAST_name_feedstocks)
        + list(ProFAST_name_fixedom)
    },
)


def pf_add_feedstocks(
    pf: ProFAST,
    mod_feedstocks: "Mod_feedstocks",
    inflation: float,
    costing_params: "Costing_params",
) -> None:

    # Tuples of name, usage, cost, usage
    # Consider cleaning up into structure in future
    values = [
        (
            ProFAST_name_feedstocks.GAS_COMPR_FUEL,
            mod_feedstocks.all_fuel_usage,
            costing_params.cf_price,
            "MMBTU",
        ),
        (
            ProFAST_name_feedstocks.ELEC_COMPR_FUEL,
            mod_feedstocks.all_fuel_elec,
            costing_params.elec_price,
            "kWh",
        ),
        (
            ProFAST_name_feedstocks.ELEC_SUPPLY_COMPR_FUEL,
            mod_feedstocks.supply_comp_fuel["elec"],
            costing_params.elec_price,
            "kWh",
        ),
        (
            ProFAST_name_feedstocks.GAS_SUPPLY_COMPR_FUEL,
            mod_feedstocks.supply_comp_fuel["gas"],
            costing_params.cf_price,
            "MMBTU",
        ),
    ]

    for val in values:
        pf.add_feedstock(
            name=val[0],
            usage=val[1],
            unit=val[3],
            cost=val[2],
            escalation=inflation,
        )


def pf_add_fixed_costs(
    pf: ProFAST,
    mod_fixed_costs: "Mod_fixed_costs",
    inflation: float,
) -> None:

    # Tuples of name, cost
    values = [
        (
            ProFAST_name_fixedom.ILI,
            mod_fixed_costs.ili_costs,
        ),
        (
            ProFAST_name_fixedom.ODOR,
            mod_fixed_costs.odorization,
        ),
        (
            ProFAST_name_fixedom.REG_MAINT,
            mod_fixed_costs.regulator_maintenance,
        ),
        (
            ProFAST_name_fixedom.LEAK_SURVEY,
            mod_fixed_costs.leak_survey,
        ),
    ]

    for val in values:
        pf.add_fixed_cost(
            name=val[0],
            usage=1,
            unit="$",
            cost=val[1],
            escalation=inflation,
        )


def pf_add_capital(
    pf: ProFAST,
    mod_capital: "Mod_capital",
    depr_period: int,
    costing_params: "Costing_params",
    new_pipe_capex: float,
) -> None:

    for name, val in mod_capital.network_costs().items():
        pf.add_capital_item(
            name=name,
            cost=new_pipe_capex if name == ProFAST_name_capital.NEW_PIPE else val,
            depr_type="Straight line",
            depr_period=depr_period,
            refurb=[0],
        )


def pf_aggregation(pf: ProFAST, lcot: float) -> dict[str, float]:

    price_breakdown = pf.get_cost_breakdown()
    price_breakdown_dict = dict(
        zip(price_breakdown["Name"].values, price_breakdown["NPV"].values)
    )

    # Feedstocks
    fs_vals = [member.value for member in ProFAST_name_feedstocks]
    # Fixed costs
    fc_vals = [member.value for member in ProFAST_name_fixedom]

    prices_all = {"LCOT: Levelized cost of transport": lcot}

    # Feedstocks and fixed costs
    for fs_val in fs_vals + fc_vals:
        prices_all[f"LCOT: {fs_val}"] = price_breakdown_cols(
            price_breakdown_dict, pos=[fs_val], neg=[]
        )

    # Generic categories
    prices_all["LCOT: Taxes"] = price_breakdown_cols(
        price_breakdown_dict,
        pos=["Income taxes payable", "Capital gains taxes payable"],
        neg=["Monetized tax losses"],
    )
    prices_all["LCOT: Financial"] = price_breakdown_cols(
        price_breakdown_dict,
        pos=["Non-depreciable assets", "Cash on hand reserve"],
        neg=["Sale of non-depreciable assets", "Cash on hand recovery"],
    )
    prices_all["LCOT: Fixed O&M"] = price_breakdown_cols(
        price_breakdown_dict,
        pos=["Administrative expenses", "Property insurance"],
        neg=[],
    )

    # CapEx allocation based on % of total capex
    cap_related_values = price_breakdown_cols(
        price_breakdown_dict,
        pos=["Repayment of debt", "Interest expense", "Dividends paid"],
        neg=["Inflow of debt", "Inflow of equity"],
    )
    total_capex = sum([ci.cost for ci in pf.capital_items.values()]) or 1
    for ci_name, ci in pf.capital_items.items():
        prices_all[f"LCOT: {ci_name}"] = (
            price_breakdown_cols(price_breakdown_dict, pos=[ci_name], neg=[])
            + ci.cost / total_capex * cap_related_values
        )
    return prices_all


def pf_get_inputs(pf: ProFAST) -> dict[str, Any]:
    pf_in = {}
    pf_in["vals"] = {i: v for i, v in pf.vals.items()}
    pf_in["capital"] = {
        i: {
            "cost": v.cost,
            "depr_type": v.depr_type,
            "depr_period": v.depr_period,
            "refurb": v.refurb,
        }
        for i, v in pf.capital_items.items()
    }
    pf_in["feedstocks"] = {
        i: {"cost": v.cost, "usage": v.usage, "escalation": v.escalation}
        for i, v in pf.feedstocks.items()
    }
    pf_in["fixed_costs"] = {
        i: {"cost": v.cost, "escalation": v.escalation}
        for i, v in pf.fixed_costs.items()
    }
    return pf_in


def price_breakdown_cols(dict_vals: dict[str, float], pos: list, neg: list) -> float:
    """
    Sum up positive and negative colums for cost breakdown
    """
    return sum([dict_vals.get(x, 0) for x in pos]) - sum(
        [dict_vals.get(x, 0) for x in neg]
    )
