from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


# Generic time-series support for BlendPATH networks.
#
# Expected demand profile format:
#
#   Excel file: demand_profile.xlsx
#   Sheet:      DEMAND
#   Columns:    time_step, demand_name, flowrate_MW
#
# Optional columns are allowed and ignored by the solver, for example:
#   node_name, min_pressure_mpa_g, scenario, note
#
# The demand_name values must match the demand_name values already present in
# the imported BlendPATH network object.
#
# Output design:
#   One Excel workbook is written for the full time series.
#   The workbook keeps the same sheet names produced by BlendPATH.to_file():
#       INFO, NODES, PIPES, COMPRESSORS, COMPOSITION, REGULATORS
#   Each sheet is stacked over time and gets two extra columns at the front:
#       step_index, time_step
#
# This keeps the original BlendPATH output structure while avoiding one output
# file per timestep.


EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
STATIC_SHEETS = {"INFO", "COMPOSITION"}


def _clean_column_name(value: Any) -> str:
    """Normalize only Excel column headers, not demand names."""
    return str(value).strip()


def _require_columns(
    df: pd.DataFrame,
    required_columns: set[str],
    sheet_name: str,
) -> None:
    """Raise a clear error if the profile sheet is missing required columns."""
    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            f"Time-series sheet '{sheet_name}' is missing required columns: {missing}. "
            f"Required columns are: {sorted(required_columns)}. "
            f"Found columns are: {list(df.columns)}."
        )


def _format_time_step(value: Any) -> str:
    """Format a time_step value for output tables and filenames."""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()

    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    return str(value)


def _safe_file_tag(value: Any) -> str:
    """Create a safe generic filename tag from a time_step value."""
    text = _format_time_step(value).strip()

    if not text:
        return "unknown_time"

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:80]


def _resolve_output_workbook(output_file: str | Path) -> Path:
    """
    Resolve the single time-series output workbook path.

    Recommended:
        output_file = CASE_FOLDER / "timeseries_output.xlsx"

    If a path without an Excel suffix is passed, '.xlsx' is appended.
    """
    output_path = Path(output_file)

    if output_path.suffix.lower() not in EXCEL_SUFFIXES:
        output_path = output_path.with_suffix(".xlsx")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _read_timeseries_demand_sheet(
    demand_profile_file: str | Path,
    *,
    profile_sheet_name: str = "DEMAND",
    max_steps: int | None = None,
) -> pd.DataFrame:
    """
    Read the generic BlendPATH demand time-series input.

    Required columns:
        time_step
        demand_name
        flowrate_MW

    One row means:
        at this time_step, set this demand_name to this flowrate_MW.
    """
    path = Path(demand_profile_file)

    if not path.exists():
        raise FileNotFoundError(f"Demand profile file was not found: {path}")

    raw = pd.read_excel(
        path,
        sheet_name=profile_sheet_name,
        engine="openpyxl",
    )

    raw.columns = [_clean_column_name(col) for col in raw.columns]

    required = {"time_step", "demand_name", "flowrate_MW"}
    _require_columns(raw, required, profile_sheet_name)

    df = raw.copy()

    # Remove fully blank rows.
    df = df.dropna(how="all").copy()

    # Remove rows that do not define a demand.
    df = df[df["demand_name"].notna()].copy()

    if df.empty:
        raise ValueError(
            f"Time-series sheet '{profile_sheet_name}' has no demand rows."
        )

    # Exact BlendPATH demand_name match. No fuzzy matching.
    df["demand_name"] = df["demand_name"].astype(str).str.strip()

    if df["time_step"].isna().any():
        bad_rows = df.index[df["time_step"].isna()].tolist()[:20]
        raise ValueError(
            f"Column 'time_step' contains blank values. "
            f"First bad row indexes: {bad_rows}"
        )

    df["flowrate_MW"] = pd.to_numeric(df["flowrate_MW"], errors="coerce")

    if df["flowrate_MW"].isna().any():
        bad = df.loc[df["flowrate_MW"].isna(), ["time_step", "demand_name"]].head(20)
        raise ValueError(
            "Column 'flowrate_MW' contains blank or non-numeric values. "
            f"First bad rows: {bad.to_dict(orient='records')}"
        )

    duplicate_mask = df.duplicated(
        subset=["time_step", "demand_name"],
        keep=False,
    )

    if duplicate_mask.any():
        dupes = (
            df.loc[duplicate_mask, ["time_step", "demand_name"]]
            .head(20)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Each demand_name can appear only once per time_step. "
            f"First duplicate rows: {dupes}"
        )

    # Preserve the order in the Excel file.
    unique_time_steps = list(pd.unique(df["time_step"]))

    if max_steps is not None:
        unique_time_steps = unique_time_steps[: int(max_steps)]
        df = df[df["time_step"].isin(unique_time_steps)].copy()

    step_lookup = {
        time_step: step_index
        for step_index, time_step in enumerate(unique_time_steps)
    }

    df["step_index"] = df["time_step"].map(step_lookup).astype(int)

    return df


def _validate_profile_demands(
    network: Any,
    profile: pd.DataFrame,
    *,
    missing_demand_policy: str,
) -> None:
    """
    Validate demand_name values against the already imported network.

    missing_demand_policy:
        "error"         every network demand must appear at every timestep
        "keep_existing" missing demand keeps its imported/original flowrate
        "zero"          missing demand is set to 0.0 MW
    """
    valid_policies = {"error", "keep_existing", "zero"}

    if missing_demand_policy not in valid_policies:
        raise ValueError(
            f"missing_demand_policy must be one of {sorted(valid_policies)}, "
            f"got {missing_demand_policy!r}."
        )

    network_demands = set(network.demand_nodes.keys())
    profile_demands = set(profile["demand_name"])

    unknown = sorted(profile_demands - network_demands)

    if unknown:
        raise KeyError(
            "The demand profile contains demand_name values that are not in the "
            f"network DEMAND sheet. First unknown names: {unknown[:20]}"
        )

    if missing_demand_policy == "error":
        missing_by_step = []

        for step_index in sorted(profile["step_index"].unique()):
            timestep_rows = profile[profile["step_index"] == step_index]
            time_step = timestep_rows["time_step"].iloc[0]

            step_demands = set(timestep_rows["demand_name"])
            missing = sorted(network_demands - step_demands)

            if missing:
                missing_by_step.append(
                    {
                        "time_step": _format_time_step(time_step),
                        "missing_demands": missing[:20],
                    }
                )

            if len(missing_by_step) >= 5:
                break

        if missing_by_step:
            raise KeyError(
                "The demand profile must include every network demand at every "
                "time_step when missing_demand_policy='error'. "
                f"First missing examples: {missing_by_step}"
            )


def _apply_timestep_demands(
    network: Any,
    timestep_rows: pd.DataFrame,
    *,
    original_flowrates_MW: dict[str, float],
    missing_demand_policy: str,
) -> float:
    """Set network demand flowrates for one timestep and return total demand MW."""
    values = dict(zip(timestep_rows["demand_name"], timestep_rows["flowrate_MW"]))

    total_mw = 0.0

    for demand_name, demand_node in network.demand_nodes.items():
        if demand_name in values:
            flowrate_mw = float(values[demand_name])
        elif missing_demand_policy == "keep_existing":
            flowrate_mw = float(original_flowrates_MW[demand_name])
        elif missing_demand_policy == "zero":
            flowrate_mw = 0.0
        else:
            raise KeyError(
                f"Demand {demand_name!r} is missing from this timestep."
            )

        demand_node.flowrate_MW = flowrate_mw
        demand_node.recalc_mdot()
        total_mw += flowrate_mw

    try:
        network.unique_demand_nodes = list(
            set(dn.node.name for dn in network.demand_nodes.values())
        )
    except Exception:
        pass

    return total_mw

def _read_standard_blendpath_output(
    workbook_file: str | Path,
    *,
    step_index: int,
    time_step: Any,
) -> dict[str, pd.DataFrame]:
    """
    Read one normal BlendPATH output workbook.

    Static metadata sheets are kept as-is:
        INFO
        COMPOSITION

    Time-varying result sheets get time-series columns:
        NODES
        PIPES
        COMPRESSORS
        REGULATORS
    """
    workbook_file = Path(workbook_file)
    time_step_label = _format_time_step(time_step)

    static_sheets = STATIC_SHEETS
    sheet_frames: dict[str, pd.DataFrame] = {}

    with pd.ExcelFile(workbook_file, engine="openpyxl") as xls:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # INFO and COMPOSITION are static metadata/configuration sheets.
            # Keep them exactly as BlendPATH.to_file() wrote them.
            if sheet_name.upper() in static_sheets:
                sheet_frames[sheet_name] = df
                continue

            # Result sheets get timestep identifiers.
            # Original BlendPATH columns remain unchanged after these two columns.
            df.insert(0, "time_step", time_step_label)
            df.insert(0, "step_index", step_index)

            sheet_frames[sheet_name] = df

    return sheet_frames


def _write_combined_timeseries_workbook(
    output_file: str | Path,
    sheet_frames: dict[str, list[pd.DataFrame]],
) -> Path:
    """
    Write the final single time-series workbook.

    Time-varying result sheets are stacked over time. Static metadata sheets
    (INFO, COMPOSITION) are written once.
    """
    output_path = _resolve_output_workbook(output_file)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frames in sheet_frames.items():
            if not frames:
                combined = pd.DataFrame()
            elif sheet_name.upper() in STATIC_SHEETS:
                combined = frames[0]
            else:
                combined = pd.concat(frames, ignore_index=True)

            combined.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return output_path


def _write_optional_gis_outputs_for_timestep(
    network: Any,
    *,
    output_file: str | Path,
    step_index: int,
    time_step: Any,
    write_gpkg_each_timestep: bool,
    write_shp_each_timestep: bool,
    gis_output_folder: str | Path | None,
    gis_crs: str | None,
) -> None:
    """Write optional GIS outputs per timestep if requested."""
    if not write_gpkg_each_timestep and not write_shp_each_timestep:
        return

    if gis_output_folder is None:
        base_output_path = _resolve_output_workbook(output_file)
        gis_folder = base_output_path.with_suffix("") / "gis"
    else:
        gis_folder = Path(gis_output_folder)

    gis_folder.mkdir(parents=True, exist_ok=True)

    tag = f"{step_index:04d}_{_safe_file_tag(time_step)}"

    if write_gpkg_each_timestep:
        network.to_gis(
            str(gis_folder / f"timestep_{tag}.gpkg"),
            crs=gis_crs,
            driver="GPKG",
        )

    if write_shp_each_timestep:
        network.to_shp(
            str(gis_folder / f"timestep_{tag}_shp"),
            crs=gis_crs,
        )


def solve_timeseries(
    network: Any,
    demand_profile_file: str | Path,
    output_file: str | Path,
    *,
    demand_map: Any = None,
    c_relax: float = 50,
    max_steps: int | None = None,
    profile_sheet_name: str = "DEMAND",
    missing_demand_policy: str = "error",
    write_gpkg_each_timestep: bool = False,
    write_shp_each_timestep: bool = False,
    gis_output_folder: str | Path | None = None,
    gis_crs: str | None = "EPSG:4326",
    initializer: int | dict[str, float] = 0,
    solve_kwargs: dict[str, Any] | None = None,
    **unused_kwargs: Any,
) -> Path:
    """
    Solve an imported BlendPATH network over a generic demand time series.

    Input demand profile:
        Excel workbook with a DEMAND sheet containing:
            time_step, demand_name, flowrate_MW

    Output:
        One Excel workbook for the full time series.
        Sheet names match the normal BlendPATH output workbook.
        Each sheet has two added columns:
            step_index, time_step

    demand_map:
        Accepted only for backward API compatibility. It is not used by the
        generic implementation.
    """
    if demand_map is not None:
        raise ValueError(
            "demand_map is not used in the generic time-series implementation. "
            "Use demand_name values in demand_profile.xlsx that exactly match "
            "network DEMAND.demand_name."
        )

    if unused_kwargs:
        raise TypeError(
            "Unsupported keyword arguments for generic solve_timeseries: "
            f"{sorted(unused_kwargs)}"
        )

    profile = _read_timeseries_demand_sheet(
        demand_profile_file,
        profile_sheet_name=profile_sheet_name,
        max_steps=max_steps,
    )

    _validate_profile_demands(
        network,
        profile,
        missing_demand_policy=missing_demand_policy,
    )

    original_flowrates_MW = {
        demand_name: float(getattr(demand_node, "flowrate_MW", 0.0) or 0.0)
        for demand_name, demand_node in network.demand_nodes.items()
    }

    sheet_frames: dict[str, list[pd.DataFrame]] = {}

    current_initializer = initializer
    extra_solve_kwargs = solve_kwargs or {}

    with tempfile.TemporaryDirectory(prefix="blendpath_timeseries_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for step_index in sorted(profile["step_index"].unique()):
            step_index_int = int(step_index)
            timestep_rows = profile[profile["step_index"] == step_index_int].copy()
            time_step = timestep_rows["time_step"].iloc[0]

            _apply_timestep_demands(
                network,
                timestep_rows,
                original_flowrates_MW=original_flowrates_MW,
                missing_demand_policy=missing_demand_policy,
            )

            network.solve(
                initializer=current_initializer,
                c_relax=c_relax,
                **extra_solve_kwargs,
            )

            current_initializer = getattr(network, "pressure_init_out", 0)

            # Write one normal BlendPATH workbook to a temporary file.
            # Then read it back and stack its original sheets into the final
            # single time-series workbook.
            temp_step_file = temp_dir / f"timestep_{step_index_int:04d}.xlsx"
            network.to_file(str(temp_step_file))

            step_sheets = _read_standard_blendpath_output(
                temp_step_file,
                step_index=step_index_int,
                time_step=time_step,
            )

            for sheet_name, sheet_df in step_sheets.items():
                sheet_frames.setdefault(sheet_name, []).append(sheet_df)

            _write_optional_gis_outputs_for_timestep(
                network,
                output_file=output_file,
                step_index=step_index_int,
                time_step=time_step,
                write_gpkg_each_timestep=write_gpkg_each_timestep,
                write_shp_each_timestep=write_shp_each_timestep,
                gis_output_folder=gis_output_folder,
                gis_crs=gis_crs,
            )

    return _write_combined_timeseries_workbook(
        output_file,
        sheet_frames=sheet_frames,
    )
