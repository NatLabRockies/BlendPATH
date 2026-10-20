from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString

import BlendPATH.Global as gl


def _safe_float(val):
    try:
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        return float(val)
    except Exception:
        return None


def _node_xy(node):
    xy = getattr(node, "plot_xy", None)
    if xy is None or not isinstance(xy, (tuple, list)) or len(xy) != 2:
        return None

    x = _safe_float(xy[0])
    y = _safe_float(xy[1])

    if x is None or y is None:
        return None
    return (x, y)


def _point_geom(node):
    xy = _node_xy(node)
    return Point(xy) if xy is not None else None


def _line_geom(from_node, to_node):
    a = _node_xy(from_node)
    b = _node_xy(to_node)

    if a is None or b is None or a == b:
        return None

    return LineString([a, b])


def _node_type(network, node):
    is_supply = any(sn.node is node for sn in network.supply_nodes.values())
    is_demand = any(dn.node is node for dn in network.demand_nodes.values())

    if is_supply and is_demand:
        return "supply_demand"
    if is_supply:
        return "supply"
    if is_demand:
        return "demand"
    return "junction"


def build_nodes_gdf(network, crs=None):
    records = []

    for node in network.nodes.values():
        if not getattr(node, "_report_out", True):
            continue

        p_pa = getattr(node, "pressure", None)

        xy = _node_xy(node)
        if xy is None:
            plot_x, plot_y = None, None
        else:
            plot_x, plot_y = xy

        records.append(
            {
                "node_name": node.name,
                "node_type": _node_type(network, node),
                "pressure_mpa_g": (
                    _safe_float(p_pa / gl.MPA2PA) if p_pa is not None else None
                ),
                "p_max_mpa_g": _safe_float(getattr(node, "p_max_mpa_g", None)),
                "x_h2": _safe_float(getattr(node, "x_h2", None)),
                "plot_x": plot_x,
                "plot_y": plot_y,
                "geometry": _point_geom(node),
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def build_pipes_gdf(network, crs=None):
    records = []

    for pipe in network.parent_pipes.values():
        flow = _safe_float(getattr(pipe, "m_dot", None))
        p_from = getattr(pipe.from_node, "pressure", None)
        p_to = getattr(pipe.to_node, "pressure", None)

        if flow is None:
            flow_dir = None
        elif flow >= 0:
            flow_dir = f"{pipe.from_node.name}->{pipe.to_node.name}"
        else:
            flow_dir = f"{pipe.to_node.name}->{pipe.from_node.name}"

        velocity = getattr(pipe, "v_avg", None)

        records.append(
            {
                "pipe_name": pipe.name,
                "parent_pipe": getattr(pipe, "_parent_pipe", pipe.name),
                "pipe_type": type(pipe).__name__,
                "from_node": pipe.from_node.name,
                "to_node": pipe.to_node.name,
                "diameter_mm": _safe_float(getattr(pipe, "diameter_mm", None)),
                "length_km": _safe_float(getattr(pipe, "length_km", None)),
                "roughness_mm": _safe_float(getattr(pipe, "roughness_mm", None)),
                "thickness_mm": _safe_float(getattr(pipe, "thickness_mm", None)),
                "rating_code": getattr(pipe, "rating_code", None),
                "p_max_mpa_g": _safe_float(getattr(pipe, "p_max_mpa_g", None)),
                "design_p_mpa": _safe_float(
                    getattr(pipe, "design_pressure_MPa", None)
                ),
                "ff_type": getattr(pipe, "ff_type", None),
                "flow_kg_s": flow,
                "flow_abs_kg_s": abs(flow) if flow is not None else None,
                "flow_dir": flow_dir,
                "velocity_m_s": _safe_float(velocity),
                "mach_number": _safe_float(getattr(pipe, "mach_number", None)),
                "x_h2": _safe_float(getattr(pipe, "x_h2", None)),
                "p_from_mpa_g": (
                    _safe_float(p_from / gl.MPA2PA) if p_from is not None else None
                ),
                "p_to_mpa_g": (
                    _safe_float(p_to / gl.MPA2PA) if p_to is not None else None
                ),
                "dp_mpa_g": (
                    _safe_float((p_from - p_to) / gl.MPA2PA)
                    if p_from is not None and p_to is not None
                    else None
                ),
                "geometry": _line_geom(pipe.from_node, pipe.to_node),
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

def build_nodes_shp_gdf(network, crs=None):
    gdf = build_nodes_gdf(network=network, crs=crs).copy()

    return gdf.rename(
        columns={
            "node_name": "NODE_NAME",
            "node_type": "NODE_TYPE",
            "pressure_mpa_g": "P_MPA_G",
            "p_max_mpa_g": "PMAX_MPA",
            "x_h2": "X_H2",
        }
    )


def build_pipes_shp_gdf(network, crs=None):
    gdf = build_pipes_gdf(network=network, crs=crs).copy()

    return gdf.rename(
        columns={
            "pipe_name": "PIPE_NAME",
            "parent_pipe": "PARENT_ID",
            "pipe_type": "PIPE_TYPE",
            "from_node": "FROM_NODE",
            "to_node": "TO_NODE",
            "diameter_mm": "DIA_MM",
            "length_km": "LEN_KM",
            "roughness_mm": "ROUGH_MM",
            "thickness_mm": "THK_MM",
            "rating_code": "RT_CODE",
            "p_max_mpa_g": "PMAX_MPA",
            "design_p_mpa": "DES_P_MPA",
            "ff_type": "FF_TYPE",
            "flow_kg_s": "Q_KG_S",
            "flow_abs_kg_s": "QABS_KGS",
            "flow_dir": "FLOW_DIR",
            "velocity_m_s": "V_M_S",
            "mach_number": "MACH_NO",
            "x_h2": "X_H2",
            "p_from_mpa_g": "PFR_MPA",
            "p_to_mpa_g": "PTO_MPA",
            "dp_mpa_g": "DP_MPA",
        }
    )


def _delete_shapefile_set(shp_path: Path) -> None:
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj"]:
        p = shp_path.with_suffix(ext)
        if p.exists():
            p.unlink()


def write_network_shp(
    network,
    foldername: str,
    *,
    crs: str | None = None,
):
    out_dir = Path(foldername)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = out_dir / "nodes.shp"
    pipes_path = out_dir / "pipes.shp"

    _delete_shapefile_set(nodes_path)
    _delete_shapefile_set(pipes_path)

    nodes_gdf = build_nodes_shp_gdf(network=network, crs=crs)
    pipes_gdf = build_pipes_shp_gdf(network=network, crs=crs)

    nodes_gdf.to_file(
        nodes_path,
        driver="ESRI Shapefile",
        index=False,
    )

    pipes_gdf.to_file(
        pipes_path,
        driver="ESRI Shapefile",
        index=False,
    )


def write_network_gis(
    network,
    filename: str,
    *,
    crs: str | None = None,
    driver: Literal["GPKG", "GeoJSON"] = "GPKG",
):
    out = Path(filename)

    nodes_gdf = build_nodes_gdf(network=network, crs=crs)
    pipes_gdf = build_pipes_gdf(network=network, crs=crs)

    if driver == "GPKG":
        if out.suffix.lower() != ".gpkg":
            out = out.with_suffix(".gpkg")

        if out.exists():
            out.unlink()

        nodes_gdf.to_file(
            out,
            layer="nodes",
            driver="GPKG",
            index=False,
            metadata={
                "network_name": str(network.name),
                "scenario_type": str(network.scenario_type),
                "eos": str(network.eos),
            },
        )
        pipes_gdf.to_file(
            out,
            layer="pipes",
            driver="GPKG",
            index=False,
        )

    elif driver == "GeoJSON":
        base = out.with_suffix("")
        nodes_gdf.to_file(
            f"{base}_nodes.geojson",
            driver="GeoJSON",
            index=False,
        )
        pipes_gdf.to_file(
            f"{base}_pipes.geojson",
            driver="GeoJSON",
            index=False,
        )
    else:
        raise ValueError(f"Unsupported GIS driver: {driver}")