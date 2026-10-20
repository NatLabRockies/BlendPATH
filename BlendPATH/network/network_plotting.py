import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot(
    network,
    fig=None,
    ax=None,
    pipe_property="v_avg",
    property_range=None,
    show_demands=True,
    savename=None,
    show_plot=True,
    cbar_range=None,
    pipe_limit_red=np.inf,
    title="",
    pipe_selections=None,
    cmap_name: str = "viridis",
    show_compressors: bool = True,
    show_regulators: bool = True,
) -> None:
    # If fig/ax is supplied, use it. Else make a fresh fig
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # Range of property values to plot
    if property_range is None:
        property_range = (0, np.inf)

    if pipe_selections is not None:
        chosen_pipes = [
            pipe for pipe in network.pipes.values() if pipe.name in pipe_selections
        ]
    else:
        chosen_pipes = network.pipes.values()

    # collect pipe and node coords and values
    cmap = mpl.colormaps[cmap_name]
    edge_collection, color_normal, edge_pos = get_pipe_values(
        pipes=chosen_pipes,
        property_range=property_range,
        pipe_property=pipe_property,
        cmap=cmap,
        cbar_range=cbar_range,
        pipe_limit_red=pipe_limit_red,
    )
    node_pos, node_sizes, node_colors = get_node_values(
        show_demands=show_demands,
        demand_nodes=network.demand_nodes.values(),
        supply_nodes=network.supply_nodes.values(),
    )

    # Get compressor and regulator positions
    compressor_pos = get_comp_regs_pos(network.compressors.values())
    regulator_pos = get_comp_regs_pos(network.regulators.values())

    # Make plots
    ax.add_collection(edge_collection)
    make_scatter(ax, node_pos, s=node_sizes, c=node_colors, marker="o", zorder=2)
    if show_compressors and compressor_pos.size > 0:
        make_scatter(ax, compressor_pos, s=50, c="blue", marker=">", zorder=3)
    if show_regulators and regulator_pos.size > 0:
        make_scatter(ax, regulator_pos, s=50, c="green", marker="<", zorder=4)

    # Clean up figs
    ax.update_datalim(get_view(edge_pos=edge_pos))
    ax.autoscale_view()
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_axis_off()
    ax.set_title(title, loc="left")

    # Set up color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_normal)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar_filter = {
        "p_avg": "Pressure (Pa)",
        "v_avg": "Velocity (m/s)",
        "v_max": "Max Velocity (m/s)",
        "m_dot": "Mass flow rate (kg/s)",
        "diameter_mm": "Inner diameter (mm)",
        "design_pressure_MPa": "Design Pressure (MPa)",
    }
    cbar.set_label(cbar_filter[pipe_property], fontsize=10)
    ax.set_aspect("equal")  # Reset axes equal after adding colorbar

    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, pad_inches=0.1)
        plt.close(fig=None)
    if show_plot:
        plt.show()


# Renormalizing values
def renorm(vals, high, low):
    if len(vals) == 1:
        return high
    return ((vals - vals.min()) / (vals.max() - vals.min())) * (high - low) + low


def get_pipe_values(
    pipes,
    pipe_property,
    property_range,
    cmap,
    cbar_range,
    pipe_limit_red,
):
    # Get pipe values
    edge_pos = []
    pipe_values = []
    widths = []
    alphas = []
    for pipe in pipes:
        x1, y1 = pipe.from_node.plot_xy
        x2, y2 = pipe.to_node.plot_xy
        edge_pos.append([(x1, y1), (x2, y2)])
        pipe_values.append(getattr(pipe, pipe_property))
        if pipe_property == "m_dot":
            pipe_values[-1] = abs(pipe_values[-1])
        widths.append(pipe.diameter_mm)
        alphas.append(
            1
            if property_range[0] < getattr(pipe, pipe_property) < property_range[1]
            else 0.1
        )
    edge_pos = np.asarray(edge_pos)
    width_norm = np.asarray(widths)
    # Set min diameter to line width of 0.5, and max of 2.5
    width_norm = renorm(width_norm, 2.5, 0.5)

    if cbar_range is None:
        pipe_prop_low, pipe_prop_high = min(pipe_values), max(pipe_values)
        color_normal = mpl.colors.Normalize(vmin=pipe_prop_low, vmax=pipe_prop_high)
    else:
        color_normal = cbar_range
    edge_color = [
        cmap(color_normal(e)) if e < pipe_limit_red else "red" for e in pipe_values
    ]

    edge_collection = mpl.collections.LineCollection(
        edge_pos,
        colors=edge_color,
        linewidths=width_norm,
        linestyle="solid",
        alpha=alphas,
        zorder=1,  # edges go behind nodes
    )
    return edge_collection, color_normal, edge_pos


def get_node_values(show_demands, demand_nodes, supply_nodes):
    # Get node values
    node_pos = []
    node_sizes = []
    node_colors = []
    if show_demands:
        for dn in demand_nodes:
            node_pos.append(dn.node.plot_xy)
            node_sizes.append(dn.flowrate_mdot)
            node_colors.append("black")
    for sn in supply_nodes:
        node_pos.append(sn.node.plot_xy)
        node_sizes.append(sn.mdot)
        node_colors.append("orange")
    node_pos = np.asarray(node_pos)
    node_sizes = np.asarray(node_sizes)
    node_sizes = renorm(node_sizes, 50, 1)
    return node_pos, node_sizes, node_colors


def get_view(edge_pos):
    # compute view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    padx, pady = 0.05 * (maxx - minx), 0.05 * (maxy - miny)
    return (minx - padx, miny - pady), (maxx + padx, maxy + pady)


def get_comp_regs_pos(values):
    return np.asarray([value.from_node.plot_xy for value in values])


def make_scatter(ax, pos, s, c, marker, zorder):
    return ax.scatter(pos[:, 0], pos[:, 1], s=s, c=c, marker=marker, zorder=zorder)
