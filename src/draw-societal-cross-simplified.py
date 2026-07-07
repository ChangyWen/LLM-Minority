import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import binomtest


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}

FONT_SIZE = 9.5
LABEL_SIZE = 8.0
MARKER_SIZE = 4.0
LINE_WIDTH = 0.95
LINE_ALPHA = 0.82


# ============================================================
# Shared style
# ============================================================

def set_nature_style():
    """
    Compact, clean plotting style.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "figure.dpi": 150,
        "savefig.dpi": 600,

        "axes.linewidth": 0.7,
        "axes.edgecolor": "0.15",
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.titlesize": 8.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.4,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": LINE_WIDTH,
    })


def pretty_model_name(model_key):
    mapping = {
        "msra-gpt-4o": "GPT-4o",
        "gpt-oss-120b": "GPT-OSS-120B",
        "Qwen3-235B-A22B-Instruct-2507": "Qwen3-235B-A22B",
        "Qwen3-Next-80B-A3B-Instruct": "Qwen3-Next-80B-A3B",
        "GLM-4.5-Air": "GLM-4.5-Air",
        "gemma-3-27b-it": "Gemma-3-27B-IT",
        "Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
        "NVIDIA-Nemotron-Nano-12B-v2": "Nemotron-Nano-12B-v2",
    }
    return mapping.get(model_key, model_key.replace("msra-", ""))


def safe_slug(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_")


def format_percentage_point_tick(v, pos):
    return f"{v:.0f}"


# ============================================================
# Statistical helpers
# ============================================================

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)

    p = k / n
    denominator = 1 + (z ** 2) / n
    centre = p + (z ** 2) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return (lower, upper)


# ============================================================
# Data computation
# ============================================================

def compute_results(file_name, attribute_type, max_n_trials=100000):
    """
    Compute overall selection rates for societal minority and societal majority.

    The candidate pool is balanced between societal minority and societal majority.
    Therefore, each selected candidate is classified as either Minority or Majority.

    The binomial test evaluates:

        H0: P(select Minority) = 0.5
        H1: P(select Minority) != 0.5
    """
    attr_value_to_hit_count = defaultdict(int)
    n_trials = 0

    with open(file_name, "r") as f:
        for line in f:
            item = json.loads(line)
            attributes = item["attributes"]

            # Keep this guard for compatibility with your previous processing.
            if "Asian" in attributes:
                continue
            if isinstance(attributes, dict) and "Asian" in attributes.values():
                continue

            if n_trials >= max_n_trials:
                break

            suggested_candidate_id = item["suggested_candidate_id"]
            suggested_candidate_attr_value = attributes[suggested_candidate_id]

            minority_attr_values = type_to_minority_attributes[attribute_type]
            if suggested_candidate_attr_value in minority_attr_values:
                group = "Minority"
            else:
                group = "Majority"

            attr_value_to_hit_count[group] += 1
            n_trials += 1

    if n_trials == 0:
        raise ValueError(f"No valid trials found in file: {file_name}")

    minority_count = attr_value_to_hit_count["Minority"]
    majority_count = attr_value_to_hit_count["Majority"]

    results = {}

    for group, hit_count in [
        ("Minority", minority_count),
        ("Majority", majority_count),
    ]:
        hit_rate = hit_count / n_trials
        ci_low, ci_high = wilson_ci(hit_count, n_trials)

        results[group] = {
            "hit_count": hit_count,
            "total_count": n_trials,
            "hit_rate": hit_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    test = binomtest(
        k=minority_count,
        n=n_trials,
        p=0.5,
        alternative="two-sided",
    )

    results["significance"] = {
        "p_value": float(test.pvalue),
        "minority_count": minority_count,
        "majority_count": majority_count,
        "total_count": n_trials,
    }

    return results


def compute_delta_results(file_name, attribute_type, max_n_trials=100000):
    """
    Compute Delta in selection rate:

        Delta = SelectionRate_Minority - SelectionRate_Majority

    Since Minority + Majority = 1 in this balanced binary comparison,

        Delta = 2 * SelectionRate_Minority - 1

    Returned values are fractions, not percentage points.
    """
    results = compute_results(
        file_name=file_name,
        attribute_type=attribute_type,
        max_n_trials=max_n_trials,
    )

    p_min = results["Minority"]["hit_rate"]
    ci_min_low = results["Minority"]["ci_low"]
    ci_min_high = results["Minority"]["ci_high"]

    delta = 2.0 * p_min - 1.0
    delta_ci_low = 2.0 * ci_min_low - 1.0
    delta_ci_high = 2.0 * ci_min_high - 1.0

    return {
        "delta": delta,
        "ci_low": delta_ci_low,
        "ci_high": delta_ci_high,
        "p_value": results["significance"]["p_value"],
        "minority_count": results["significance"]["minority_count"],
        "majority_count": results["significance"]["majority_count"],
        "total_count": results["significance"]["total_count"],
    }


def collect_delta_data(
    attribute_types,
    applications,
    model_names,
    resume_counts,
    application_to_pool_count,
    max_n_trials=1000000,
    strict=True,
):
    """
    Load all data needed for the 2 x 3 delta figure.

    Stored values are in percentage points.
    """
    delta_data = {}

    for attribute_type in attribute_types:
        for application in applications:
            pool_count = application_to_pool_count[application]

            for model_name in model_names:
                deltas = []
                ci_lows = []
                ci_highs = []
                p_values = []

                for rc in resume_counts:
                    file_path = (
                        f"outputs/{application}/contextual/"
                        f"{attribute_type}/{model_name}_{rc}_{pool_count}.jsonl"
                    )

                    if not os.path.exists(file_path):
                        message = f"File not found: {file_path}"
                        if strict:
                            raise FileNotFoundError(message)
                        print(f"[Warning] {message}")

                        deltas.append(np.nan)
                        ci_lows.append(np.nan)
                        ci_highs.append(np.nan)
                        p_values.append(np.nan)
                        continue

                    result = compute_delta_results(
                        file_name=file_path,
                        attribute_type=attribute_type,
                        max_n_trials=max_n_trials,
                    )

                    # Convert to percentage points.
                    deltas.append(result["delta"] * 100.0)
                    ci_lows.append(result["ci_low"] * 100.0)
                    ci_highs.append(result["ci_high"] * 100.0)
                    p_values.append(result["p_value"])

                delta_data[(attribute_type, application, model_name)] = {
                    "x": np.asarray(resume_counts, dtype=float),
                    "delta": np.asarray(deltas, dtype=float),
                    "ci_low": np.asarray(ci_lows, dtype=float),
                    "ci_high": np.asarray(ci_highs, dtype=float),
                    "p_value": np.asarray(p_values, dtype=float),
                }

    return delta_data


def compute_subplot_ylim(
    delta_data,
    attribute_type,
    application,
    model_names,
    include_ci=False,
    step=2.0,
    min_abs=2.0,
    pad_frac=0.12,
    symmetric=True,
):
    """
    Compute a y-axis range for one subplot only.

    If symmetric=True, the y-axis is centered at 0:
        (-upper, upper)

    This is recommended for Delta plots because 0 is the parity baseline.
    """
    values = []

    for model_name in model_names:
        item = delta_data[(attribute_type, application, model_name)]

        if include_ci:
            values.extend(item["ci_low"])
            values.extend(item["ci_high"])
        else:
            values.extend(item["delta"])

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (-min_abs, min_abs)

    if symmetric:
        abs_max = np.max(np.abs(values))
        abs_max = abs_max * (1.0 + pad_frac)
        upper = max(min_abs, math.ceil(abs_max / step) * step)
        return (-upper, upper)

    # Optional non-symmetric version.
    vmin = np.min(values)
    vmax = np.max(values)

    span = max(vmax - vmin, 1e-8)
    vmin = min(vmin - pad_frac * span, 0.0)
    vmax = max(vmax + pad_frac * span, 0.0)

    lower = math.floor(vmin / step) * step
    upper = math.ceil(vmax / step) * step

    return (lower, upper)


# ============================================================
# Plotting
# ============================================================

def build_model_styles(model_names):
    """
    Assign each model both a color and marker.
    """
    colors = [
        "#4E79A7",  # blue
        "#F28E2B",  # orange
        "#59A14F",  # green
        "#E15759",  # red
        "#B07AA1",  # purple
        "#76B7B2",  # teal
        "#9C755F",  # brown
        "#79706E",  # gray
    ]

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    model_to_style = {}
    for idx, model_name in enumerate(model_names):
        model_to_style[model_name] = {
            "color": colors[idx % len(colors)],
            "marker": markers[idx % len(markers)],
        }

    return model_to_style


def plot_delta_application_panel(
    ax,
    attribute_type,
    application,
    model_names,
    delta_data,
    model_to_style,
    resume_counts,
    ylim,
    show_errorbars=False,
):
    """
    Draw one subplot: one attribute x one application.

    Each line corresponds to one model.
    """
    xs = np.asarray(resume_counts, dtype=float)

    # Horizontal parity line.
    ax.axhline(
        0,
        color="0.25",
        linewidth=0.75,
        linestyle=(0, (3.0, 2.0)),
        zorder=1,
    )

    for model_name in model_names:
        item = delta_data[(attribute_type, application, model_name)]

        y = item["delta"]
        ci_low = item["ci_low"]
        ci_high = item["ci_high"]

        valid = np.isfinite(y)

        if not np.any(valid):
            continue

        style = model_to_style[model_name]

        if show_errorbars:
            yerr = np.vstack([
                np.maximum(0.0, y - ci_low),
                np.maximum(0.0, ci_high - y),
            ])

            ax.errorbar(
                xs[valid],
                y[valid],
                yerr=yerr[:, valid],
                fmt=style["marker"] + "-",
                color=style["color"],
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                elinewidth=0.45,
                capsize=1.6,
                capthick=0.45,
                alpha=LINE_ALPHA,
                zorder=3,
            )
        else:
            ax.plot(
                xs[valid],
                y[valid],
                linestyle="-",
                linewidth=LINE_WIDTH,
                color=style["color"],
                marker=style["marker"],
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                markersize=MARKER_SIZE,
                alpha=LINE_ALPHA,
                zorder=3,
            )

    # Axes formatting.
    ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.45)
    ax.set_ylim(*ylim)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(format_percentage_point_tick))

    ax.tick_params(
        axis="x",
        length=0.0,
        width=0.0,
        color="black",
        labelcolor="black",
        labelsize=LABEL_SIZE,
        bottom=True,
        labelbottom=True,
    )

    ax.tick_params(
        axis="y",
        length=0.0,
        width=0.0,
        color="black",
        labelcolor="black",
        labelsize=LABEL_SIZE,
        left=True,
        labelleft=True,
    )

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color("0.15")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        color="0.89",
        linewidth=0.55,
        linestyle="-",
        zorder=0,
    )
    ax.set_axisbelow(True)


def draw_combined_delta_figure(
    model_names,
    applications,
    resume_counts,
    application_to_pool_count,
    max_n_trials=1000000,
    output_dir="outputs/societal",
    show_errorbars=False,
):
    """
    Draw the revised Figure 3.

    Layout:
        Panel a: Gender identity
            Hiring | Loan approval | Scholarship application

        Panel b: Sexual orientation
            Hiring | Loan approval | Scholarship application

    Y-axis:
        Delta selection rate in percentage points:
        Delta = SelectionRate_Minority - SelectionRate_Majority
    """
    set_nature_style()
    os.makedirs(output_dir, exist_ok=True)

    attribute_types = ["Gender Identity", "Sexual Orientation"]

    attribute_title_map = {
        "Gender Identity": "Gender identity",
        "Sexual Orientation": "Sexual orientation",
    }

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship allocation",
    }

    model_to_style = build_model_styles(model_names)

    delta_data = collect_delta_data(
        attribute_types=attribute_types,
        applications=applications,
        model_names=model_names,
        resume_counts=resume_counts,
        application_to_pool_count=application_to_pool_count,
        max_n_trials=max_n_trials,
        strict=True,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.2, 4.9),
        sharex=True,
        sharey=False,
    )

    plt.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.865,
        bottom=0.245,
        wspace=0.18,
        hspace=0.50,
    )

    for row_idx, attribute_type in enumerate(attribute_types):
        for col_idx, application in enumerate(applications):
            ax = axes[row_idx, col_idx]

            subplot_ylim = compute_subplot_ylim(
                delta_data=delta_data,
                attribute_type=attribute_type,
                application=application,
                model_names=model_names,
                include_ci=show_errorbars,
                step=2.0,
                min_abs=2.0,
                pad_frac=0.12,
                symmetric=True,
            )

            plot_delta_application_panel(
                ax=ax,
                attribute_type=attribute_type,
                application=application,
                model_names=model_names,
                delta_data=delta_data,
                model_to_style=model_to_style,
                resume_counts=resume_counts,
                ylim=subplot_ylim,
                show_errorbars=show_errorbars,
            )

            ax.set_title(
                application_title_map[application],
                fontsize=FONT_SIZE,
                fontweight="bold",
                pad=5,
            )

    # Shared axis labels.
    fig.supxlabel(
        "Number of candidates",
        fontsize=FONT_SIZE,
        y=0.145,
    )

    fig.supylabel(
        r"$\Delta$ selection rate "
        r"(societal minority $-$ societal majority, percentage points)",
        fontsize=FONT_SIZE,
        x=0.035,
    )

    # Panel labels.
    panel_labels = ["a", "b"]

    for row_idx, attribute_type in enumerate(attribute_types):
        row_pos = axes[row_idx, 0].get_position()

        fig.text(
            row_pos.x0,
            row_pos.y1 + 0.050,
            f"{panel_labels[row_idx]}  {attribute_title_map[attribute_type]}",
            ha="left",
            va="bottom",
            fontsize=FONT_SIZE + 0.6,
            fontweight="bold",
        )

    # Shared model legend outside plotting area.
    legend_handles = []

    for model_name in model_names:
        style = model_to_style[model_name]

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE + 0.5,
                markerfacecolor=style["color"],
                markeredgecolor="white",
                markeredgewidth=0.35,
                alpha=1.0,
                label=pretty_model_name(model_name),
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.25,
        handletextpad=0.45,
        fontsize=7.2,
    )

    base = "Figure3_societal_cross_candidate_delta"
    pdf_path = os.path.join(output_dir, base + ".pdf")
    png_path = os.path.join(output_dir, base + ".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    max_n_trials = 1000000

    model_names_order = [
        "msra-gpt-4o",
        "gpt-oss-120b",
        "Qwen3-235B-A22B-Instruct-2507",
        "Qwen3-Next-80B-A3B-Instruct",
        "GLM-4.5-Air",
        "gemma-3-27b-it",
        "Llama-3.3-70B-Instruct",
        "NVIDIA-Nemotron-Nano-12B-v2",
    ]

    applications = ["hiring", "loan", "edu"]

    application_to_pool_count = {
        "hiring": 200,
        "loan": 500,
        "edu": 500,
    }

    resume_counts = [2, 4, 6, 8, 10]

    draw_combined_delta_figure(
        model_names=model_names_order,
        applications=applications,
        resume_counts=resume_counts,
        application_to_pool_count=application_to_pool_count,
        max_n_trials=max_n_trials,
        output_dir="outputs/societal",
        show_errorbars=False,
    )