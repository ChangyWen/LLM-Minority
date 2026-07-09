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

# Reviewer-style semantic colors.
# Lines are colored by the qualitative pattern, while markers/linestyles
# identify models. This makes the main story readable without 48 small panels.
BEHAVIOR_COLORS = {
    "Minority favored": "#1B9E77",
    "Reverses as pool grows": "#E68613",
    "Majority favored": "#C9253C",
    "Near parity": "#9E9E9E",
}

BEHAVIOR_ORDER = [
    "Minority favored",
    "Reverses as pool grows",
    "Majority favored",
    "Near parity",
]

MINORITY_BG = "#EEF7F4"
MAJORITY_BG = "#FBF0F2"

# Manual y-axis limits for panels that need extra space for annotations.
# Key format: (attribute_type, application).
PANEL_YLIM_OVERRIDES = {
    ("Gender Identity", "hiring"): (-10.0, 10.0),
}


def build_model_styles(model_names):
    """
    Assign each model a distinct marker/linestyle.

    In the revised figure, color encodes the behavioral pattern rather than
    model identity. Model identity is encoded by marker and line style.
    """
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    linestyles = [
        "-",
        "--",
        "-.",
        ":",
        "-",
        "--",
        "-.",
        (0, (1.0, 2.0)),
    ]

    model_to_style = {}
    for idx, model_name in enumerate(model_names):
        model_to_style[model_name] = {
            "marker": markers[idx % len(markers)],
            "linestyle": linestyles[idx % len(linestyles)],
        }

    return model_to_style


def classify_behavior(
    y,
    near_mean_threshold=1.0,
    near_max_threshold=2.0,
    favored_point_threshold=1.0,
):
    """
    Classify one model line into a reviewer-style qualitative category.

    Parameters are in percentage points. The defaults are intentionally mild:
    - near parity: small average and maximum absolute deviations;
    - reverses: clear sign change as the pool grows;
    - minority/majority favored: most points lie above/below zero.

    You can override any individual classification via behavior_overrides in
    draw_combined_delta_figure().
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return "Near parity"

    mean_abs = float(np.mean(np.abs(y)))
    max_abs = float(np.max(np.abs(y)))
    mean_y = float(np.mean(y))

    if mean_abs <= near_mean_threshold and max_abs <= near_max_threshold:
        return "Near parity"

    first = float(y[0])
    last = float(y[-1])

    clear_positive_start = first > favored_point_threshold
    clear_negative_start = first < -favored_point_threshold
    clear_positive_end = last > favored_point_threshold
    clear_negative_end = last < -favored_point_threshold

    if (clear_positive_start and clear_negative_end) or (
        clear_negative_start and clear_positive_end
    ):
        return "Reverses as pool grows"

    positive_frac = float(np.mean(y > favored_point_threshold))
    negative_frac = float(np.mean(y < -favored_point_threshold))

    if positive_frac >= 0.60 and mean_y > 0:
        return "Minority favored"

    if negative_frac >= 0.60 and mean_y < 0:
        return "Majority favored"

    # Mixed lines that cross zero but not strongly enough for the stricter
    # rule above are still useful to highlight as reversals.
    if np.nanmin(y) < -favored_point_threshold and np.nanmax(y) > favored_point_threshold:
        return "Reverses as pool grows"

    if mean_y > favored_point_threshold:
        return "Minority favored"

    if mean_y < -favored_point_threshold:
        return "Majority favored"

    return "Near parity"


def get_behavior_for_line(
    attribute_type,
    application,
    model_name,
    y,
    behavior_overrides=None,
):
    """
    Return the behavior category for one line.

    behavior_overrides can be used for exact manual control, e.g.,
        behavior_overrides = {
            ("Gender Identity", "hiring", "msra-gpt-4o"): "Reverses as pool grows",
        }
    """
    key = (attribute_type, application, model_name)
    if behavior_overrides is not None and key in behavior_overrides:
        return behavior_overrides[key]
    return classify_behavior(y)


def compute_subplot_ylim_reviewer(
    delta_data,
    attribute_type,
    application,
    model_names,
    include_ci=False,
    step=2.5,
    min_pos=2.0,
    min_neg=2.0,
    pad_frac=0.12,
):
    """
    Compute reviewer-style, non-symmetric y-limits for one subplot.

    The original delta figure used symmetric y-limits around zero. The reviewer
    illustration uses non-symmetric ranges to save space, while keeping zero
    visible and leaving enough room for both positive and negative shaded zones.
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
        return (-min_neg, min_pos)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, 1.0)

    lower = min(vmin - pad_frac * span, -min_neg)
    upper = max(vmax + pad_frac * span, min_pos)

    lower = math.floor(lower / step) * step
    upper = math.ceil(upper / step) * step

    if lower >= 0:
        lower = -min_neg
    if upper <= 0:
        upper = min_pos

    return (lower, upper)


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
    behavior_overrides=None,
    annotate_favored_regions=False,
):
    """
    Draw one subplot: one attribute x one application.

    Each line corresponds to one model. Line color encodes the qualitative
    behavioral pattern; marker/linestyle encodes the model.
    """
    xs = np.asarray(resume_counts, dtype=float)

    ax.set_ylim(*ylim)

    # Positive/negative regions.
    ax.axhspan(0, ylim[1], color=MINORITY_BG, zorder=0)
    ax.axhspan(ylim[0], 0, color=MAJORITY_BG, zorder=0)

    # Horizontal parity line.
    ax.axhline(
        0,
        color="0.10",
        linewidth=0.85,
        linestyle=(0, (3.0, 2.2)),
        zorder=2,
    )

    for model_name in model_names:
        item = delta_data[(attribute_type, application, model_name)]

        y = item["delta"]
        ci_low = item["ci_low"]
        ci_high = item["ci_high"]
        valid = np.isfinite(y)

        if not np.any(valid):
            continue

        behavior = get_behavior_for_line(
            attribute_type=attribute_type,
            application=application,
            model_name=model_name,
            y=y,
            behavior_overrides=behavior_overrides,
        )

        color = BEHAVIOR_COLORS[behavior]
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
                fmt=style["marker"],
                linestyle=style["linestyle"],
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.35,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                elinewidth=0.45,
                capsize=1.6,
                capthick=0.45,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax.plot(
                xs[valid],
                y[valid],
                linestyle=style["linestyle"],
                linewidth=LINE_WIDTH,
                color=color,
                marker=style["marker"],
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.35,
                markersize=MARKER_SIZE,
                alpha=0.95,
                zorder=3,
            )

    # Axes formatting.
    ax.set_xlim(xs[0] - 0.40, xs[-1] + 0.40)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(format_percentage_point_tick))

    ax.tick_params(
        axis="x",
        length=0.0,
        width=0.0,
        labelsize=LABEL_SIZE,
        bottom=True,
        labelbottom=True,
    )

    ax.tick_params(
        axis="y",
        length=0.0,
        width=0.0,
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

    # ax.grid(
    #     axis="y",
    #     color="white",
    #     linewidth=0.75,
    #     linestyle="-",
    #     zorder=1,
    # )
    ax.set_axisbelow(True)

    if annotate_favored_regions:
        ax.text(
            0.06,
            0.90,
            "societal minority favored",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=LABEL_SIZE + 0.4,
            fontweight="bold",
            fontstyle="italic",
            color=BEHAVIOR_COLORS["Minority favored"],
        )
        ax.text(
            0.06,
            0.09,
            "societal majority favored",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=LABEL_SIZE + 0.4,
            fontweight="bold",
            fontstyle="italic",
            color=BEHAVIOR_COLORS["Majority favored"],
        )


def draw_combined_delta_figure(
    model_names,
    applications,
    resume_counts,
    application_to_pool_count,
    max_n_trials=1000000,
    output_dir="outputs/societal",
    show_errorbars=False,
    behavior_overrides=None,
):
    """
    Draw the reviewer-style revised Figure 3.

    Layout:
        Rows:    gender identity, sexual orientation
        Columns: hiring, loan approval, scholarship allocation

    Y-axis:
        Delta selection rate in percentage points:
        Delta = SelectionRate_Minority - SelectionRate_Majority

    Visual encoding:
        Color:       qualitative behavior category
        Marker/line: model identity
        Background:  positive = societal minority favored;
                     negative = societal majority favored
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

    application_subtitle_map = {
        "hiring": "mixed; preference often\nweakens or reverses",
        "loan": "majority favored,\ngrowing with pool size",
        "edu": "minority favored,\npersists under comparison",
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
        figsize=(9.4, 4.95),
        sharex=True,
        sharey=False,
    )

    plt.subplots_adjust(
        left=0.125,
        right=0.985,
        top=0.800,
        bottom=0.245,
        wspace=0.24,
        hspace=0.34,
    )

    for row_idx, attribute_type in enumerate(attribute_types):
        for col_idx, application in enumerate(applications):
            ax = axes[row_idx, col_idx]

            subplot_ylim = compute_subplot_ylim_reviewer(
                delta_data=delta_data,
                attribute_type=attribute_type,
                application=application,
                model_names=model_names,
                include_ci=show_errorbars,
                step=2.5,
                min_pos=2.0,
                min_neg=2.0,
                pad_frac=0.12,
            )

            subplot_ylim = PANEL_YLIM_OVERRIDES.get(
                (attribute_type, application),
                subplot_ylim,
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
                behavior_overrides=behavior_overrides,
                annotate_favored_regions=(row_idx == 0 and col_idx == 0),
            )

            if row_idx == 0:
                ax.set_title(
                    application_title_map[application],
                    fontsize=FONT_SIZE + 2.0,
                    fontweight="bold",
                    pad=47,
                )

                ax.text(
                    0.5,
                    1.155,
                    application_subtitle_map[application],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZE - 0.8,
                    fontstyle="italic",
                    color="0.35",
                    linespacing=1.05,
                )

    # Shared axis labels.
    fig.supxlabel(
        "Number of candidates in pool",
        fontsize=FONT_SIZE + 0.3,
        y=0.205,
    )

    fig.supylabel(
        f"Group-level selection-rate different: minority − majority (pp)",
        fontsize=FONT_SIZE + 0.3,
        x=0.025,
    )

    # Row labels.
    for row_idx, row_label in enumerate(["Gender identity", "Sexual orientation"]):
        row_pos = axes[row_idx, 0].get_position()
        row_center_y = 0.5 * (row_pos.y0 + row_pos.y1)
        fig.text(
            0.077,
            row_center_y,
            row_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=FONT_SIZE + 0.5,
            fontweight="bold",
        )

    # Behavior legend: semantic color meaning.
    behavior_handles = [
        Line2D(
            [0],
            [0],
            color=BEHAVIOR_COLORS[name],
            linestyle="-",
            linewidth=2.2,
            label=name,
        )
        for name in BEHAVIOR_ORDER
    ]

    fig.legend(
        handles=behavior_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.155),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.35,
        handletextpad=0.55,
        title="Behavior category",
        title_fontsize=FONT_SIZE - 0.3,
        fontsize=FONT_SIZE - 0.6,
    )

    # Model legend: marker and line style meaning.
    model_handles = []
    for model_name in model_names:
        style = model_to_style[model_name]
        model_handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=LINE_WIDTH + 0.25,
                markersize=MARKER_SIZE + 0.4,
                markerfacecolor="0.35",
                markeredgecolor="white",
                markeredgewidth=0.35,
                label=pretty_model_name(model_name),
            )
        )

    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.020),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.35,
        handletextpad=0.55,
        fontsize=FONT_SIZE - 1.3,
    )

    base = "Figure3_societal_cross_candidate_delta_reviewer_style"
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

    # Optional manual corrections if you want a line to use a specific reviewer
    # category after inspecting the generated plot. Leave empty by default.
    behavior_overrides = {
        # Example:
        # ("Gender Identity", "hiring", "msra-gpt-4o"): "Reverses as pool grows",
    }

    draw_combined_delta_figure(
        model_names=model_names_order,
        applications=applications,
        resume_counts=resume_counts,
        application_to_pool_count=application_to_pool_count,
        max_n_trials=max_n_trials,
        output_dir="outputs/societal",
        show_errorbars=False,
        behavior_overrides=behavior_overrides,
    )
