import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}

FIG_FONT_SIZE = 9.5
MARKER_SIZE = 4.2

# Matches the current Fig. 2 caption:
# H1: societal minority candidates receive higher scores than societal majority candidates.
# Change to "two-sided" if you want to keep the old statistical test.
MANN_WHITNEY_ALTERNATIVE = "greater"

SHOW_ERRORBARS = True
SHOW_PANEL_SUMMARY = True


# ============================================================
# Statistical helpers
# ============================================================

def p_to_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def t_ci(scores, confidence=0.95):
    scores = np.asarray(scores, dtype=float)

    mean = np.mean(scores)
    sem = stats.sem(scores)
    df = len(scores) - 1

    if df <= 0 or np.isnan(sem):
        return mean, mean, mean

    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    lower = mean - t_crit * sem
    upper = mean + t_crit * sem

    return mean, lower, upper


def compute_results(attribute_type, file_name, alternative=MANN_WHITNEY_ALTERNATIVE):
    """
    Compute mean scores and 95% CIs for minority and majority groups.

    Default p-value test:
        H0: minority scores are not greater than majority scores
        H1: minority scores are greater than majority scores

    Test: Mann-Whitney U test.
    """
    minority_scores = []
    majority_scores = []

    minority_attributes = type_to_minority_attributes[attribute_type]

    with open(file_name, "r") as f:
        for line in f:
            item = json.loads(line)
            attribute = item["attribute"]
            score = item["score"]

            if attribute in minority_attributes:
                minority_scores.append(score)
            else:
                majority_scores.append(score)

    minority_scores = np.asarray(minority_scores, dtype=float)
    majority_scores = np.asarray(majority_scores, dtype=float)

    if len(minority_scores) == 0 or len(majority_scores) == 0:
        return None

    minority_mean, minority_ci_low, minority_ci_high = t_ci(minority_scores)
    majority_mean, majority_ci_low, majority_ci_high = t_ci(majority_scores)

    stat, p_value = stats.mannwhitneyu(
        minority_scores,
        majority_scores,
        alternative=alternative,
    )

    return {
        "minority": {
            "mean": minority_mean,
            "ci_low": minority_ci_low,
            "ci_high": minority_ci_high,
        },
        "majority": {
            "mean": majority_mean,
            "ci_low": majority_ci_low,
            "ci_high": majority_ci_high,
        },
        "p_value": p_value,
    }


# ============================================================
# Style helpers
# ============================================================

def set_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        # Keep text editable in Illustrator / Inkscape
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "figure.dpi": 150,
        "savefig.dpi": 600,

        "axes.linewidth": 0.7,
        "axes.edgecolor": "0.15",
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.titlesize": FIG_FONT_SIZE,
        "axes.labelsize": FIG_FONT_SIZE,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": 1.15,
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


def get_model_style_map(model_names):
    """
    Monochrome model styles.

    The reviewer suggested avoiding visually salient attribute colors.
    Here, attributes are separated by panels, and models are distinguished
    by line styles and markers.
    """
    style_cycle = [
        {"linestyle": "-",                 "marker": "o"},
        {"linestyle": "--",                "marker": "s"},
        {"linestyle": "-.",                "marker": "^"},
        {"linestyle": ":",                 "marker": "D"},
        {"linestyle": (0, (5, 1)),          "marker": "v"},
        {"linestyle": (0, (3, 1, 1, 1)),    "marker": "P"},
        {"linestyle": (0, (1, 1)),          "marker": "X"},
        {"linestyle": (0, (5, 2, 1, 2)),    "marker": "h"},
    ]

    # Neutral greys; not used to encode attributes.
    grey_cycle = ["0.05", "0.15", "0.25", "0.35", "0.45", "0.55", "0.65", "0.75"]

    model_style_map = {}
    for i, model_name in enumerate(model_names):
        style = style_cycle[i % len(style_cycle)].copy()
        style["color"] = grey_cycle[i % len(grey_cycle)]
        model_style_map[model_name] = style

    return model_style_map


# ============================================================
# Panel plotting
# ============================================================

def plot_scenario_attribute_panel(
    ax,
    application,
    attribute_type,
    model_names,
    model_style_map,
    show_errorbars=True,
    show_panel_summary=True,
):
    """
    Draw one panel:
        one scenario x one attribute.

    X-axis:
        1 = societal majority
        2 = societal minority

    Each line:
        one model, connecting majority score to minority score.
    """
    x = np.array([1.0, 2.0])
    xtick_labels = ["Majority", "Minority"]

    panel_low_values = []
    panel_high_values = []

    n_valid = 0
    n_minority_higher = 0
    n_significant = 0

    for model_name in model_names:
        file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"

        if not os.path.exists(file_name):
            print(f"[Warning] Missing file: {file_name}")
            continue

        res = compute_results(attribute_type, file_name)

        if res is None:
            print(f"[Warning] No valid result: {application}, {attribute_type}, {model_name}")
            continue

        min_res = res["minority"]
        maj_res = res["majority"]

        maj_mean = float(maj_res["mean"])
        maj_low = float(maj_res["ci_low"])
        maj_high = float(maj_res["ci_high"])

        min_mean = float(min_res["mean"])
        min_low = float(min_res["ci_low"])
        min_high = float(min_res["ci_high"])

        y = np.array([maj_mean, min_mean])
        yerr = np.array([
            [maj_mean - maj_low, min_mean - min_low],
            [maj_high - maj_mean, min_high - min_mean],
        ])

        style = model_style_map[model_name]
        color = style["color"]

        ax.plot(
            x,
            y,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=MARKER_SIZE,
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            linewidth=1.15,
            alpha=0.95,
            zorder=3,
        )

        if show_errorbars:
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="none",
                ecolor=color,
                elinewidth=0.65,
                capsize=1.8,
                capthick=0.65,
                alpha=0.75,
                zorder=2,
            )

        panel_low_values.extend([maj_low, min_low])
        panel_high_values.extend([maj_high, min_high])

        p_value = float(res["p_value"])
        stars = p_to_stars(p_value)

        n_valid += 1
        if min_mean > maj_mean:
            n_minority_higher += 1
        if p_value < 0.05:
            n_significant += 1

        print(
            f"{application} | {attribute_type} | {pretty_model_name(model_name)}: "
            f"majority={maj_mean:.3f}, minority={min_mean:.3f}, "
            f"P({MANN_WHITNEY_ALTERNATIVE})={p_value:.4g}, {stars}"
        )

    # ------------------------------------------------------------
    # Y-limits
    # ------------------------------------------------------------
    if panel_low_values:
        data_min = min(panel_low_values)
        data_max = max(panel_high_values)
    else:
        data_min, data_max = 0.0, 1.0

    data_span = max(data_max - data_min, 0.35)
    y_lower = data_min - 0.12 * data_span
    y_upper = data_max + 0.18 * data_span
    ax.set_ylim(y_lower, y_upper)

    # ------------------------------------------------------------
    # Main-message summary inside panel
    # ------------------------------------------------------------
    if show_panel_summary and n_valid > 0:
        summary_text = f"↑ {n_minority_higher}/{n_valid} models"
        if n_significant > 0:
            summary_text += f"\nP<0.05: {n_significant}/{n_valid}"

        ax.text(
            0.04,
            0.95,
            summary_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color="0.20",
        )

    # ------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(0.78, 2.22)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))

    ax.grid(
        axis="y",
        color="0.90",
        linewidth=0.55,
        linestyle="-",
        zorder=0,
    )

    ax.tick_params(
        axis="both",
        direction="out",
        length=2.5,
        width=0.7,
        color="0.15",
        labelcolor="black",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


# ============================================================
# Big figure drawing
# ============================================================

def draw_societal_super_figure(
    applications,
    attribute_types,
    model_names,
    output_dir="outputs/societal",
):
    """
    Revised Fig. 2.

    Layout:
        3 scenarios x 2 attributes = 6 panels

    Each panel:
        eight model lines from societal majority to societal minority.
    """
    set_nature_style()
    os.makedirs(output_dir, exist_ok=True)

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship application",
    }

    attribute_title_map = {
        "Gender Identity": "Gender identity (GI)",
        "Sexual Orientation": "Sexual orientation (SO)",
    }

    model_style_map = get_model_style_map(model_names)

    fig, axes = plt.subplots(
        nrows=len(applications),
        ncols=len(attribute_types),
        figsize=(7.2, 5.8),
        sharex=True,
        sharey=False,
    )

    if len(applications) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(attribute_types) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, application in enumerate(applications):
        for col_idx, attribute_type in enumerate(attribute_types):
            ax = axes[row_idx, col_idx]

            plot_scenario_attribute_panel(
                ax=ax,
                application=application,
                attribute_type=attribute_type,
                model_names=model_names,
                model_style_map=model_style_map,
                show_errorbars=SHOW_ERRORBARS,
                show_panel_summary=SHOW_PANEL_SUMMARY,
            )

            # Column titles: attributes
            if row_idx == 0:
                ax.set_title(
                    attribute_title_map.get(attribute_type, attribute_type),
                    fontsize=FIG_FONT_SIZE,
                    fontweight="bold",
                    pad=6,
                )

            # Row labels: scenarios
            if col_idx == 0:
                ax.text(
                    -0.38,
                    0.5,
                    application_title_map.get(application, application),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=FIG_FONT_SIZE,
                    fontweight="bold",
                )

    # ------------------------------------------------------------
    # Shared axis labels
    # ------------------------------------------------------------
    fig.supxlabel(
        "Societal group",
        fontsize=FIG_FONT_SIZE,
        y=0.090,
    )

    fig.supylabel(
        "Assessment score",
        fontsize=FIG_FONT_SIZE,
        x=0.035,
    )

    # ------------------------------------------------------------
    # Shared legend: models
    # ------------------------------------------------------------
    legend_handles = []

    for model_name in model_names:
        style = model_style_map[model_name]

        legend_handles.append(
            Line2D(
                [0], [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
                linewidth=1.15,
                markersize=4.2,
                label=pretty_model_name(model_name),
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        handletextpad=0.45,
        fontsize=7.8,
    )

    fig.subplots_adjust(
        left=0.120,
        right=0.990,
        bottom=0.165,
        top=0.915,
        wspace=0.260,
        hspace=0.360,
    )

    # Save
    base = "societal_individual_assessment_six_panel"
    for ext in ["pdf", "png", "svg"]:
        path = os.path.join(output_dir, f"{base}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    applications = ["hiring", "loan", "edu"]

    attribute_types = [
        "Gender Identity",
        "Sexual Orientation",
    ]

    model_names = [
        "msra-gpt-4o",
        "gpt-oss-120b",
        "Qwen3-235B-A22B-Instruct-2507",
        "Qwen3-Next-80B-A3B-Instruct",
        "GLM-4.5-Air",
        "gemma-3-27b-it",
        "Llama-3.3-70B-Instruct",
        "NVIDIA-Nemotron-Nano-12B-v2",
    ]

    draw_societal_super_figure(
        applications=applications,
        attribute_types=attribute_types,
        model_names=model_names,
        output_dir="outputs/societal",
    )