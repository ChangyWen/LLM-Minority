import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}

FIG_FONT_SIZE = 9.5
MARKER_SIZE = 4.2


# ============================================================
# Statistical helpers
# ============================================================

def p_to_stars(p):
    """
    Convert p-value to significance stars.
    """
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


def compute_results(attribute_type, file_name):
    """
    Compute mean scores and 95% CIs for minority and majority groups.

    The p-value tests:
        H0: minority and majority score distributions are the same
        H1: minority and majority score distributions are different

    Test: two-sided Mann-Whitney U test.
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
        alternative="two-sided",
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
    """
    Compact, clean plotting style suitable for Nature-style multi-panel figures.
    """
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

        "axes.titlesize": 6,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 3,
        "ytick.labelsize": 3,
        "legend.fontsize": 8.5,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": 1.25,
    })


def pretty_model_name(model_key):
    """
    Shorter display names for compact multi-panel figures.
    """
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


def add_sig_bracket(
    ax,
    x1,
    x2,
    y,
    text,
    bar_height,
    text_offset,
    linewidth=0.8,
):
    """
    Add a significance bracket between two x positions.
    """
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + bar_height, y + bar_height, y],
        color="black",
        linewidth=linewidth,
        clip_on=False,
        zorder=5,
    )

    ax.text(
        (x1 + x2) / 2,
        y + bar_height + text_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=FIG_FONT_SIZE,
        color="black",
        clip_on=False,
        zorder=6,
    )


# ============================================================
# Panel plotting
# ============================================================

def plot_model_panel_societal(
    ax,
    application,
    model_name,
    attribute_types,
    attribute_to_color,
):
    """
    Draw one panel:
        one application x one model.

    X-axis:
        Gender Identity, Sexual Orientation

    For each attribute:
        minority and majority scores with 95% CI.
    """
    minority_marker = "D"
    majority_marker = "^"

    x_gap = 0.2
    x_base = np.arange(len(attribute_types), dtype=float) * x_gap
    dodge = 0.04

    x_minority = x_base - dodge
    x_majority = x_base + dodge

    panel_low_values = []
    panel_high_values = []
    bracket_info = []

    attribute_tick_labels = {
        "Gender Identity": "GI",
        "Sexual Orientation": "SO",
    }

    for i, attribute_type in enumerate(attribute_types):
        file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"

        if not os.path.exists(file_name):
            print(f"[Warning] Missing file: {file_name}")
            continue

        res = compute_results(attribute_type, file_name)

        if res is None:
            print(f"[Warning] No valid result: {application}, {attribute_type}, {model_name}")
            continue

        color = attribute_to_color[attribute_type]

        min_res = res["minority"]
        maj_res = res["majority"]

        # Minority
        min_mean = float(min_res["mean"])
        min_low = float(min_res["ci_low"])
        min_high = float(min_res["ci_high"])

        ax.errorbar(
            x_minority[i],
            min_mean,
            yerr=[[min_mean - min_low], [min_high - min_mean]],
            marker=minority_marker,
            markersize=MARKER_SIZE,
            capsize=2.2,
            capthick=0.8,
            elinewidth=0.8,
            linestyle="",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=3,
        )

        # Majority
        maj_mean = float(maj_res["mean"])
        maj_low = float(maj_res["ci_low"])
        maj_high = float(maj_res["ci_high"])

        ax.errorbar(
            x_majority[i],
            maj_mean,
            yerr=[[maj_mean - maj_low], [maj_high - maj_mean]],
            marker=majority_marker,
            markersize=MARKER_SIZE,
            capsize=2.2,
            capthick=0.8,
            elinewidth=0.8,
            linestyle="",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=3,
        )

        panel_low_values.extend([min_low, maj_low])
        panel_high_values.extend([min_high, maj_high])

        p_value = float(res["p_value"])
        stars = p_to_stars(p_value)

        y_pair_top = max(min_high, maj_high)
        bracket_info.append({
            "x1": x_minority[i],
            "x2": x_majority[i],
            "y_top": y_pair_top,
            "stars": stars,
        })

        print(
            f"{application} | {model_name} | {attribute_type}: "
            f"minority vs majority, two-sided P={p_value:.4g}"
        )

    # Panel-specific y-limits with space for brackets
    if panel_high_values:
        data_min = min(panel_low_values)
        data_max = max(panel_high_values)
    else:
        data_min, data_max = 0.0, 1.0

    data_span = max(data_max - data_min, 0.5)

    bracket_offset = 0.08 * data_span
    bracket_height = 0.035 * data_span
    text_offset = 0.020 * data_span

    bracket_tops = []

    for b in bracket_info:
        y_bracket = b["y_top"] + bracket_offset

        add_sig_bracket(
            ax=ax,
            x1=b["x1"],
            x2=b["x2"],
            y=y_bracket,
            text=b["stars"],
            bar_height=bracket_height,
            text_offset=text_offset,
        )

        bracket_tops.append(y_bracket + bracket_height + text_offset)

    y_lower = data_min - 0.12 * data_span
    y_upper = max([data_max] + bracket_tops) + 0.14 * data_span
    ax.set_ylim(y_lower, y_upper)

    # Axis formatting
    ax.set_xticks(x_base)
    ax.set_xticklabels([attribute_tick_labels.get(a, a) for a in attribute_types])
    ax.set_xlim(x_base[0] - 0.1, x_base[-1] + 0.1)

    for tick_label, attribute_type in zip(ax.get_xticklabels(), attribute_types):
        tick_label.set_color(attribute_to_color[attribute_type])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))

    ax.tick_params(
        axis="both",
        # direction="out",
        length=0.0,
        width=0.0,
        # color="black",
        labelcolor="black",
        labelsize=9.0,
    )

    # Re-apply x tick colors after tick_params
    for tick_label, attribute_type in zip(ax.get_xticklabels(), attribute_types):
        tick_label.set_color(attribute_to_color[attribute_type])

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
    Draw one big Nature-style figure:

        Hiring
        Loan approval
        Scholarship

    Each application block contains eight model panels arranged as 2 x 4.
    """
    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship",
    }

    attribute_to_color = {
        "Gender Identity": "#009E73",     # green
        "Sexual Orientation": "#CC79A7",  # purple
    }

    fig = plt.figure(figsize=(8.5, 9.0))

    outer_gs = fig.add_gridspec(
        3,
        1,
        left=0.105,
        right=0.990,
        bottom=0.110,
        top=0.955,
        hspace=0.28,
    )

    all_axes = {}

    for app_idx, application in enumerate(applications):
        inner_gs = outer_gs[app_idx].subgridspec(
            2,
            4,
            wspace=0.18,
            hspace=0.45,
        )

        axes = np.empty((2, 4), dtype=object)
        all_axes[application] = axes

        for idx, model_name in enumerate(model_names):
            row = idx // 4
            col = idx % 4

            ax = fig.add_subplot(inner_gs[row, col])
            axes[row, col] = ax

            plot_model_panel_societal(
                ax=ax,
                application=application,
                model_name=model_name,
                attribute_types=attribute_types,
                attribute_to_color=attribute_to_color,
            )

            ax.set_title(
                pretty_model_name(model_name),
                loc="center",
                pad=4,
                fontsize=FIG_FONT_SIZE,
            )

    # ------------------------------------------------------------
    # Shared labels
    # ------------------------------------------------------------
    fig.supxlabel(
        "Attribute",
        fontsize=FIG_FONT_SIZE,
        y=0.065,
    )

    fig.supylabel(
        "Score",
        fontsize=FIG_FONT_SIZE,
        x=0.05,
    )

    # ------------------------------------------------------------
    # Application block titles
    # ------------------------------------------------------------
    for i, application in enumerate(applications):
        axes = all_axes[application]

        pos_left = axes[0, 0].get_position()
        pos_right = axes[0, 3].get_position()

        x0 = pos_left.x0
        x1 = pos_right.x1
        y1 = pos_left.y1

        fig.text(
            (x0 + x1) / 2,
            y1 + 0.020,
            application_title_map[application],
            ha="center",
            va="bottom",
            fontsize=FIG_FONT_SIZE,
            fontweight="bold",
        )

    # ------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------
    legend_handles = [
        # text-only entries for model names
        Line2D(
            [0], [0],
            linestyle="",
            marker=None,
            markersize=0.0,
            color="#009E73",
            label="GI: Gender identity",
        ),
        Line2D(
            [0], [0],
            linestyle="",
            marker=None,
            markersize=0.0,
            color="#CC79A7",
            label="SO: Sexual orientation",
        ),

        Line2D(
            [0], [0],
            marker="D",
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=5.0,
            label="Minority",
        ),
        Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=5.0,
            label="Majority",
        ),
    ]

    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=4,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.40,
        fontsize=FIG_FONT_SIZE,
    )

    # Hide the first two dummy handles
    for h in leg.legend_handles[:2]:
        h.set_visible(False)

    # Color the first two legend texts
    legend_texts = leg.get_texts()
    legend_texts[0].set_color("#009E73")   # GI text
    legend_texts[1].set_color("#CC79A7")   # SO text

    # Save
    base = "societal_all_applications_nature_style"
    pdf_path = os.path.join(output_dir, base + ".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

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