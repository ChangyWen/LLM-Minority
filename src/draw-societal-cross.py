import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
CI_ALPHA = 0.35

# ============================================================
# Shared style
# ============================================================

def set_nature_style():
    """
    Compact, clean plotting style consistent with your previous figures.
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
        "axes.labelsize": 6.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 8.5,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": 1.25,
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


def format_percent_tick(v, pos):
    return f"{v * 100:.0f}"


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
    Compute overall selection rates for Minority and Majority.

    Also performs a two-sided binomial test:

        H0: p_Minority = p_Majority = 0.5
        H1: p_Minority != p_Majority

    Since each selected candidate is either Minority or Majority,
    this is equivalent to testing whether the number of Minority
    selections differs from half of all selections.
    """
    attr_value_to_hit_count = defaultdict(int)
    n_trials = 0

    with open(file_name, "r") as f:
        for line in f:
            item = json.loads(line)
            attributes = item["attributes"]

            if "Asian" in attributes:
                continue

            suggested_candidate_id = item["suggested_candidate_id"]

            if n_trials >= max_n_trials:
                break
            n_trials += 1

            suggested_candidate_attr_value = attributes[suggested_candidate_id]

            minority_attr_values = type_to_minority_attributes[attribute_type]
            if suggested_candidate_attr_value in minority_attr_values:
                suggested_candidate_attr_value = "Minority"
            else:
                suggested_candidate_attr_value = "Majority"

            attr_value_to_hit_count[suggested_candidate_attr_value] += 1

    # Ensure both keys exist, even if one group is never selected.
    minority_count = attr_value_to_hit_count["Minority"]
    majority_count = attr_value_to_hit_count["Majority"]

    results = {}

    for attr_value, hit_count in [
        ("Minority", minority_count),
        ("Majority", majority_count),
    ]:
        hit_rate = hit_count / n_trials
        ci_low, ci_high = wilson_ci(hit_count, n_trials)

        results[attr_value] = {
            "hit_count": hit_count,
            "total_count": n_trials,
            "hit_rate": hit_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    # Two-sided significance test: Minority vs Majority selection rates.
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


# ============================================================
# Panel plotting
# ============================================================


def plot_model_panel(
    ax,
    application,
    attribute_type,
    resume_counts,
    model_name,
    pool_count,
    max_n_trials,
    minority_color,
    majority_color,
):
    """
    Draw one model panel.

    Stars above each x-position indicate whether the Minority and
    Majority selection rates are significantly different at that
    candidate-pool size.

        *   P < 0.05
        **  P < 0.01
        *** P < 0.001
    """

    minority_marker = "^"
    majority_marker = "o"

    xs = np.array(resume_counts, dtype=float)

    # Add 50% parity reference line
    ax.axhline(
        0.5,
        color="0.35",
        linestyle="--",
        linewidth=0.95,
        alpha=0.85,
        zorder=1,
    )

    y_min, lo_min, hi_min = [], [], []
    y_maj, lo_maj, hi_maj = [], [], []
    p_values = []

    for rc in resume_counts:
        file_path = (
            f"outputs/{application}/contextual/"
            f"{attribute_type}/{model_name}_{rc}_{pool_count}.jsonl"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        results = compute_results(
            file_name=file_path,
            attribute_type=attribute_type,
            max_n_trials=max_n_trials,
        )

        if "Minority" not in results or "Majority" not in results:
            raise KeyError(
                f'Expected both "Minority" and "Majority" in results, '
                f"got keys={list(results.keys())} for file={file_path}"
            )

        y_min.append(results["Minority"]["hit_rate"])
        lo_min.append(results["Minority"]["ci_low"])
        hi_min.append(results["Minority"]["ci_high"])

        y_maj.append(results["Majority"]["hit_rate"])
        lo_maj.append(results["Majority"]["ci_low"])
        hi_maj.append(results["Majority"]["ci_high"])

        p_values.append(results["significance"]["p_value"])

    y_min = np.asarray(y_min, dtype=float)
    lo_min = np.asarray(lo_min, dtype=float)
    hi_min = np.asarray(hi_min, dtype=float)

    y_maj = np.asarray(y_maj, dtype=float)
    lo_maj = np.asarray(lo_maj, dtype=float)
    hi_maj = np.asarray(hi_maj, dtype=float)

    p_values = np.asarray(p_values, dtype=float)

    yerr_min = np.vstack([y_min - lo_min, hi_min - y_min])
    yerr_maj = np.vstack([y_maj - lo_maj, hi_maj - y_maj])

    # Minority
    line_min, caplines_min, barlines_min = ax.errorbar(
        xs,
        y_min,
        yerr=yerr_min,
        fmt=minority_marker + "-",
        color=minority_color,
        markerfacecolor=minority_color,
        markeredgecolor=minority_color,
        markeredgewidth=1.0,
        markersize=MARKER_SIZE,
        linewidth=1.15,
        elinewidth=0.75,
        capsize=2.0,
        capthick=0.75,
        zorder=3,
    )

    # Majority
    line_maj, caplines_maj, barlines_maj = ax.errorbar(
        xs,
        y_maj,
        yerr=yerr_maj,
        fmt=majority_marker + "-",
        color=majority_color,
        markerfacecolor=majority_color,
        markeredgecolor=majority_color,
        markeredgewidth=1.0,
        markersize=MARKER_SIZE,
        linewidth=1.15,
        elinewidth=0.75,
        capsize=2.0,
        capthick=0.75,
        zorder=3,
    )

    for cap in caplines_min + caplines_maj:
        cap.set_alpha(CI_ALPHA)

    for bar in barlines_min + barlines_maj:
        bar.set_alpha(CI_ALPHA)

    # ------------------------------------------------------------
    # Significance stars
    # ------------------------------------------------------------
    panel_low = min(np.min(lo_min), np.min(lo_maj))
    panel_high = max(np.max(hi_min), np.max(hi_maj))
    panel_span = max(panel_high - panel_low, 0.05)

    star_offset = 0.065 * panel_span
    star_positions = []

    for x, p, h_min_i, h_maj_i in zip(xs, p_values, hi_min, hi_maj):
        stars = p_to_stars(p)

        if stars:
            y_star = max(h_min_i, h_maj_i) + star_offset
            star_positions.append(y_star)

            ax.text(
                x,
                y_star,
                stars,
                ha="center",
                va="bottom",
                fontsize=7,
                color="0.10",
                clip_on=False,
                zorder=5,
            )

    # Add enough vertical space for significance stars.
    if star_positions:
        y_upper = max(max(star_positions), panel_high) + 0.12 * panel_span
    else:
        y_upper = panel_high + 0.12 * panel_span

    y_lower = max(0.0, panel_low - 0.10 * panel_span)
    ax.set_ylim(y_lower, y_upper)

    # Axes formatting
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in resume_counts])
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(format_percent_tick))

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

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color("0.15")

    ax.set_axisbelow(True)


# ============================================================
# Big figure drawing
# ============================================================

def draw_attribute_big_figure(
    attribute_type,
    model_names,
    applications,
    resume_counts,
    application_to_pool_count,
    max_n_trials=1000000,
    output_dir="outputs/societal",
):
    """
    Draw one big figure for one attribute type.

    Layout:
        Hiring
        Loan approval
        Scholarship application

    Each application block contains 8 model panels arranged as 2 x 4.
    """

    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship allocation",
    }

    attribute_panel_title_map = {
        "Gender Identity": ("a", "Evaluation results by gender identity"),
        "Sexual Orientation": ("b", "Evaluation results by sexual orientation"),
    }

    # Use the same professional color style as your previous figures
    minority_color = "orange"   # vermillion
    majority_color = "blue"   # blue

    fig = plt.figure(figsize=(9.5, 10))

    panel_letter, panel_title = attribute_panel_title_map[attribute_type]

    fig.text(
        0.08,
        0.970,
        panel_letter,
        ha="left",
        va="top",
        fontsize=FONT_SIZE + 1.0,
        fontweight="bold",
    )

    fig.text(
        0.08 + 0.025,
        0.970,
        panel_title,
        ha="left",
        va="top",
        fontsize=FONT_SIZE + 1.0,
        fontweight="bold",
    )

    outer_gs = fig.add_gridspec(
        3,
        1,
        left=0.125,
        right=0.875,
        bottom=0.090,
        top=0.915,
        hspace=0.25,
    )

    all_axes = {}

    for app_idx, application in enumerate(applications):
        inner_gs = outer_gs[app_idx].subgridspec(
            2,
            4,
            wspace=0.2,
            hspace=0.36,
        )

        axes = np.empty((2, 4), dtype=object)
        all_axes[application] = axes

        pool_count = application_to_pool_count[application]

        for idx, model_name in enumerate(model_names):
            row = idx // 4
            col = idx % 4

            ax = fig.add_subplot(inner_gs[row, col])
            axes[row, col] = ax

            ax.set_title(
                pretty_model_name(model_name),
                loc="center",
                pad=4,
                fontsize=FONT_SIZE,
            )

            sample_path = (
                f"outputs/{application}/contextual/{attribute_type}/"
                f"{model_name}_{resume_counts[0]}_{pool_count}.jsonl"
            )

            if not os.path.exists(sample_path):
                print(f"[Warning] File not found, skipping: {sample_path}")
                ax.set_visible(False)
                continue

            plot_model_panel(
                ax=ax,
                application=application,
                attribute_type=attribute_type,
                resume_counts=resume_counts,
                model_name=model_name,
                pool_count=pool_count,
                max_n_trials=max_n_trials,
                minority_color=minority_color,
                majority_color=majority_color,
            )

    # ------------------------------------------------------------
    # Shared labels
    # ------------------------------------------------------------
    fig.supxlabel(
        "Number of candidates",
        fontsize=FONT_SIZE,
        y=0.055,
    )

    fig.supylabel(
        "Group-level selection rate (%)",
        fontsize=FONT_SIZE,
        x=0.08,
    )

    # ------------------------------------------------------------
    # Application row titles
    # ------------------------------------------------------------
    for application in applications:
        axes = all_axes[application]

        pos_left = axes[0, 0].get_position()
        pos_right = axes[0, 3].get_position()

        x0 = pos_left.x0
        x1 = pos_right.x1
        y1 = pos_left.y1

        fig.text(
            (x0 + x1) / 2,
            y1 + 0.022,
            application_title_map[application],
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )

    # ------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=minority_color,
            marker="^",
            markerfacecolor=minority_color,
            markeredgecolor=minority_color,
            markeredgewidth=1.0,
            linewidth=1.20,
            markersize=4.0,
            label="Selection rate (societal minority)",
        ),
        Line2D(
            [0],
            [0],
            color=majority_color,
            marker="o",
            markerfacecolor=majority_color,
            markeredgecolor=majority_color,
            markeredgewidth=1.0,
            linewidth=1.20,
            markersize=4.0,
            label="Selection rate (societal majority)",
        ),
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle="--",
            linewidth=1.20,
            label="Uniform-random selection rate (50%)",
        )
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.020),
        ncol=3,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.4,
        handletextpad=0.55,
        fontsize=FONT_SIZE,
    )

    base = f"{safe_slug(attribute_type)}_all_applications_cross_candidate_societal_nature_style"
    pdf_path = os.path.join(output_dir, base + ".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

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

    # If your pool count differs by application, set it here.
    # If all are 500, just keep them all 500.
    application_to_pool_count = {
        "hiring": 200,
        "loan": 500,
        "edu": 500,
    }

    resume_counts = [2, 4, 6, 8, 10]

    for attribute_type in ["Gender Identity", "Sexual Orientation"]:
        draw_attribute_big_figure(
            attribute_type=attribute_type,
            model_names=model_names_order,
            applications=applications,
            resume_counts=resume_counts,
            application_to_pool_count=application_to_pool_count,
            max_n_trials=max_n_trials,
            output_dir="outputs/societal",
        )