import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}

FONT_SIZE = 9.5
LABEL_SIZE = 8.0
MARKER_SIZE = 4.0


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
    Compute overall selection rates for Minority and Majority
    for one model under one application, one attribute type,
    and one candidate-pool size.
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

    results = {}

    for attr_value, hit_count in attr_value_to_hit_count.items():
        hit_rate = hit_count / n_trials
        ci_low, ci_high = wilson_ci(hit_count, n_trials)

        results[attr_value] = {
            "hit_rate": hit_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
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
    Style aligned with your previous contextual figures.
    """

    minority_marker = "o"
    majority_marker = "s"

    xs = np.array(resume_counts, dtype=float)

    y_min, lo_min, hi_min = [], [], []
    y_maj, lo_maj, hi_maj = [], [], []

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

    y_min = np.asarray(y_min, dtype=float)
    lo_min = np.asarray(lo_min, dtype=float)
    hi_min = np.asarray(hi_min, dtype=float)

    y_maj = np.asarray(y_maj, dtype=float)
    lo_maj = np.asarray(lo_maj, dtype=float)
    hi_maj = np.asarray(hi_maj, dtype=float)

    yerr_min = np.vstack([y_min - lo_min, hi_min - y_min])
    yerr_maj = np.vstack([y_maj - lo_maj, hi_maj - y_maj])

    # Minority
    ax.errorbar(
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
    ax.errorbar(
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

    # Axes formatting
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in resume_counts])
    ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)

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

    ax.grid(
        axis="y",
        color="0.88",
        linewidth=0.6,
        linestyle="-",
        zorder=0,
    )
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
        "edu": "Scholarship application",
    }

    # Use the same professional color style as your previous figures
    minority_color = "#D55E00"   # vermillion
    majority_color = "#0072B2"   # blue

    fig = plt.figure(figsize=(9.5, 10))

    outer_gs = fig.add_gridspec(
        3,
        1,
        left=0.125,
        right=0.875,
        bottom=0.090,
        top=0.940,
        hspace=0.30,
    )

    all_axes = {}

    for app_idx, application in enumerate(applications):
        inner_gs = outer_gs[app_idx].subgridspec(
            2,
            4,
            wspace=0.32,
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
        y=0.047,
    )

    fig.supylabel(
        "Selection rate (%)",
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
            marker="o",
            markerfacecolor=minority_color,
            markeredgecolor=minority_color,
            markeredgewidth=1.0,
            linewidth=1.20,
            markersize=4.0,
            label="Minority",
        ),
        Line2D(
            [0],
            [0],
            color=majority_color,
            marker="s",
            markerfacecolor=majority_color,
            markeredgecolor=majority_color,
            markeredgewidth=1.0,
            linewidth=1.20,
            markersize=4.0,
            label="Majority",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.010),
        ncol=2,
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