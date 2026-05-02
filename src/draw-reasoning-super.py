import json
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy import stats


# ============================================================
# Shared style
# ============================================================

def set_nature_style():
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

        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": 1.2,
    })


# ============================================================
# Contextual minority computation
# ============================================================

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0

    p = k / n
    denominator = 1 + (z ** 2) / n
    centre = p + (z ** 2) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))

    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return lower, upper


def compute_contextual_results(file_name, attribute_type, max_n_trials=1000000):
    """
    Compute absolute selection-rate difference at the minimum contextual ratio.
    Returns:
        {
            "delta": ...,
            "ci_low": ...,
            "ci_high": ...
        }
    """

    attr_value_to_results = defaultdict(lambda: {
        "same_attr_count_to_count": defaultdict(int),
        "same_attr_count_to_hit_count": defaultdict(int),
    })

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

            for inner_idx, attr_value in enumerate(attributes):
                same_attr_count = attributes.count(attr_value) - 1

                attr_value_to_results[attr_value]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results[attr_value]["same_attr_count_to_hit_count"][same_attr_count] += (
                    1 if inner_idx == suggested_candidate_id else 0
                )

    attr_counts_A = None
    attr_counts_B = None

    for attr_value, attr_value_results in attr_value_to_results.items():
        attr_counts = {}

        same_attr_count_to_count = dict(
            sorted(
                attr_value_results["same_attr_count_to_count"].items(),
                key=lambda x: x[0],
            )
        )

        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
            attr_counts[same_attr_count] = (hit_count, count)

        if attr_value in ["Black", "Female"]:
            attr_counts_A = attr_counts
        else:
            attr_counts_B = attr_counts

    if attr_counts_A is None or attr_counts_B is None:
        return None

    results = {}

    for c in sorted(set(attr_counts_A) & set(attr_counts_B)):
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]

        pA = hA / nA
        pB = hB / nB

        ciA_low, ciA_high = wilson_ci(hA, nA)
        ciB_low, ciB_high = wilson_ci(hB, nB)

        if pA > pB:
            delta = pA - pB
            ci_low = ciA_low - ciB_high
            ci_high = ciA_high - ciB_low
        else:
            delta = pB - pA
            ci_low = ciB_low - ciA_high
            ci_high = ciB_high - ciA_low

        results[c] = {
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    if 1 not in results:
        return None

    return results[1]


# ============================================================
# Societal minority computation
# ============================================================

type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}


def bootstrap_relative_diff_ci(
    minority_scores,
    majority_scores,
    n_bootstrap=5000,
    alpha=0.05,
    seed=42,
):
    rng = np.random.default_rng(seed)

    minority_scores = np.asarray(minority_scores, dtype=float)
    majority_scores = np.asarray(majority_scores, dtype=float)

    n_min = len(minority_scores)
    n_maj = len(majority_scores)

    deltas = []

    for _ in range(n_bootstrap):
        min_sample = rng.choice(minority_scores, size=n_min, replace=True)
        maj_sample = rng.choice(majority_scores, size=n_maj, replace=True)

        min_mean = np.mean(min_sample)
        maj_mean = np.mean(maj_sample)

        if maj_mean == 0:
            continue

        delta = (min_mean - maj_mean) / maj_mean
        deltas.append(delta)

    deltas = np.asarray(deltas, dtype=float)

    if len(deltas) == 0:
        return None

    lower = np.percentile(deltas, 100 * alpha / 2)
    upper = np.percentile(deltas, 100 * (1 - alpha / 2))

    return {
        "delta": float(np.mean(deltas)),
        "ci_low": float(lower),
        "ci_high": float(upper),
    }


def compute_societal_results(file_name, attribute_type):
    """
    Compute relative score difference:
        (minority_mean - majority_mean) / majority_mean
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

    if len(minority_scores) == 0 or len(majority_scores) == 0:
        return None

    return bootstrap_relative_diff_ci(minority_scores, majority_scores)


# ============================================================
# Plotting helpers
# ============================================================

def get_file_contextual(application, attribute_type, model_name):
    if "no_thinking" in model_name:
        base_model = model_name[:-12]
        file_name = (
            f"outputs/{application}/contextual/{attribute_type}/"
            f"{base_model}_5_500_no_thinking.jsonl"
        )
        if not os.path.exists(file_name):
            file_name = (
                f"outputs/{application}/contextual/{attribute_type}/"
                f"{base_model}_5_200_no_thinking.jsonl"
            )
    else:
        file_name = (
            f"outputs/{application}/contextual/{attribute_type}/"
            f"{model_name}_5_500.jsonl"
        )
        if not os.path.exists(file_name):
            file_name = (
                f"outputs/{application}/contextual/{attribute_type}/"
                f"{model_name}_5_200.jsonl"
            )

    return file_name


def get_file_societal(application, attribute_type, model_name):
    return f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"


def collect_results(
    applications,
    attribute_types,
    model_names,
    result_type,
):
    """
    result_type: "contextual" or "societal"
    """

    attribute_type_to_application_to_model_to_delta = {}

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(dict))

        for application in applications:
            for model_name in model_names:
                if result_type == "contextual":
                    file_name = get_file_contextual(application, attribute_type, model_name)
                    compute_fn = compute_contextual_results
                elif result_type == "societal":
                    file_name = get_file_societal(application, attribute_type, model_name)
                    compute_fn = compute_societal_results
                else:
                    raise ValueError(f"Unknown result_type: {result_type}")

                if not os.path.exists(file_name):
                    raise FileNotFoundError(
                        f"File not found: {application} {attribute_type} {model_name}\n{file_name}"
                    )

                delta = compute_fn(file_name, attribute_type)

                if delta is None:
                    delta = {
                        "delta": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }

                application_to_model_to_delta[application][model_name] = delta

        attribute_type_to_application_to_model_to_delta[attribute_type] = application_to_model_to_delta

    return attribute_type_to_application_to_model_to_delta


def draw_reasoning_block(
    fig,
    outer_spec,
    attribute_type_to_application_to_model_to_delta,
    attribute_types,
    applications,
    panel_letter,
    block_title,
    ylabel,
    base_to_color,
    mode_to_marker,
):
    """
    Draw one 2 x 3 block:
        rows = attribute types
        columns = applications
    """

    panel_titles = {
        "edu": "Scholarship",
        "hiring": "Hiring",
        "loan": "Loan",
    }

    pair_defs = [
        ("GLM-4.5-Air", "GLM-4.5-Air_no_thinking", "GLM-4.5-Air"),
        ("NVIDIA-Nemotron-Nano-12B-v2", "NVIDIA-Nemotron-Nano-12B-v2_no_thinking", "Nemotron-12B"),
    ]

    model_order = [p[2] for p in pair_defs]
    model_to_x = {m: i for i, m in enumerate(model_order)}

    dodge = 0.16
    mode_to_dx = {
        "reasoning": -dodge,
        "non-reasoning": +dodge,
    }

    inner_gs = outer_spec.subgridspec(
        2,
        3,
        wspace=0.18,
        hspace=0.70,
    )

    axes = np.empty((2, 3), dtype=object)

    for row_idx, attribute_type in enumerate(attribute_types):
        application_to_model_to_delta = attribute_type_to_application_to_model_to_delta[attribute_type]

        for col_idx, application in enumerate(applications):
            ax = fig.add_subplot(inner_gs[row_idx, col_idx])
            axes[row_idx, col_idx] = ax

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            rows = []

            for base_name, no_think_name, short_label in pair_defs:
                d = application_to_model_to_delta[application][base_name]
                rows.append({
                    "model": short_label,
                    "mode": "reasoning",
                    "delta": float(d["delta"]),
                    "ci_low": float(d["ci_low"]),
                    "ci_high": float(d["ci_high"]),
                })

                d = application_to_model_to_delta[application][no_think_name]
                rows.append({
                    "model": short_label,
                    "mode": "non-reasoning",
                    "delta": float(d["delta"]),
                    "ci_low": float(d["ci_low"]),
                    "ci_high": float(d["ci_high"]),
                })

            for r in rows:
                x0 = model_to_x[r["model"]]
                x = x0 + mode_to_dx[r["mode"]]
                y = r["delta"]

                yerr_low = max(0.0, y - r["ci_low"])
                yerr_high = max(0.0, r["ci_high"] - y)
                yerr = np.array([[yerr_low], [yerr_high]])

                ax.errorbar(
                    [x],
                    [y],
                    yerr=yerr,
                    fmt=mode_to_marker[r["mode"]],
                    markersize=9.0,
                    capsize=3.2,
                    elinewidth=1.0,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    color=base_to_color[r["model"]],
                    zorder=3,
                )

            if row_idx == 0:
                ax.set_title(
                    panel_titles[application],
                    fontsize=10.5,
                    pad=6,
                )

            ax.set_xticks([model_to_x[m] for m in model_order])
            ax.set_xticklabels(model_order)

            for tick_label in ax.get_xticklabels():
                model_name = tick_label.get_text()
                tick_label.set_color(base_to_color[model_name])
                tick_label.set_fontweight("bold")

            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v * 100:.0f}%"))

            ax.tick_params(
                axis="both",
                which="major",
                direction="out",
                length=3.2,
                width=0.75,
                color="black",
                labelcolor="black",
                bottom=True,
                left=True,
                top=False,
                right=False,
            )

            ax.grid(
                axis="y",
                color="0.88",
                linewidth=0.6,
                linestyle="-",
                zorder=0,
            )
            ax.set_axisbelow(True)

    # Geometry
    pos_top_left = axes[0, 0].get_position()
    pos_top_right = axes[0, 2].get_position()
    pos_bottom_left = axes[1, 0].get_position()

    block_x0 = pos_top_left.x0
    block_x1 = pos_top_right.x1
    block_y1 = pos_top_left.y1
    block_y0 = pos_bottom_left.y0
    block_x_center = (block_x0 + block_x1) / 2
    block_y_center = (block_y0 + block_y1) / 2

    # Panel letter and block title
    fig.text(
        block_x0 - 0.030,
        block_y1 + 0.055,
        panel_letter,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    fig.text(
        block_x0 + 0.005,
        block_y1 + 0.055,
        block_title,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    # Row subtitles
    row_title_offset = 0.026

    for row_idx, attribute_type in enumerate(attribute_types):
        pos_row_left = axes[row_idx, 0].get_position()
        pos_row_right = axes[row_idx, 2].get_position()

        row_x_center = (pos_row_left.x0 + pos_row_right.x1) / 2
        row_y = pos_row_left.y1 + row_title_offset

        fig.text(
            row_x_center,
            row_y,
            attribute_type,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Block-level labels
    fig.text(
        block_x_center,
        block_y0 - 0.055,
        "Model",
        ha="center",
        va="top",
        fontsize=11,
    )

    fig.text(
        block_x0 - 0.045,
        block_y_center,
        ylabel,
        ha="center",
        va="center",
        rotation=90,
        fontsize=11,
    )

    return axes


def draw_reasoning_super_figure(
    contextual_results,
    societal_results,
    output_dir="outputs/reasoning",
):
    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    applications = ["hiring", "loan", "edu"]

    contextual_attribute_types = ["Gender", "Race"]
    societal_attribute_types = ["Gender Identity", "Sexual Orientation"]

    # Professional, colorblind-safe colors
    base_to_color = {
        "GLM-4.5-Air": "#0072B2",
        "Nemotron-12B": "#D55E00",
    }

    mode_to_marker = {
        "reasoning": "o",
        "non-reasoning": "X",
    }

    fig = plt.figure(figsize=(7.45, 8.4))

    outer_gs = fig.add_gridspec(
        2,
        1,
        left=0.115,
        right=0.995,
        bottom=0.165,
        top=0.925,
        hspace=0.42,
    )

    draw_reasoning_block(
        fig=fig,
        outer_spec=outer_gs[0],
        attribute_type_to_application_to_model_to_delta=contextual_results,
        attribute_types=contextual_attribute_types,
        applications=applications,
        panel_letter="a",
        block_title="Contextual minority bias",
        ylabel="Absolute selection-rate difference (%)",
        base_to_color=base_to_color,
        mode_to_marker=mode_to_marker,
    )

    draw_reasoning_block(
        fig=fig,
        outer_spec=outer_gs[1],
        attribute_type_to_application_to_model_to_delta=societal_results,
        attribute_types=societal_attribute_types,
        applications=applications,
        panel_letter="b",
        block_title="Societal minority bias",
        ylabel="Relative difference in score (%)",
        base_to_color=base_to_color,
        mode_to_marker=mode_to_marker,
    )

    # One shared legend below the whole figure
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=base_to_color["GLM-4.5-Air"],
            markeredgecolor="none",
            markersize=8.0,
            label="GLM-4.5-Air",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=base_to_color["Nemotron-12B"],
            markeredgecolor="none",
            markersize=8.0,
            label="Nemotron-12B",
        ),
        Line2D(
            [0],
            [0],
            marker=mode_to_marker["reasoning"],
            linestyle="",
            color="0.20",
            markerfacecolor="0.20",
            markeredgecolor="none",
            markersize=8.0,
            label="Reasoning",
        ),
        Line2D(
            [0],
            [0],
            marker=mode_to_marker["non-reasoning"],
            linestyle="",
            color="0.20",
            markerfacecolor="0.20",
            markeredgecolor="none",
            markersize=8.0,
            label="Non-reasoning",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        fontsize=10,
        handletextpad=0.45,
        columnspacing=1.35,
    )

    pdf_path = os.path.join(output_dir, "reasoning_contextual_societal_super_figure.pdf")

    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {pdf_path}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    applications = ["edu", "hiring", "loan"]

    model_names = [
        "GLM-4.5-Air",
        "GLM-4.5-Air_no_thinking",
        "NVIDIA-Nemotron-Nano-12B-v2",
        "NVIDIA-Nemotron-Nano-12B-v2_no_thinking",
    ]

    contextual_attribute_types = ["Gender", "Race"]
    societal_attribute_types = ["Gender Identity", "Sexual Orientation"]

    contextual_results = collect_results(
        applications=applications,
        attribute_types=contextual_attribute_types,
        model_names=model_names,
        result_type="contextual",
    )

    societal_results = collect_results(
        applications=applications,
        attribute_types=societal_attribute_types,
        model_names=model_names,
        result_type="societal",
    )

    draw_reasoning_super_figure(
        contextual_results=contextual_results,
        societal_results=societal_results,
        output_dir="outputs/reasoning",
    )