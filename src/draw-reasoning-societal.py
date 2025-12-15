import json
import sys
from collections import defaultdict
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter, ScalarFormatter
from scipy.stats import pearsonr
from scipy.stats import t
from scipy import stats


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
    "Disability Status": ["Colorblindness", "Hearing Impairment", "Mobility Impairment"],
    "Chronic Health Condition Status": ["HIV Positive", "Chronic Hepatitis", "Type 1 Diabetes", "Asthma"],
    "Religious Affiliation": ["Jewish", "Jain", "Taoist"],
    "Political Affiliation": ["Green Party", "Libertarian"],
    "Race": ["Black"],
}


def t_ci(scores, confidence=0.95):
    mean = np.mean(scores)
    sem = stats.sem(scores)
    df = len(scores) - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    lower = mean - t_crit * sem
    upper = mean + t_crit * sem
    return (mean, lower, upper)


def bootstrap_relative_diff_ci(minority_scores, majority_scores,
                               n_bootstrap=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)

    deltas = []

    n_min = len(minority_scores)
    n_maj = len(majority_scores)

    for _ in range(n_bootstrap):
        min_sample = rng.choice(minority_scores, size=n_min, replace=True)
        maj_sample = rng.choice(majority_scores, size=n_maj, replace=True)

        min_mean = np.mean(min_sample)
        maj_mean = np.mean(maj_sample)

        # avoid pathological division
        if maj_mean == 0:
            continue

        delta = (min_mean - maj_mean) / maj_mean
        deltas.append(delta)

    deltas = np.array(deltas)

    lower = np.percentile(deltas, 100 * alpha / 2)
    upper = np.percentile(deltas, 100 * (1 - alpha / 2))

    return np.mean(deltas), lower, upper


def compute_results(file_name, attribute_type):
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

    minority_scores = np.array(minority_scores)
    majority_scores = np.array(majority_scores)

    # Guard against empty groups
    if len(minority_scores) == 0 or len(majority_scores) == 0:
        return None

    delta_mean, delta_ci_low, delta_ci_high = bootstrap_relative_diff_ci(
        minority_scores, majority_scores
    )

    return {
        "delta": delta_mean,
        "ci_low": delta_ci_low,
        "ci_high": delta_ci_high,
    }


def draw_results_by_application(application_to_model_to_delta, attribute_type):
    """
    For each application, draw a point plot with 2 model-pairs:
      - GLM-4.5-Air vs GLM-4.5-Air_no_thinking
      - NVIDIA-Nemotron-Nano-12B-v2 vs NVIDIA-Nemotron-Nano-12B-v2_no_thinking

    Each point: delta with vertical 95% CI error bar.
    One figure per application (per attribute_type).
    """

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
    })
    sns.set_theme(style="whitegrid")

    applications = sorted(application_to_model_to_delta.keys())

    # Define the exact ordering and labeling you want on x-axis
    pair_defs = [
        ("GLM-4.5-Air", "GLM-4.5-Air_no_thinking", "GLM-4.5-Air"),
        ("NVIDIA-Nemotron-Nano-12B-v2", "NVIDIA-Nemotron-Nano-12B-v2_no_thinking", "Nemotron-12B"),
    ]
    mode_order = ["thinking", "no thinking"]

    # Colors: one color per base model, shared by its two modes (thinking vs no thinking)
    base_palette = sns.color_palette("Set2", n_colors=len(pair_defs))
    base_to_color = {pair_defs[i][2]: base_palette[i] for i in range(len(pair_defs))}
    mode_to_marker = {"thinking": "o", "no thinking": "X"}

    for application in applications:
        fig, ax = plt.subplots(figsize=(7.5, 5.6))

        # Remove upper and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Build rows to plot
        rows = []
        for base_name, no_think_name, short_label in pair_defs:
            # thinking
            d = application_to_model_to_delta[application][base_name]
            rows.append({
                "model": short_label,
                "mode": "thinking",
                "delta": float(d["delta"]),
                "ci_low": float(d["ci_low"]),
                "ci_high": float(d["ci_high"]),
            })
            # no thinking
            d = application_to_model_to_delta[application][no_think_name]
            rows.append({
                "model": short_label,
                "mode": "no thinking",
                "delta": float(d["delta"]),
                "ci_low": float(d["ci_low"]),
                "ci_high": float(d["ci_high"]),
            })

        # X positions (two models), with small horizontal dodge for the 2 modes
        model_order = [p[2] for p in pair_defs]
        model_to_x = {m: i for i, m in enumerate(model_order)}
        dodge = 0.16
        mode_to_dx = {"thinking": -dodge, "no thinking": +dodge}

        # Plot points + error bars
        for r in rows:
            x0 = model_to_x[r["model"]]
            x = x0 + mode_to_dx[r["mode"]]
            y = r["delta"]
            yerr = np.array([[y - r["ci_low"]], [r["ci_high"] - y]])  # asymmetric error

            ax.errorbar(
                [x],
                [y],
                yerr=yerr,
                fmt=mode_to_marker[r["mode"]],
                markersize=18,
                capsize=4.5,
                elinewidth=1.6,
                markeredgecolor="black",
                markeredgewidth=0.7,
                color=base_to_color[r["model"]],
                zorder=3,
            )

        # Legend (custom handles)
        from matplotlib.lines import Line2D
        handles = []
        # Mode legend
        handles.append(Line2D([0], [0], marker=mode_to_marker["thinking"], linestyle="",
                              markeredgecolor="black", markeredgewidth=0.7,
                              markersize=8.5, color="black", label="Reasoning"))
        handles.append(Line2D([0], [0], marker=mode_to_marker["no thinking"], linestyle="",
                              markeredgecolor="black", markeredgewidth=0.7,
                              markersize=8.5, color="black", label="Non-reasoning"))
        ax.legend(handles=handles, loc="best", frameon=True)

        # Axes labels/format
        ax.set_xlabel("Model")
        ax.set_ylabel("Relative Diff. of Scores [(minority - majority) / majority]")

        ax.set_xticks([model_to_x[m] for m in model_order])
        ax.set_xticklabels(model_order)

        # Color xtick labels to match model color
        for tick_label in ax.get_xticklabels():
            model_name = tick_label.get_text()
            tick_label.set_color(base_to_color[model_name])
            tick_label.set_fontweight("bold")

        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))

        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

        ax.set_title(
            f"{application.capitalize()} - {attribute_type}",
            fontsize=16,
            fontweight="bold",
            pad=12,
        )

        plt.tight_layout()
        out_path = f"outputs/reasoning/societal_{application}_{attribute_type}.png"
        plt.savefig(out_path, dpi=512, bbox_inches="tight")
        print(f"Saved point plot to {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    applications = ["edu", "hiring", "loan"]

    model_names = [
        "GLM-4.5-Air",
        "GLM-4.5-Air_no_thinking",
        "NVIDIA-Nemotron-Nano-12B-v2",
        "NVIDIA-Nemotron-Nano-12B-v2_no_thinking",
    ]

    attribute_types = ["Gender Identity", "Sexual Orientation"]

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(dict))
        for application in applications:
            for model_name in model_names:
                file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"
                if not os.path.exists(file_name):
                    file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"
                if not os.path.exists(file_name):
                    assert False, f"File not found: {application} {attribute_type} {model_name}"
                delta = compute_results(file_name, attribute_type)
                application_to_model_to_delta[application][model_name] = delta

        draw_results_by_application(
            application_to_model_to_delta=application_to_model_to_delta,
            attribute_type=attribute_type,
        )