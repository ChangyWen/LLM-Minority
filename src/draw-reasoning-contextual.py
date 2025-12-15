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


# -----------------------------
# Wilson 95% CI for proportions
# -----------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)

    p = k / n
    denominator = 1 + (z**2) / n
    centre = p + (z**2) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return (lower, upper)


def compute_results(file_name, attribute_type, max_n_trials=1000000):

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

    results = {}
    attr_counts_A = None
    attr_counts_B = None
    for attr_value, attr_value_results in attr_value_to_results.items():
        results[attr_value] = {}

        attr_counts = {}
        same_attr_count_to_count = dict(
            sorted(attr_value_results["same_attr_count_to_count"].items(), key=lambda x: x[0])
        )
        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
            attr_counts[same_attr_count] = (hit_count, count)

        if attr_value == "Black" or attr_value == "Female":
            attr_counts_A = attr_counts
        else:
            attr_counts_B = attr_counts

    results["delta"] = {}
    for c in sorted(set(attr_counts_A) & set(attr_counts_B)):
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]
        pA = hA / nA
        pB = hB / nB
        if pA > pB:
            delta = pA - pB
            ciA_low, ciA_high = wilson_ci(hA, nA)
            ciB_low, ciB_high = wilson_ci(hB, nB)
            ci_low = ciA_low - ciB_high
            ci_high = ciA_high - ciB_low
        else:
            delta = pB - pA
            ciA_low, ciA_high = wilson_ci(hA, nA)
            ciB_low, ciB_high = wilson_ci(hB, nB)
            ci_low = ciB_low - ciA_high
            ci_high = ciB_high - ciA_low
        results["delta"][c] = {
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return results["delta"][1]


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
                markersize=8.5,
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
                              markersize=8.5, color="black", label="Thinking"))
        handles.append(Line2D([0], [0], marker=mode_to_marker["no thinking"], linestyle="",
                              markeredgecolor="black", markeredgewidth=0.7,
                              markersize=8.5, color="black", label="No thinking"))
        ax.legend(handles=handles, loc="best", frameon=True)

        # Axes labels/format
        ax.set_xlabel("")
        ax.set_ylabel("Abs. Δ (w.r.t. min. contextual ratio)")

        ax.set_xticks([model_to_x[m] for m in model_order])
        ax.set_xticklabels(model_order)

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
        out_path = f"outputs/reasoning/contextual_{application}_{attribute_type}.png"
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

    attribute_types = ["Gender"]

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(float))
        for application in applications:
            for model_name in model_names:
                if "no_thinking" in model_name:
                    file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name[:-12]}_5_500_no_thinking.jsonl"
                    if not os.path.exists(file_name):
                        file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name[:-12]}_5_200_no_thinking.jsonl"
                else:
                    file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_5_500.jsonl"
                    if not os.path.exists(file_name):
                        file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_5_200.jsonl"
                if not os.path.exists(file_name):
                    raise FileNotFoundError(f"File not found: {application} {attribute_type} {model_name}")

                delta = compute_results(file_name, attribute_type)
                application_to_model_to_delta[application][model_name] = delta

        draw_results_by_application(
            application_to_model_to_delta=application_to_model_to_delta,
            attribute_type=attribute_type,
        )