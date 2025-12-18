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


def compute_results(file_name, context_size, max_n_trials=1000000):

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
    random_selection_rate = 1 / context_size
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
            "delta": delta / random_selection_rate,
            "ci_low": ci_low / random_selection_rate,
            "ci_high": ci_high / random_selection_rate,
        }

    return results["delta"][1]


def draw_results_by_application(application_to_model_to_delta, attribute_type, model_names, context_sizes):
    """
    One figure per application.
    8 subplots (2x4) in the order of `model_names`.
    Each subplot: x = context_sizes (5, 10), y = delta (with 95% CI).
    """

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
    })
    sns.set_theme(style="whitegrid")

    applications = sorted(application_to_model_to_delta.keys())
    context_sizes = list(sorted(context_sizes))

    # one color per model (consistent across subplots)
    palette = sns.color_palette("Set2", n_colors=len(model_names))
    model_to_color = {m: palette[i] for i, m in enumerate(model_names)}

    for application in applications:
        fig, axes = plt.subplots(
            2, 4,
            figsize=(16.5, 7.2),
            sharex=True,
            sharey=True
        )
        axes = axes.flatten()

        for i, model_name in enumerate(model_names):
            ax = axes[i]

            # Remove upper and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Collect points
            xs, ys, yerr_lo, yerr_hi = [], [], [], []
            for cs in context_sizes:
                d = application_to_model_to_delta[application][model_name][cs]
                y = float(d["delta"])
                lo = float(d["ci_low"])
                hi = float(d["ci_high"])

                xs.append(cs)
                ys.append(y)
                yerr_lo.append(y - lo)
                yerr_hi.append(hi - y)

            yerr = np.vstack([yerr_lo, yerr_hi])  # (2, N) asymmetric

            ax.errorbar(
                xs, ys,
                yerr=yerr,
                fmt="o-",
                markersize=7.5,
                capsize=4.0,
                elinewidth=1.6,
                linewidth=1.2,
                markeredgecolor="black",
                markeredgewidth=0.7,
                color=model_to_color[model_name],
                zorder=3,
            )

            ax.set_title(model_name, fontweight="bold", pad=8)

            ax.set_xticks(context_sizes)
            ax.set_xticklabels([str(x) for x in context_sizes])

            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)

            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)

            # Label only outer axes for cleanliness
            if i % 4 == 0:
                ax.set_ylabel("Norm. Abs. Diff. (Δ / random-rate)")
            if i // 4 == 1:
                ax.set_xlabel("Context size")

        # If fewer than 8 models, hide unused axes (safe)
        for j in range(len(model_names), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"{application.capitalize()} - {attribute_type}",
            fontsize=16,
            fontweight="bold",
            y=0.98
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = f"outputs/size/{application}_{attribute_type}.png"
        plt.savefig(out_path, dpi=512, bbox_inches="tight")
        print(f"Saved subplot grid to {out_path}")
        plt.close(fig)


if __name__ == "__main__":
    applications = ["loan"]

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

    context_sizes = [5, 10]

    attribute_types = ["Gender"]

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(dict))
        for application in applications:
            for model_name in model_names:
                for context_size in context_sizes:
                    if "no_thinking" in model_name:
                        file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name[:-12]}_{context_size}_500_no_thinking.jsonl"
                        if not os.path.exists(file_name):
                            file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name[:-12]}_{context_size}_200_no_thinking.jsonl"
                    else:
                        file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_{context_size}_500.jsonl"
                        if not os.path.exists(file_name):
                            file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_{context_size}_200.jsonl"
                    if not os.path.exists(file_name):
                        raise FileNotFoundError(f"File not found: {application} {attribute_type} {model_name}")

                    delta = compute_results(file_name, context_size)
                    application_to_model_to_delta[application][model_name][context_size] = delta

        draw_results_by_application(
            application_to_model_to_delta=application_to_model_to_delta,
            attribute_type=attribute_type,
            model_names=model_names,
            context_sizes=context_sizes,
        )