import json
import sys
from collections import defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, norm
import statsmodels.api as sm
import numpy as np
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}


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


def compute_results(file_name, attribute_type, max_n_trials=100000):

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
            if attribute_type in type_to_minority_attributes:
                minority_attr_values = type_to_minority_attributes[attribute_type]
                if suggested_candidate_attr_value in minority_attr_values:
                    suggested_candidate_attr_value = "Minority"
                else:
                    suggested_candidate_attr_value = "Majority"
            attr_value_to_hit_count[suggested_candidate_attr_value] += 1

    print(f"Attribute type: {attribute_type}")
    results = {}
    for attr_value, attr_value_hit_count in attr_value_to_hit_count.items():
        hit_rate = attr_value_hit_count / n_trials
        ci_low, ci_high = wilson_ci(attr_value_hit_count, n_trials)
        print(f"attr_value: {attr_value}, hit_rate: {hit_rate:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
        results[attr_value] = {
            "hit_rate": hit_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return results


def draw_results_grid(attribute_type, resume_counts, model_names, pool_count, max_n_trials):
    """
    One figure per model.
    x-axis: resume_counts (# of Candidate)
    y-axis: hit_rate (Selection Rate)
    At each x, plot two points w/ 95% CI: Minority and Majority.
    Minority: different marker + dashed error bars
    Majority: different marker + solid error bars
    """

    # -----------------------------
    # 2) Plot style
    # -----------------------------
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })


    palette = sns.color_palette("pastel", 2)

    minority_marker = "o"
    majority_marker = "s"
    minority_color = palette[0]
    majority_color = palette[1]

    # Small horizontal offset so Minority/Majority don’t overlap at the same xtick
    x_offset = 0.12

    # -----------------------------
    # 3) Draw one figure per model
    # -----------------------------
    for model_name in model_names:
        xs = np.array(resume_counts, dtype=float)

        # Collect y + CI for minority/majority
        y_min, lo_min, hi_min = [], [], []
        y_maj, lo_maj, hi_maj = [], [], []

        for rc in resume_counts:
            file_path = f"outputs/loan/contextual/{attribute_type}/{model_name}_{rc}_{pool_count}.jsonl"
            results = compute_results(file_path, attribute_type, max_n_trials=max_n_trials)

            # Expecting keys "Minority" and "Majority" after mapping
            # If your raw data uses other labels, adjust here.
            if "Minority" not in results or "Majority" not in results:
                raise KeyError(
                    f'Expected both "Minority" and "Majority" in results, got keys={list(results.keys())} '
                    f"for file={file_path}"
                )

            y_min.append(results["Minority"]["hit_rate"])
            lo_min.append(results["Minority"]["ci_low"])
            hi_min.append(results["Minority"]["ci_high"])

            y_maj.append(results["Majority"]["hit_rate"])
            lo_maj.append(results["Majority"]["ci_low"])
            hi_maj.append(results["Majority"]["ci_high"])

        y_min = np.array(y_min)
        lo_min = np.array(lo_min)
        hi_min = np.array(hi_min)

        y_maj = np.array(y_maj)
        lo_maj = np.array(lo_maj)
        hi_maj = np.array(hi_maj)

        # Convert CI bounds to error lengths for matplotlib
        yerr_min = np.vstack([y_min - lo_min, hi_min - y_min])
        yerr_maj = np.vstack([y_maj - lo_maj, hi_maj - y_maj])

        fig, ax = plt.subplots(figsize=(7.6, 4.8))

        # Minority (dashed error bars)
        line_min, cap_min, bar_min = ax.errorbar(
            xs - x_offset,
            y_min,
            yerr=yerr_min,
            fmt=minority_marker,
            linestyle="none",          # only points (no connecting line)
            capsize=4,
            elinewidth=1.6,
            markersize=5,
            color=minority_color,
            label="Minority",
            zorder=3,
        )
        # Make errorbar lines dashed (LineCollection)
        for bc in bar_min:
            bc.set_linestyle("--")

        # Majority (solid error bars)
        line_maj, cap_maj, bar_maj = ax.errorbar(
            xs + x_offset,
            y_maj,
            yerr=yerr_maj,
            fmt=majority_marker,
            linestyle="none",
            capsize=4,
            elinewidth=1.6,
            markersize=5,
            color=majority_color,
            label="Majority",
            zorder=3,
        )
        # bar_maj stays solid by default

        # Axes / labels
        ax.set_title(f"{model_name.replace("msra-", "")} — {attribute_type}")
        ax.set_xlabel("# of Candidates")
        ax.set_ylabel("Selection Rate")

        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in resume_counts])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
        ax.legend(frameon=True, fancybox=True)

        # Optional: y-limits (comment out if you don’t want it)
        # ax.set_ylim(0.0, 1.0)

        plt.tight_layout()

        save_file = f"outputs/loan/mixed_{model_name}_{attribute_type}.png"
        plt.savefig(save_file, dpi=300)
        plt.close(fig)

        print(f"[Saved] {save_file}")


if __name__ == "__main__":
    pool_count = 500
    max_n_trials = 1000000

    model_names_order = [
        "msra-gpt-4o",
        "gemma-3-27b-it",
        "Qwen3-Next-80B-A3B-Instruct",
        "gpt-oss-120b",
        "NVIDIA-Nemotron-Nano-12B-v2",
        "Qwen3-235B-A22B-Instruct-2507",
        "Llama-3.3-70B-Instruct",
        "GLM-4.5-Air",
    ]

    for attribute_type in ["Gender Identity", "Sexual Orientation"]:
        draw_results_grid(
            attribute_type=attribute_type,
            resume_counts=[2, 4, 6, 8, 10],
            model_names=model_names_order,
            pool_count=pool_count,
            max_n_trials=max_n_trials,
        )
