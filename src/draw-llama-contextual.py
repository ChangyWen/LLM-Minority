import json
import sys
from collections import defaultdict
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D

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

def se_diff_of_props(hA, nA, hB, nB):
    pA = hA / nA
    pB = hB / nB
    return math.sqrt(pA*(1-pA)/nA + pB*(1-pB)/nB)

def abs_diff(h1, n1, h2, n2):
    return abs((h1 / n1) - (h2 / n2))

def compute_results(file_name, context_size, max_n_trials=1000000):
    """
    Returns: dict like { "20%": {"delta":..., "ci_low":..., "ci_high":...}, ...}
    delta is normalized by random rate (1/context_size), consistent with your original code.
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

    attr_counts_A, attr_counts_B = None, None
    for attr_value, attr_value_results in attr_value_to_results.items():
        attr_counts = {}
        same_attr_count_to_count = dict(
            sorted(attr_value_results["same_attr_count_to_count"].items(), key=lambda x: x[0])
        )
        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
            attr_counts[same_attr_count] = (hit_count, count)

        # Keep unchanged from your code
        if attr_value == "Black" or attr_value == "Female":
            attr_counts_A = attr_counts
        else:
            attr_counts_B = attr_counts

    results_delta = {}
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

        ratio = (c + 1) / context_size
        ratio_str = f"{ratio * 100:.0f}%"
        if ratio_str in ["20%", "40%", "60%", "80%"]:
            results_delta[ratio_str] = {
                "random_selection_rate": random_selection_rate,
                "delta": delta / random_selection_rate,
                "ci_low": ci_low / random_selection_rate,
                "ci_high": ci_high / random_selection_rate,
            }

    return results_delta

def draw_1x2_per_application(
    app_attr_model_to_delta,
    applications,
    attribute_types,
    model_names,
):
    """
    One figure per application (1x2).
    Each subplot = one attribute type.
    Each subplot contains 2 lines = 2 models.
    Context size is fixed (not shown in the figure).
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

    ratio_strs = ["20%", "40%", "60%", "80%"]
    ratio_x = np.array([20, 40, 60, 80], dtype=float)

    # Model colors
    palette = sns.color_palette("tab10", n_colors=max(2, len(model_names)))
    model_to_color = {m: palette[i] for i, m in enumerate(model_names)}

    # Shared legend
    legend_handles = [
        Line2D(
            [0], [0],
            color=model_to_color[m],
            marker="o",
            linestyle="-",
            linewidth=1.6,
            markersize=6.5,
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        for m in model_names
    ]
    legend_labels = model_names[:]

    for application in applications:
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0), sharex=False, sharey=False)

        for ax, attribute_type in zip(axes, attribute_types):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            for model_name in model_names:
                delta_by_ratio = (
                    app_attr_model_to_delta
                    .get(application, {})
                    .get(attribute_type, {})
                    .get(model_name, None)
                )
                if not delta_by_ratio:
                    continue

                xs, ys, yerr_lo, yerr_hi = [], [], [], []
                for rx, rstr in zip(ratio_x, ratio_strs):
                    if rstr not in delta_by_ratio:
                        continue
                    d = delta_by_ratio[rstr]
                    y = float(d["delta"])
                    lo = float(d["ci_low"])
                    hi = float(d["ci_high"])
                    xs.append(rx)
                    ys.append(y)
                    yerr_lo.append(y - lo)
                    yerr_hi.append(hi - y)

                if len(xs) == 0:
                    continue

                yerr = np.vstack([yerr_lo, yerr_hi])

                ax.errorbar(
                    xs, ys,
                    yerr=yerr,
                    fmt="-o",
                    markersize=6.5,
                    capsize=4.0,
                    elinewidth=1.5,
                    linewidth=1.6,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    color=model_to_color[model_name],
                    zorder=3,
                )

            ax.set_title(attribute_type, fontweight="bold", pad=8, fontsize=16)

            # Keep x-axis unchanged
            ax.set_xticks(ratio_x)
            ax.set_xticklabels(ratio_strs)
            ax.set_xlim(10, 90)

            # Keep y-axis unchanged
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))

            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)

        fig.suptitle(f"{application.capitalize()}", fontsize=16, fontweight="bold", y=0.98)

        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=len(model_names),
            frameon=False,
            fontsize=12,
            handlelength=2.4,
            columnspacing=1.6,
            handletextpad=0.6,
        )

        fig.supylabel(
            "Norm. Abs. Diff. of Selection Rate\n(Δ / random-rate)",
            fontweight="bold",
            fontsize=16,
            x=0.03,
            ha="center"
        )
        fig.supxlabel("Contextual Ratio", fontweight="bold", fontsize=16, y=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.90])

        os.makedirs("outputs/llama", exist_ok=True)
        out_path = f"outputs/llama/{application}.png"
        plt.savefig(out_path, dpi=512, bbox_inches="tight")
        print(f"Saved 1x2 figure to {out_path}")
        plt.close(fig)

if __name__ == "__main__":
    applications = ["loan", "hiring", "edu"]
    model_names = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct"]
    attribute_types = ["Gender", "Race"]

    # Fixed default context size
    context_size = 5

    # application -> attribute_type -> model -> delta_by_ratio
    app_attr_model_to_delta = defaultdict(lambda: defaultdict(dict))

    for application in applications:
        for attribute_type in attribute_types:
            for model_name in model_names:
                file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_{context_size}_500.jsonl"
                if not os.path.exists(file_name):
                    file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_{context_size}_200.jsonl"
                if not os.path.exists(file_name):
                    continue

                delta = compute_results(file_name, context_size=context_size)
                app_attr_model_to_delta[application][attribute_type][model_name] = delta

    draw_1x2_per_application(
        app_attr_model_to_delta=app_attr_model_to_delta,
        applications=applications,
        attribute_types=attribute_types,
        model_names=model_names,
    )
