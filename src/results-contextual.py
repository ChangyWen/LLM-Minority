import json
import sys
from collections import defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, norm


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


def compute_results(file_name, attribute_type):

    attr_value_to_results = defaultdict(lambda: {
        "same_attr_count_to_count": defaultdict(int),
        "same_attr_count_to_hit_count": defaultdict(int),
    })
    n_trials = 0

    with open(file_name, "r") as f:
        for line in f:
            n_trials += 1

            item = json.loads(line)
            attributes = item["attributes"]
            suggested_candidate_id = item["suggested_candidate_id"]

            for inner_idx, attr_value in enumerate(attributes):
                same_attr_count = attributes.count(attr_value) - 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_hit_count"][same_attr_count] += (1 if inner_idx == suggested_candidate_id else 0)

                attr_value_to_results[attr_value]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results[attr_value]["same_attr_count_to_hit_count"][same_attr_count] += (1 if inner_idx == suggested_candidate_id else 0)

    print(f"Attribute type: {attribute_type}")
    results = {}
    for attr_value, attr_value_results in attr_value_to_results.items():
        # sort the attr_value_results by same_attr_count
        print(f"attr_value: {attr_value}")
        results[attr_value] = {}
        # sort attr_value_results["same_attr_count_to_count"]
        same_attr_count_to_count = dict(sorted(attr_value_results["same_attr_count_to_count"].items(), key=lambda x: x[0]))
        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
            hit_rate = hit_count / count
            ci_low, ci_high = wilson_ci(hit_count, count)
            print(f"same_attr_count: {same_attr_count}, total: {count}, hit_rate: {hit_rate:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
            results[attr_value][same_attr_count] = {
                "hit_rate": hit_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

    return results, n_trials


def draw_results(model_name, attribute_type, resume_count, all_results, n_trials):
    """
    all_results: dict mapping attribute_type -> results dict (as returned by compute_results)
    """
    # Match your previous style
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    # Modern color palette
    non_all_values = set(all_results.keys()) - {"all_attr_values"}
    attribute_values = sorted(list(non_all_values)) + ["all_attr_values"]
    palette = sns.color_palette("husl", len(attribute_values))

    fig, ax = plt.subplots(dpi=1024)
    baseline_value = 0
    xticks = []

    all_barlines = []
    for i, attribute_value in enumerate(attribute_values):
        res = all_results[attribute_value]

        # ensure x is sorted
        xs = sorted(res.keys())
        xticks = xs
        baseline_value = 1 / (len(xs))
        ys = [res[x]["hit_rate"] for x in xs]

        # asymmetric error bars from CI
        lower_err = [res[x]["hit_rate"] - res[x]["ci_low"] for x in xs]
        upper_err = [res[x]["ci_high"] - res[x]["hit_rate"] for x in xs]
        yerr = [lower_err, upper_err]

        line, caplines, barlines = ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker="o",
            markersize=4,
            linewidth=1.5,
            linestyle="--" if attribute_value != "all_attr_values" else "-",
            label=attribute_value if attribute_value != "all_attr_values" else "All",
            color=palette[i],
            capsize=4,
            capthick=1.5,
        )
        if attribute_value != "all_attr_values":
            all_barlines.extend(barlines)

    # barlines is a list of Line2D objects for the error bars
    for bar in all_barlines:
        bar.set_linestyle("--")     # or "--", ":", "-.", etc.
        bar.set_linewidth(1.2)

    # Baseline at 0.20 (1 out of 5 candidates)
    ax.axhline(
        y=baseline_value,
        color="black",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
    )
    # Optional annotation for baseline
    ax.text(
        0.02, baseline_value,
        f"Random ({baseline_value:.3f})",
        transform=ax.get_yaxis_transform(),  # x in data coords, y in axis coords
        fontsize=9,
        fontweight="bold",
        color="black",
        ha="left",
        va="bottom",
    )

    ax.set_xticks(xticks)
    ax.set_xlim(-0.1, len(xticks) - 1 + 0.1)

    ax.set_xlabel("Number of same-attribute candidates", fontsize=11, fontweight="bold")
    ax.set_ylabel("Selection rate of randomly anchored candidate", fontsize=11, fontweight="bold")
    model_name = model_name.replace("msra-", "")
    ax.set_title(f"{attribute_type} ({model_name})\n# of trials: {n_trials}; Mean w/ 95% CI", pad=15, weight="bold")

    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    # Remove top/right spines for a cleaner look
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(title="Attribute type", fontsize=12, title_fontsize=13, markerscale=1.6)

    plt.tight_layout()
    save_file = f"outputs/contextual_{model_name}_{attribute_type}_{resume_count}.png"
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    pool_count = 200

    for attribute_type in ["Gender"]:
        for resume_count in [5]:
            for model_name in ["msra-gpt-4o", "msra-gpt-4.1-nano", "Qwen3-Next-80B-A3B-Instruct"]:
                file_name = f"outputs/contextual/{attribute_type}/{model_name}_{resume_count}_{pool_count}.jsonl"
                if os.path.exists(file_name):
                    print(f"------------------------------------\n\n{file_name}")
                    results, n_trials = compute_results(file_name, attribute_type)
                    draw_results(model_name, attribute_type, resume_count, results, n_trials)
