import json
import sys
from collections import defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os


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


def compute_results(file_name, attribute_type, resume_count):
    total_count = 0

    # check the total count of the file
    with open(file_name, "r") as f:
        total_count = sum(1 for _ in f)
    part_size = total_count // 6

    attr_value_to_results = defaultdict(lambda: {
        "same_attr_count_to_count": defaultdict(int),
        "same_attr_count_to_hit_count": defaultdict(int),
    })

    for anchor_index in range(6):
        start_index = anchor_index * part_size
        end_index = start_index + part_size - 1
        # print(f"part {anchor_index}: {end_index - start_index + 1}; start_index: {start_index}, end_index: {end_index}")

        cur_index = 0
        with open(file_name, "r") as f:
            for line in f:
                if cur_index < start_index or cur_index > end_index:
                    cur_index += 1
                    continue
                cur_index += 1
                item = json.loads(line)
                attributes = item["attributes"]
                hit_candidate_id = item["hit_candidate_id"]
                candidate_order = item["candidate_order"]

                if anchor_index not in candidate_order:
                    continue

                anchor_index_attr_value = attributes[candidate_order.index(anchor_index)]

                # record the results for the anchor attribute
                same_attr_count = attributes.count(anchor_index_attr_value) - 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_hit_count"][same_attr_count] += (1 if anchor_index == hit_candidate_id else 0)

                attr_value_to_results[anchor_index_attr_value]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results[anchor_index_attr_value]["same_attr_count_to_hit_count"][same_attr_count] += (1 if anchor_index == hit_candidate_id else 0)

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

    return results


def draw_results(model_name, attribute_type, resume_count, all_results):
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
    ax.set_title(f"{model_name} ({attribute_type}) – Selection Rate vs. Same-attribute Count\n(Mean w/ 95% CI)", pad=15, weight="bold")

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

    for attribute_type in ["Gender", "Race"]:
        for resume_count in [5]:
            for model_name in ["msra-gpt-4o", "msra-gpt-5", "msra-gpt-4.1-nano"]:
                file_name = f"outputs/contextual/{attribute_type}/{model_name}_{resume_count}.jsonl"
                if os.path.exists(file_name):
                    print(f"------------------------------------\n\n{file_name}")
                    results = compute_results(file_name, attribute_type, resume_count)
                    draw_results(model_name, attribute_type, resume_count, results)
