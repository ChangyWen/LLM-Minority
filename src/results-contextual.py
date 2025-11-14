import json
import sys
from collections import defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt


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


def compute_results(attribute_type):
    # {"genders": ["Female", "Male", "Male", "Female", "Female"], "candidate_order": [1, 2, 3, 4, 0], "suggested_candidate_id": 2, "hit_candidate_id": 3}

    anchor_index_to_results = {}

    total_count = 0

    # check the total count of the file
    with open(f"outputs/contextual/{attribute_type}/consultant_samples.jsonl", "r") as f:
        total_count = sum(1 for _ in f)
    # print(f"total_count: {total_count}")
    # equally divide the total count into 5 parts
    part_size = total_count // 5
    # print(f"part_size: {part_size}")

    for anchor_index in range(5):
        start_index = anchor_index * part_size
        end_index = start_index + part_size - 1
        # print(f"part {anchor_index}: {end_index - start_index + 1}; start_index: {start_index}, end_index: {end_index}")

        if anchor_index not in anchor_index_to_results:
            anchor_index_to_results[anchor_index] = {
                "same_gender_count_to_count": defaultdict(int),
                "same_gender_count_to_hit_count": defaultdict(int),
            }

        cur_index = 0
        with open(f"outputs/contextual/{attribute_type}/consultant_samples.jsonl", "r") as f:
            for line in f:
                if cur_index < start_index or cur_index > end_index:
                    cur_index += 1
                    continue
                cur_index += 1
                item = json.loads(line)
                if "genders" in item:
                    attributes = item["genders"]
                else:
                    attributes = item["attributes"]

                hit_candidate_id = item["hit_candidate_id"]

                same_gender_count = attributes.count(attributes[anchor_index]) - 1
                anchor_index_to_results[anchor_index]["same_gender_count_to_count"][same_gender_count] += 1
                anchor_index_to_results[anchor_index]["same_gender_count_to_hit_count"][same_gender_count] += (1 if anchor_index == hit_candidate_id else 0)

    same_gender_count_to_count = defaultdict(int)
    same_gender_count_to_hit_count = defaultdict(int)

    for anchor_index, results in anchor_index_to_results.items():
        for same_gender_count, count in results["same_gender_count_to_count"].items():
            same_gender_count_to_count[same_gender_count] += count
        for same_gender_count, hit_count in results["same_gender_count_to_hit_count"].items():
            same_gender_count_to_hit_count[same_gender_count] += hit_count

    same_gender_count_to_count = dict(sorted(same_gender_count_to_count.items(), key=lambda x: x[0]))

    print(f"Attribute type: {attribute_type}")
    results = {}
    for same_gender_count, count in same_gender_count_to_count.items():
        hit_count = same_gender_count_to_hit_count[same_gender_count]
        hit_rate = hit_count / count
        ci_low, ci_high = wilson_ci(hit_count, count)
        print(f"same_gender_count: {same_gender_count}, count: {count}, hit_rate: {hit_rate:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
        results[same_gender_count] = {
            "count": count,
            "hit_rate": hit_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return results


def draw_results(all_results):
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
    attribute_types = ["Race", "Gender", "Religious Affiliation"]
    palette = sns.color_palette("Set2", len(attribute_types))

    fig, ax = plt.subplots(dpi=1024)

    attribute_types = ["Race", "Gender", "Religious Affiliation"]
    markers = ["o", "*", "d"]

    for i, attribute_type in enumerate(attribute_types):
        res = all_results[attribute_type]

        # ensure x is sorted
        xs = sorted(res.keys())
        ys = [res[x]["hit_rate"] for x in xs]

        # asymmetric error bars from CI
        lower_err = [res[x]["hit_rate"] - res[x]["ci_low"] for x in xs]
        upper_err = [res[x]["ci_high"] - res[x]["hit_rate"] for x in xs]
        yerr = [lower_err, upper_err]

        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=markers[i],
            markersize=5,
            linewidth=1.5,
            linestyle="-",
            label=attribute_type,
            color=palette[i],
            capsize=8,
        )

    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlim(-0.1, 4.1)

    ax.set_xlabel("Number of same-attribute candidates", fontsize=11, fontweight="bold")
    ax.set_ylabel("Selection rate of randomly anchored candidate", fontsize=11, fontweight="bold")
    ax.set_title("Contextual Minority – Selection Rate vs. Same-attribute Count\n(Mean w/ 95% CI)", pad=15, weight="bold")

    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    # Remove top/right spines for a cleaner look
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(title="Attribute type", fontsize=9, title_fontsize=10)

    plt.tight_layout()
    save_file = "outputs/contextual.png"
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    all_results = {}
    for attribute_type in ["Race", "Gender", "Religious Affiliation"]:
        results = compute_results(attribute_type)
        all_results[attribute_type] = results

    draw_results(all_results)