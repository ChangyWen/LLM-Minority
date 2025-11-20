import json
import sys
from collections import defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from scipy import stats


def prob_ci_from_estimates(p_hats, alpha=0.05):
    """
    p_hats: list/array of independent probability estimates in [0, 1]
    alpha: 1 - confidence level (0.05 for 95% CI)

    Returns:
        mean_estimate, lower_ci, upper_ci
    """
    p_hats = np.asarray(p_hats, dtype=float)
    m = len(p_hats)
    if m < 2:
        raise ValueError("Need at least 2 estimates to compute a CI")

    mean = p_hats.mean()
    sd = p_hats.std(ddof=1)
    se = sd / np.sqrt(m)

    # t critical value for two-sided (1 - alpha) CI
    tcrit = stats.t.ppf(1 - alpha / 2, df=m - 1)
    lower = mean - tcrit * se
    upper = mean + tcrit * se

    # Clamp to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return mean, lower, upper


def get_anchor_idx_to_resume_idx(pool_count):
    dataset_dir = "dataset"
    anchor_idx_to_resume_idx = defaultdict(set)
    all_job_files = [file for file in os.listdir(dataset_dir) if file.startswith("job_")]
    for job_file in all_job_files:
        anchor_idx = 0
        with open(os.path.join(dataset_dir, job_file), "r") as f:
            for anchor_idx, line in enumerate(f):
                if anchor_idx >= pool_count:
                    break
                item = json.loads(line)
                resume_idx = item["idx"]
                anchor_idx_to_resume_idx[anchor_idx].add(resume_idx)
    return anchor_idx_to_resume_idx


def compute_results(file_name, attribute_type, pool_count, anchor_idx_to_resume_idx, max_line=float('inf')):
    total_count = 0
    # check the total count of the file
    with open(file_name, "r") as f:
        total_count = sum(1 for _ in f)
    total_count = min(total_count, max_line)
    part_size = total_count // pool_count

    print(f"Attribute type: {attribute_type}")
    results = {}

    repeat_count = 0
    while repeat_count < 20:
        all_line_idxs = [i for i in range(part_size * pool_count)]
        random.shuffle(all_line_idxs)
        line_idx_to_anchor_idx = {}
        for abs_idx, line_idx in enumerate(all_line_idxs):
            anchor_idx = abs_idx // part_size
            line_idx_to_anchor_idx[line_idx] = anchor_idx

        attr_value_to_results = defaultdict(lambda: {
            "same_attr_count_to_count": defaultdict(int),
            "same_attr_count_to_hit_count": defaultdict(int),
        })

        with open(file_name, "r") as f:
            for line_idx, line in enumerate(f):
                if line_idx >= part_size * pool_count:
                    break
                anchor_index = line_idx_to_anchor_idx[line_idx]
                item = json.loads(line)
                attributes = item["attributes"]
                hit_candidate_id = item["hit_candidate_id"]
                candidate_order = item["candidate_order"]

                matched_candidate_idx = set(candidate_order) & anchor_idx_to_resume_idx[anchor_index]
                if len(matched_candidate_idx) <= 0:
                    print(f"No matched candidate for anchor {anchor_index}")
                    continue
                assert len(matched_candidate_idx) == 1
                matched_candidate_idx = list(matched_candidate_idx)[0]

                anchor_index_attr_value = attributes[candidate_order.index(matched_candidate_idx)]

                # record the results for the anchor attribute
                same_attr_count = attributes.count(anchor_index_attr_value) - 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_hit_count"][same_attr_count] += (1 if matched_candidate_idx == hit_candidate_id else 0)

                attr_value_to_results[anchor_index_attr_value]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results[anchor_index_attr_value]["same_attr_count_to_hit_count"][same_attr_count] += (1 if matched_candidate_idx == hit_candidate_id else 0)

        for attr_value, attr_value_results in attr_value_to_results.items():
            if attr_value not in results:
                results[attr_value] = {}
            same_attr_count_to_count = dict(sorted(attr_value_results["same_attr_count_to_count"].items(), key=lambda x: x[0]))
            for same_attr_count, count in same_attr_count_to_count.items():
                if same_attr_count not in results[attr_value]:
                    results[attr_value][same_attr_count] = {"hit_rate": []}
                hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
                hit_rate = hit_count / count
                results[attr_value][same_attr_count]["hit_rate"].append(hit_rate)
        repeat_count += 1

    for attr_value, attr_value_results in results.items():
        print(f"attr_value: {attr_value}")
        for same_attr_count, same_attr_count_results in attr_value_results.items():
            all_hit_rates = same_attr_count_results["hit_rate"]
            mean_hit_rate, lower_ci, upper_ci = prob_ci_from_estimates(all_hit_rates)
            print(f"same_attr_count: {same_attr_count}, hit_rate: {mean_hit_rate:.6f} [{lower_ci:.6f}, {upper_ci:.6f}]")
            same_attr_count_results["hit_rate"] = mean_hit_rate
            same_attr_count_results["ci_low"] = lower_ci
            same_attr_count_results["ci_high"] = upper_ci

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
    save_file = f"outputs/contextual_{model_name}_{attribute_type}_{resume_count}-v2.png"
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    pool_count = 5
    anchor_idx_to_resume_idx = get_anchor_idx_to_resume_idx(pool_count)

    for attribute_type in ["Gender"]:
        for resume_count in [5]:
            for model_name in ["msra-gpt-4.1-nano", "msra-gpt-4o"]:
                file_name = f"outputs/contextual/{attribute_type}/{model_name}_{resume_count}_{pool_count}.jsonl"
                if os.path.exists(file_name):
                    print(f"------------------------------------\n\n{file_name}")
                    results = compute_results(file_name, attribute_type, pool_count, anchor_idx_to_resume_idx, max_line=3800000)
                    draw_results(model_name, attribute_type, resume_count, results)
