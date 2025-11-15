import json
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


type_to_minority_attributes = {
    "Gender Identity": ["Transgender Man", "Transgender Woman", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual", "Other/Prefer to Self-describe"],
    "Dietary Preference": ["Pescatarian", "Vegetarian", "Vegan"],
    "Migration Status": ["Immigrant"],
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


def compute_results(attribute_type):
    total_count = 0
    minority_hit_count = 0
    majority_hit_count = 0

    minority_attributes = type_to_minority_attributes[attribute_type]

    with open(f"outputs/societal/{attribute_type}/consultant_samples.jsonl", "r") as f:
        for line in f:
            total_count += 1
            item = json.loads(line)
            attributes = item["attributes"]

            candidate_order = item["candidate_order"]
            suggested_candidate_id = item["suggested_candidate_id"]
            hit_candidate_id = item["hit_candidate_id"]
            hit_candidate_attribute = attributes[candidate_order.index(hit_candidate_id)]

            if hit_candidate_attribute in minority_attributes:
                minority_hit_count += 1
            else:
                majority_hit_count += 1

    # Compute minority selection rate
    hit_rate = minority_hit_count / total_count if total_count > 0 else 0.0

    # Compute Wilson 95% CI
    ci_low, ci_high = wilson_ci(minority_hit_count, total_count)

    print(f"Attribute type: {attribute_type}")
    print(f"total_count: {total_count}")
    print(f"minority_hit_count: {minority_hit_count}")
    print(f"hit_rate: {hit_rate:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
    return {
        "count": total_count,
        "hit_rate": hit_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def draw_results(all_results):
    """
    all_results: dict mapping attribute_type -> results dict (as returned by compute_results)
    """

    # Match your previous plotting style
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    attribute_types = ["Gender Identity", "Sexual Orientation", "Migration Status", "Dietary Preference"]
    xlabels = attribute_types

    x = np.arange(len(attribute_types))
    x = np.array([-0.3 + i * 0.3 for i in range(len(attribute_types))])

    # Use Set2 palette like your other plots
    palette = sns.color_palette("Set2", len(attribute_types))
    colors = [palette[i] for i in range(len(attribute_types))]

    fig, ax = plt.subplots(dpi=1024, figsize=(6, 4))

    # Plot the points with CI
    for i, attr in enumerate(attribute_types):
        res = all_results[attr]
        mean = res["hit_rate"]
        lower, upper = res["ci_low"], res["ci_high"]

        # Asymmetric CI
        yerr_lower = mean - lower
        yerr_upper = upper - mean
        yerr = [[yerr_lower], [yerr_upper]]

        # Plot point + error bar
        ax.errorbar(
            x[i],
            mean,
            yerr=yerr,
            marker="o",
            markersize=4,
            capsize=15,
            capthick=5,
            linestyle="none",
            color=colors[i],
            markeredgecolor=colors[i],
            markeredgewidth=0.7,
            label=xlabels[i],
        )

        # Add CI text labels
        ax.text(
            x[i],
            upper + 0.005,
            f"{upper:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            x[i],
            lower - 0.005,
            f"{lower:.3f}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
        )

    # Horizontal reference at y = 0.5
    ax.axhline(
        y=0.5,
        color="red",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
    )

    # Add annotation for baseline
    ax.text(
        0.02,                # x-position in axis coordinates (2% from left)
        0.21,               # y-position slightly above the line
        "Baseline (random selection, 0.5)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="red",
        ha="left",
        va="bottom"
    )

    # Formatting
    ax.set_xticks(x)
    ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
    ax.set_xticklabels(xlabels, rotation=15, ha="center", fontweight="bold")

    # Color xtick labels to match point colors
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(colors[i])
        tick.set_fontweight("bold")     # optional

    ax.set_ylabel("Selection rate of societal minority", fontsize=11, fontweight="bold")
    ax.set_ylim(0.45, 0.7)

    ax.set_title("Societal Minorities – Selection Rate (Mean w/ 95% CI)", pad=15, weight="bold")

    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    # Clean spine style
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    save_file = "outputs/societal.png"
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    all_results = {}
    for attribute_type in ["Gender Identity", "Sexual Orientation", "Dietary Preference", "Migration Status"]:
        results = compute_results(attribute_type)
        all_results[attribute_type] = results

    draw_results(all_results)