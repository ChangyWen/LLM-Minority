import json
import sys
from collections import defaultdict
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


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
                attr_value_to_results[attr_value]["same_attr_count_to_hit_count"][same_attr_count] += (1 if inner_idx == suggested_candidate_id else 0)

    # print(f"Attribute type: {attribute_type}")
    results = {}
    attr_counts_A = None
    attr_counts_B = None
    for attr_value, attr_value_results in attr_value_to_results.items():
        # sort the attr_value_results by same_attr_count
        # print(f"attr_value: {attr_value}")
        results[attr_value] = {}

        # store raw counts for global and trend tests
        attr_counts = {}

        # sort attr_value_results["same_attr_count_to_count"]
        same_attr_count_to_count = dict(sorted(attr_value_results["same_attr_count_to_count"].items(), key=lambda x: x[0]))
        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
            attr_counts[same_attr_count] = (hit_count, count)

        # attr_counts_A/B for delta trend test
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
        delta = pA - pB
        results["delta"][c] = delta

    return abs(results["delta"][1])


def draw_results(application_to_model_to_delta, attribute_type):
    """
    Draw a radar chart for one attribute_type.

    - One figure per attribute_type
    - Angular axis: models
    - Radial axis: delta
    - One polygon per application (different colors)
    """
    # Match global style with draw_results_grid
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

    # Applications will be polygons (areas)
    applications = sorted(application_to_model_to_delta.keys())
    # Assume all applications share the same model set
    some_app = next(iter(application_to_model_to_delta.values()))
    models = sorted(some_app.keys())

    n_models = len(models)

    # Angles for each model (and wrap around to close the loop)
    angles = np.linspace(0, 2 * np.pi, n_models, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    # Prepare figure (size similar visual weight as grid panels)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))

    # Color palette: one color per application
    palette = sns.color_palette("Set2", n_colors=len(applications))

    # Consistent radial limit
    max_delta = max(
        delta for app_dict in application_to_model_to_delta.values()
        for delta in app_dict.values()
    )
    ax.set_ylim(0, max_delta * 1.05)

    # Fewer radial ticks, nicely formatted (like in plot_model_panel)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))  # <= fewer ticks
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Style grid similar to main figures
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)

    # Style polar spine to match axes.edgecolor / linewidth
    if "polar" in ax.spines:
        ax.spines["polar"].set_edgecolor("black")
        ax.spines["polar"].set_linewidth(0.8)

    # Plot each application
    for i, application in enumerate(applications):
        # Collect delta values across models for this application
        values = [application_to_model_to_delta[application][m] for m in models]
        # Close the loop
        values = np.concatenate([values, [values[0]]])

        ax.plot(angles, values, label=application, color=palette[i], linewidth=1.8)
        ax.fill(angles, values, color=palette[i], alpha=0.18)

    # Set angular ticks to models
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models, fontsize=10, rotation=30, ha="right")

    # Radial axis label position; we don't actually show a text label,
    # to keep it clean (like your commented-out ylabel)
    ax.set_rlabel_position(0)

    # Title = attribute type
    ax.set_title(
        f"{attribute_type} (Abs. Δ of different groups)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Legend for applications – match style used in plot_model_panel
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=9,
        title="Application",
        title_fontsize=10,
        markerscale=1,
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1),
        frameon=True,
        framealpha=0.5,
        borderpad=0.3,
    )

    plt.tight_layout()
    out_path = f"outputs/radar_contextual_{attribute_type}.png"
    plt.savefig(out_path, dpi=512, bbox_inches="tight")
    print(f"Saved radar chart to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    applications = ["edu", "hiring", "loan"]

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

    attribute_types = ["Gender", "Race"]

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(float))
        for application in applications:
            for model_name in model_names:
                file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_5_500.jsonl"
                if not os.path.exists(file_name):
                    file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_5_200.jsonl"
                if not os.path.exists(file_name):
                    assert False, f"File not found: {application} {attribute_type} {model_name}"
                delta = compute_results(file_name, attribute_type)
                application_to_model_to_delta[application][model_name] = delta

        draw_results(application_to_model_to_delta, attribute_type)

