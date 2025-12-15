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
        results["delta"][c] = pA - pB

    return abs(results["delta"][1])


def fit_linear_with_ci(x, y, alpha=0.05):
    """
    Fit y = a*x + b and return:
    - x_grid
    - y_pred
    - lower CI
    - upper CI
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Linear fit
    coef = np.polyfit(x, y, 1)
    y_hat = np.polyval(coef, x)

    n = len(x)
    dof = n - 2
    s_err = np.sqrt(np.sum((y - y_hat) ** 2) / dof)

    x_mean = np.mean(x)
    t_val = t.ppf(1 - alpha / 2, dof)

    x_grid = np.linspace(x.min(), x.max(), 200)
    y_grid = np.polyval(coef, x_grid)

    ci = t_val * s_err * np.sqrt(
        1 / n + (x_grid - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
    )

    return x_grid, y_grid, y_grid - ci, y_grid + ci


def draw_scatter_by_application(
    application_to_model_to_delta,
    model_to_training_compute,
    attribute_type,
):
    """
    For each application, draw a scatter plot:
      x-axis: training compute
      y-axis: delta
    One figure per application (per attribute_type).
    """

    # Match global style with your current code
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
    some_app = next(iter(application_to_model_to_delta.values()))
    models = sorted(some_app.keys())

    # One consistent color per model
    palette = sns.color_palette("Set2", n_colors=len(models))
    model_to_color = {m: palette[i] for i, m in enumerate(models)}

    for application in applications:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Remove upper and right-hand side boundaries
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        xs, ys, cs = [], [], []
        for m in models:
            xs.append(model_to_training_compute[m])
            ys.append(application_to_model_to_delta[application][m])
            cs.append(model_to_color[m])

        # Scatter
        ax.scatter(
            xs,
            ys,
            s=55,
            c=cs,
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )

        # ---- Regression (log-space x) ----
        x_log = np.log10(xs)
        y_arr = np.array(ys)

        # Pearson r (one-sided: positive correlation)
        r, p_two_sided = pearsonr(x_log, y_arr)
        p_one_sided = p_two_sided / 2 if r > 0 else 1.0

        # Fit line + CI
        xg, yg, yl, yu = fit_linear_with_ci(x_log, y_arr)

        # Convert back to original x-scale for plotting
        ax.plot(
            10 ** xg,
            yg,
            color="black",
            linewidth=2.0,
            zorder=2,
            label="Linear fit",
        )
        ax.fill_between(
            10 ** xg,
            yl,
            yu,
            color="black",
            alpha=0.15,
            zorder=1,
            label="95% CI",
        )
        # ax.text(
        #     0.02,
        #     0.95,
        #     f"Pearson r = {r:.2f}\n"
        #     f"p (one-sided) = {p_one_sided:.3g}",
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=11,
        #     bbox=dict(
        #         boxstyle="round,pad=0.3",
        #         facecolor="white",
        #         edgecolor="black",
        #         alpha=0.9,
        #     ),
        # )

        # Keep labels beside each point
        for x, y, m in zip(xs, ys, models):
            m = m.replace("msra-", "")
            ax.annotate(
                m,
                (x, y),
                textcoords="offset points",
                xytext=(6, 5),
                ha="left",
                va="bottom",
                fontsize=9,
                alpha=0.9,
            )

        # Axes
        ax.set_xlabel("Training compute (FLOP)")
        ax.set_ylabel("Abs. Δ of Selection Rate (w.r.t. min. contextual ratio)")

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())

        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda v, pos: f"{v*100:.0f}%")
        )

        # Manually add scale label on x-axis (without scientific ticks)
        ax.text(
            1.01, -0.03, "1e+23",
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="top",
        )

        # Grid & spines (same style as before)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

        # ✅ Updated title format
        ax.set_title(
            f"{application.capitalize()} - {attribute_type}",
            fontsize=16,
            fontweight="bold",
            pad=12,
        )

        plt.tight_layout()
        out_path = f"outputs/training-compute/contextual_compute_vs_delta_{application}_{attribute_type}.png"
        plt.savefig(out_path, dpi=512, bbox_inches="tight")
        print(f"Saved scatter plot to {out_path}")
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

    model_to_training_compute = {
        "msra-gpt-4o": 3.8e+2,
        "gpt-oss-120b": 4.94e+1,
        "Qwen3-235B-A22B-Instruct-2507": 4.752e+1,
        "Qwen3-Next-80B-A3B-Instruct": 2.7e+0,
        "GLM-4.5-Air": 1.656e+1,
        "gemma-3-27b-it": 2.268e+1,
        "Llama-3.3-70B-Instruct": 6.86498e+1,
        "NVIDIA-Nemotron-Nano-12B-v2": 1.5192e+1,
    }

    attribute_types = ["Gender", "Race"]

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(float))
        for application in applications:
            for model_name in model_names:
                file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_5_500.jsonl"
                if not os.path.exists(file_name):
                    file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_5_200.jsonl"
                if not os.path.exists(file_name):
                    raise FileNotFoundError(f"File not found: {application} {attribute_type} {model_name}")

                delta = compute_results(file_name, attribute_type)
                application_to_model_to_delta[application][model_name] = delta

        draw_scatter_by_application(
            application_to_model_to_delta=application_to_model_to_delta,
            model_to_training_compute=model_to_training_compute,
            attribute_type=attribute_type,
        )
