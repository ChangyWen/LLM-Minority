import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, t


def compute_contextual_results(file_name, attribute_type, max_n_trials=1000000):

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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

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


def set_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "figure.dpi": 150,
        "savefig.dpi": 600,

        "axes.linewidth": 0.7,
        "axes.edgecolor": "0.15",
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": 1.2,
    })


def pretty_model_name(model_key):
    mapping = {
        "msra-gpt-4o": "GPT-4o",
        "gpt-oss-120b": "GPT-OSS-120B",
        "Qwen3-235B-A22B-Instruct-2507": "Qwen3-235B-A22B",
        "Qwen3-Next-80B-A3B-Instruct": "Qwen3-Next-80B-A3B",
        "GLM-4.5-Air": "GLM-4.5-Air",
        "gemma-3-27b-it": "Gemma-3-27B-IT",
        "Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
        "NVIDIA-Nemotron-Nano-12B-v2": "Nemotron-Nano-12B-v2",
    }
    return mapping.get(model_key, model_key.replace("msra-", ""))


def safe_slug(text):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def format_param_ticks(val, pos):
    return f"{int(val):d}" if val >= 1 else f"{val:g}"


def draw_combined_scatter_panels(
    attribute_type_to_application_to_model_to_delta,
    model_to_x_value,
    attribute_types,
    model_names,
    xlabel,
    output_basename,
    output_dir="outputs/parameter",
):
    """
    Draw one combined Nature-style 2x3 figure:
        Top row: Gender
        Bottom row: Race
        Columns: Hiring, Loan, Scholarship

    The x-axis can represent model parameters or effective training compute.
    """

    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    applications = ["hiring", "loan", "edu"]
    panel_titles = {
        "edu": "Scholarship",
        "hiring": "Hiring",
        "loan": "Loan",
    }

    # Colorblind-friendly palette
    palette = [
        "#9ba415ff",  # Olive
        "#459434ff",  # Green
        "#019aa3ff",  # Teal
        "#0272b2ff",  # Blue
        "#a84e94ff",  # Purple
        "#c93e3fff",  # Red
        "#ec6f00ff",  # Orange
        "#cca02cff",  # Yellow
    ]
    model_to_color = {m: palette[i % len(palette)] for i, m in enumerate(model_names)}

    # Global x-range across all panels
    all_x = [model_to_x_value[m] for m in model_names]
    x_min = min(all_x) * 0.85
    x_max = max(all_x) * 1.20

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.45, 5.35),
        sharex=False,
        sharey=False,
    )

    for row_idx, attribute_type in enumerate(attribute_types):
        application_to_model_to_delta = attribute_type_to_application_to_model_to_delta[attribute_type]

        for col_idx, application in enumerate(applications):
            ax = axes[row_idx, col_idx]

            # Remove upper and right boundaries
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            xs = np.array(
                [model_to_x_value[m] for m in model_names],
                dtype=float,
            )
            ys = np.array(
                [application_to_model_to_delta[application][m] for m in model_names],
                dtype=float,
            )

            # Scatter points
            for m, x, y in zip(model_names, xs, ys):
                ax.scatter(
                    x,
                    y,
                    s=50,
                    color=model_to_color[m],
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.95,
                    zorder=3,
                )

            # Regression in log-x space
            x_log = np.log10(xs)
            r, p_two_sided = pearsonr(x_log, ys)
            p_one_sided = p_two_sided / 2 if r > 0 else 1.0

            xg, yg, yl, yu = fit_linear_with_ci(x_log, ys)

            ax.plot(
                10 ** xg,
                yg,
                color="0.15",
                linewidth=1.35,
                zorder=2,
            )
            ax.fill_between(
                10 ** xg,
                yl,
                yu,
                color="0.35",
                alpha=0.12,
                zorder=1,
            )

            ax.set_title(
                panel_titles[application],
                pad=5,
                fontsize=12,
            )

            # Correlation text
            ax.text(
                0.03,
                0.96,
                rf"$r={r:.2f}$" + "\n" + rf"$P={p_one_sided:.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                color="0.15",
            )

            # Axes and styling
            ax.set_xscale("log")
            ax.set_xlim(x_min, x_max)
            ax.set_axisbelow(True)

            ax.tick_params(
                axis="both",
                which="major",
                direction="out",
                length=3.5,
                width=0.8,
                color="black",
                labelcolor="black",
                bottom=True,
                left=True,
                top=False,
                right=False,
            )

            ax.tick_params(
                axis="x",
                which="minor",
                direction="out",
                length=2.0,
                width=0.6,
                color="black",
                bottom=True,
                top=False,
            )

            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, pos: f"{v * 100:.0f}")
            )
            ax.xaxis.set_major_formatter(
                FuncFormatter(format_param_ticks)
            )

    # Shared labels
    fig.supxlabel(
        xlabel,
        fontsize=12,
        y=0.060,
    )

    fig.supylabel(
        "Absolute difference in selection rate (%)",
        fontsize=12,
        x=0.045,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.145,
        top=0.835,
        wspace=0.12,
        hspace=0.58,
    )

    # Row subtitles: Gender and Race
    title_offset = 0.045

    for row_idx, attribute_type in enumerate(attribute_types):
        pos_left = axes[row_idx, 0].get_position()

        x_center = 0.5
        y_text = pos_left.y1 + title_offset

        fig.text(
            x_center,
            y_text,
            attribute_type,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    pdf_path = os.path.join(output_dir, output_basename + ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

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

    # in billions
    model_to_parameter_count = {
        "msra-gpt-4o": 250,
        "gpt-oss-120b": 120,
        "Qwen3-235B-A22B-Instruct-2507": 235,
        "Qwen3-Next-80B-A3B-Instruct": 80,
        "GLM-4.5-Air": 110,
        "gemma-3-27b-it": 27,
        "Llama-3.3-70B-Instruct": 70,
        "NVIDIA-Nemotron-Nano-12B-v2": 12,
    }

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

    attribute_type_to_application_to_model_to_delta = {}

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(float))

        for application in applications:
            for model_name in model_names:
                file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_5_500.jsonl"
                if not os.path.exists(file_name):
                    file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_5_200.jsonl"
                if not os.path.exists(file_name):
                    raise FileNotFoundError(
                        f"File not found: {application} {attribute_type} {model_name}"
                    )

                delta = compute_contextual_results(file_name, attribute_type)
                application_to_model_to_delta[application][model_name] = delta

        attribute_type_to_application_to_model_to_delta[attribute_type] = application_to_model_to_delta

    # Plot against model parameters
    draw_combined_scatter_panels(
        attribute_type_to_application_to_model_to_delta=attribute_type_to_application_to_model_to_delta,
        model_to_x_value=model_to_parameter_count,
        attribute_types=attribute_types,
        model_names=model_names,
        xlabel=r"Model parameters ($\times 1$B, log scale)",
        output_basename="contextual_parameter_vs_delta_Gender_Race_combined_nature_style",
    )

    # Plot against effective training compute
    draw_combined_scatter_panels(
        attribute_type_to_application_to_model_to_delta=attribute_type_to_application_to_model_to_delta,
        model_to_x_value=model_to_training_compute,
        attribute_types=attribute_types,
        model_names=model_names,
        xlabel=r"Training compute in FLOPs ($\times 10^{23}$, log scale)",
        output_basename="contextual_training_compute_vs_delta_Gender_Race_combined_nature_style",
    )