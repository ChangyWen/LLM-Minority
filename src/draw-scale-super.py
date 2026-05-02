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
from scipy import stats


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}


def t_ci(scores, confidence=0.95):
    scores = np.asarray(scores, dtype=float)

    mean = np.mean(scores)
    sem = stats.sem(scores)
    df = len(scores) - 1

    if df <= 0 or np.isnan(sem):
        return mean, mean, mean

    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    lower = mean - t_crit * sem
    upper = mean + t_crit * sem

    return mean, lower, upper


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


def compute_societal_results(file_name, attribute_type):
    """
    Compute relative score difference:

        (minority_mean - majority_mean) / majority_mean

    Positive values indicate higher average scores for societal minorities.
    Negative values indicate lower average scores for societal minorities.
    """

    minority_scores = []
    majority_scores = []

    minority_attributes = type_to_minority_attributes[attribute_type]

    with open(file_name, "r") as f:
        for line in f:
            item = json.loads(line)
            attribute = item["attribute"]
            score = item["score"]

            if attribute in minority_attributes:
                minority_scores.append(score)
            else:
                majority_scores.append(score)

    minority_scores = np.asarray(minority_scores, dtype=float)
    majority_scores = np.asarray(majority_scores, dtype=float)

    if len(minority_scores) == 0 or len(majority_scores) == 0:
        return None

    minority_mean, _, _ = t_ci(minority_scores)
    majority_mean, _, _ = t_ci(majority_scores)

    if majority_mean == 0:
        return None

    delta = (minority_mean - majority_mean) / majority_mean

    return delta


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


def format_param_ticks(val, pos):
    return f"{int(val):d}" if val >= 1 else f"{val:g}"


def get_model_colors(model_names):
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
    return {m: palette[i % len(palette)] for i, m in enumerate(model_names)}


def format_log_ticks(val, pos):
    return f"{val:.1f}"


def draw_scale_block(
    fig,
    outer_spec,
    attribute_type_to_application_to_model_to_delta,
    model_to_x_value,
    attribute_types,
    model_names,
    model_to_color,
    panel_letter,
    block_title,
    xlabel,
    ylabel=None,
):
    """
    Draw one 2x3 block inside a larger GridSpec cell.

    Rows: attribute types
    Columns: Hiring, Loan, Scholarship

    The x-axis is log10-transformed before plotting.
    """

    applications = ["hiring", "loan", "edu"]
    panel_titles = {
        "edu": "Scholarship application",
        "hiring": "Hiring",
        "loan": "Loan approval",
    }

    inner_gs = outer_spec.subgridspec(
        2,
        3,
        wspace=0.12,
        hspace=0.90,
    )

    axes = np.empty((2, 3), dtype=object)

    all_x = [model_to_x_value[m] for m in model_names if model_to_x_value[m] > 0]
    all_x_log = np.log10(all_x)

    x_min = min(all_x_log) - 0.08
    x_max = max(all_x_log) + 0.08

    for row_idx, attribute_type in enumerate(attribute_types):
        application_to_model_to_delta = attribute_type_to_application_to_model_to_delta[attribute_type]

        for col_idx, application in enumerate(applications):
            ax = fig.add_subplot(inner_gs[row_idx, col_idx])
            axes[row_idx, col_idx] = ax

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            xs_raw = np.array(
                [model_to_x_value[m] for m in model_names],
                dtype=float,
            )
            ys = np.array(
                [application_to_model_to_delta[application][m] for m in model_names],
                dtype=float,
            )

            valid_mask = np.isfinite(xs_raw) & np.isfinite(ys) & (xs_raw > 0)

            xs_valid = np.log10(xs_raw[valid_mask])
            ys_valid = ys[valid_mask]
            valid_models = [
                m for m, ok in zip(model_names, valid_mask)
                if ok
            ]

            for m, x, y in zip(valid_models, xs_valid, ys_valid):
                ax.scatter(
                    x,
                    y,
                    s=54,
                    color=model_to_color[m],
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.95,
                    zorder=3,
                )

            if len(xs_valid) >= 3 and np.std(ys_valid) > 0 and np.std(xs_valid) > 0:
                r, p_two_sided = pearsonr(xs_valid, ys_valid)
                p_one_sided = p_two_sided / 2 if r > 0 else 1.0

                xg, yg, yl, yu = fit_linear_with_ci(xs_valid, ys_valid)

                ax.plot(
                    xg,
                    yg,
                    color="0.15",
                    linewidth=1.35,
                    zorder=2,
                )

                ax.fill_between(
                    xg,
                    yl,
                    yu,
                    color="0.35",
                    alpha=0.12,
                    zorder=1,
                )

                corr_text = rf"$r={r:.2f}$" + "\n" + rf"$P={p_two_sided:.2f}$"
            else:
                corr_text = r"$r=\mathrm{NA}$" + "\n" + r"$P=\mathrm{NA}$"

            if row_idx == 0:
                ax.set_title(
                    panel_titles[application],
                    pad=8,
                    fontsize=16,
                )

            ax.text(
                0.03,
                0.96,
                corr_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=16,
                color="0.15",
            )

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

            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, pos: f"{v * 100:.0f}")
            )
            ax.xaxis.set_major_formatter(
                FuncFormatter(format_log_ticks)
            )

    pos_top_left = axes[0, 0].get_position()
    pos_top_right = axes[0, 2].get_position()
    pos_bottom_left = axes[1, 0].get_position()

    block_x0 = pos_top_left.x0
    block_x1 = pos_top_right.x1
    block_y1 = pos_top_left.y1
    block_y0 = pos_bottom_left.y0
    block_x_center = (block_x0 + block_x1) / 2
    block_y_center = (block_y0 + block_y1) / 2

    fig.text(
        block_x0 - 0.020,
        block_y1 + 0.060,
        panel_letter,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )

    fig.text(
        block_x0 + 0.005,
        block_y1 + 0.060,
        block_title,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )

    row_title_offset = 0.03

    for row_idx, attribute_type in enumerate(attribute_types):
        pos_row_left = axes[row_idx, 0].get_position()
        pos_row_right = axes[row_idx, 2].get_position()

        row_x_center = (pos_row_left.x0 + pos_row_right.x1) / 2
        row_y = pos_row_left.y1 + row_title_offset

        fig.text(
            row_x_center,
            row_y,
            attribute_type,
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    fig.text(
        block_x_center,
        block_y0 - 0.043,
        xlabel,
        ha="center",
        va="top",
        fontsize=16,
    )

    if ylabel is not None:
        fig.text(
            block_x0 - 0.03,
            block_y_center,
            ylabel,
            ha="center",
            va="center",
            rotation=90,
            fontsize=16,
        )

    return axes


def draw_super_scale_figure(
    contextual_attribute_type_to_application_to_model_to_delta,
    societal_attribute_type_to_application_to_model_to_delta,
    model_to_parameter_count,
    model_to_training_compute,
    contextual_attribute_types,
    societal_attribute_types,
    model_names,
    output_dir="outputs/parameter",
):
    """
    Draw one super big Nature-style figure.

    Layout:
        a: Contextual results vs model parameters
        b: Contextual results vs training compute
        c: Societal results vs model parameters
        d: Societal results vs training compute
    """

    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    model_to_color = get_model_colors(model_names)

    fig = plt.figure(figsize=(15.0, 11.4))

    outer_gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.995,
        bottom=0.22,
        top=0.910,
        wspace=0.08,
        hspace=0.73,
    )

    # a. Contextual results vs parameters
    draw_scale_block(
        fig=fig,
        outer_spec=outer_gs[0, 0],
        attribute_type_to_application_to_model_to_delta=contextual_attribute_type_to_application_to_model_to_delta,
        model_to_x_value=model_to_parameter_count,
        attribute_types=contextual_attribute_types,
        model_names=model_names,
        model_to_color=model_to_color,
        panel_letter="a",
        block_title="Bias towards contextual minority vs. model parameters",
        xlabel=r"$\log_{10}$(Model parameters / $10^9$)",
        ylabel="Absolute selection-rate difference (%)",
    )

    # b. Contextual results vs training compute
    draw_scale_block(
        fig=fig,
        outer_spec=outer_gs[0, 1],
        attribute_type_to_application_to_model_to_delta=contextual_attribute_type_to_application_to_model_to_delta,
        model_to_x_value=model_to_training_compute,
        attribute_types=contextual_attribute_types,
        model_names=model_names,
        model_to_color=model_to_color,
        panel_letter="b",
        block_title="Bias towards contextual minority vs. training compute",
        xlabel=r"$\log_{10}$(Training compute / $10^{23}$ FLOPs)",
        ylabel=None,
    )

    # c. Societal results vs parameters
    draw_scale_block(
        fig=fig,
        outer_spec=outer_gs[1, 0],
        attribute_type_to_application_to_model_to_delta=societal_attribute_type_to_application_to_model_to_delta,
        model_to_x_value=model_to_parameter_count,
        attribute_types=societal_attribute_types,
        model_names=model_names,
        model_to_color=model_to_color,
        panel_letter="c",
        block_title="Bias towards societal minorities vs. model parameters",
        xlabel=r"$\log_{10}$(Model parameters / $10^9$)",
        ylabel="Relative difference in score (%)",
    )

    # d. Societal results vs training compute
    draw_scale_block(
        fig=fig,
        outer_spec=outer_gs[1, 1],
        attribute_type_to_application_to_model_to_delta=societal_attribute_type_to_application_to_model_to_delta,
        model_to_x_value=model_to_training_compute,
        attribute_types=societal_attribute_types,
        model_names=model_names,
        model_to_color=model_to_color,
        panel_letter="d",
        block_title="Bias towards societal minorities vs. training compute",
        xlabel=r"$\log_{10}$(Training compute / $10^{23}$ FLOPs)",
        ylabel=None,
    )

    # Shared model legend below the whole figure.
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=model_to_color[m],
            markeredgecolor="none",
            markersize=8.0,
            label=pretty_model_name(m),
        )
        for m in model_names
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=4,
        frameon=False,
        fontsize=16,
        handletextpad=0.45,
        columnspacing=1.35,
    )

    base = "scale_super_figure_contextual_societal_parameter_compute_nature_style"

    pdf_path = os.path.join(output_dir, base + ".pdf")
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

    # ------------------------------------------------------------
    # Load contextual results
    # ------------------------------------------------------------
    contextual_attribute_types = ["Gender", "Race"]
    contextual_attribute_type_to_application_to_model_to_delta = {}

    for attribute_type in contextual_attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(float))

        for application in applications:
            for model_name in model_names:
                file_name = (
                    f"outputs/{application}/contextual/"
                    f"{attribute_type}/{model_name}_5_500.jsonl"
                )

                if not os.path.exists(file_name):
                    file_name = (
                        f"outputs/{application}/contextual/"
                        f"{attribute_type}/{model_name}_5_200.jsonl"
                    )

                if not os.path.exists(file_name):
                    raise FileNotFoundError(
                        f"File not found: {application} {attribute_type} {model_name}"
                    )

                delta = compute_contextual_results(file_name, attribute_type)
                application_to_model_to_delta[application][model_name] = delta

        contextual_attribute_type_to_application_to_model_to_delta[attribute_type] = application_to_model_to_delta

    # ------------------------------------------------------------
    # Load societal results
    # ------------------------------------------------------------
    societal_attribute_types = ["Gender Identity", "Sexual Orientation"]
    societal_attribute_type_to_application_to_model_to_delta = {}

    for attribute_type in societal_attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(float))

        for application in applications:
            for model_name in model_names:
                file_name = (
                    f"outputs/{application}/societal/"
                    f"{attribute_type}/{model_name}.jsonl"
                )

                if not os.path.exists(file_name):
                    raise FileNotFoundError(
                        f"File not found: {application} {attribute_type} {model_name}"
                    )

                delta = compute_societal_results(file_name, attribute_type)

                if delta is None:
                    delta = np.nan

                application_to_model_to_delta[application][model_name] = delta

        societal_attribute_type_to_application_to_model_to_delta[attribute_type] = application_to_model_to_delta

    # ------------------------------------------------------------
    # Draw super figure
    # ------------------------------------------------------------
    draw_super_scale_figure(
        contextual_attribute_type_to_application_to_model_to_delta=contextual_attribute_type_to_application_to_model_to_delta,
        societal_attribute_type_to_application_to_model_to_delta=societal_attribute_type_to_application_to_model_to_delta,
        model_to_parameter_count=model_to_parameter_count,
        model_to_training_compute=model_to_training_compute,
        contextual_attribute_types=contextual_attribute_types,
        societal_attribute_types=societal_attribute_types,
        model_names=model_names,
        output_dir="outputs/parameter",
    )