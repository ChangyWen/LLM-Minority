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


def format_param_ticks(val, pos):
    return f"{int(val):d}" if val >= 1 else f"{val:g}"


def draw_combined_societal_scatter_panels(
    attribute_type_to_application_to_model_to_delta,
    model_to_parameter_count,
    attribute_types,
    model_names,
    output_dir="outputs/parameter",
):
    """
    Draw one combined Nature-style 2x3 figure for societal minority results:
        Top row: Gender Identity
        Bottom row: Sexual Orientation
        Columns: Hiring, Loan, Scholarship

    This follows the same visual style as the contextual-result figure.
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

    # Same palette as contextual figure
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
    all_x = [model_to_parameter_count[m] for m in model_names]
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
                [model_to_parameter_count[m] for m in model_names],
                dtype=float,
            )
            ys = np.array(
                [application_to_model_to_delta[application][m] for m in model_names],
                dtype=float,
            )

            valid_mask = np.isfinite(xs) & np.isfinite(ys)
            xs_valid = xs[valid_mask]
            ys_valid = ys[valid_mask]
            valid_models = [
                m for m, ok in zip(model_names, valid_mask)
                if ok
            ]

            # Scatter points
            for m, x, y in zip(valid_models, xs_valid, ys_valid):
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
            x_log = np.log10(xs_valid)

            if len(xs_valid) >= 3 and np.std(ys_valid) > 0:
                r, p_two_sided = pearsonr(x_log, ys_valid)
                p_one_sided = p_two_sided / 2 if r > 0 else 1.0

                xg, yg, yl, yu = fit_linear_with_ci(x_log, ys_valid)

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

                corr_text = rf"$r={r:.2f}$" + "\n" + rf"$P={p_one_sided:.2f}$"
            else:
                corr_text = r"$r=\mathrm{NA}$" + "\n" + r"$P=\mathrm{NA}$"

            ax.set_title(
                panel_titles[application],
                pad=5,
                fontsize=10,
            )

            # Correlation text
            ax.text(
                0.03,
                0.96,
                corr_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="0.15",
            )

            # Axes and styling
            ax.set_xscale("log")
            ax.set_xlim(x_min, x_max)

            ax.set_axisbelow(True)

            ax.tick_params(
                axis="both",
                direction="out",
                length=3.0,
                width=0.7,
            )
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, pos: f"{v * 100:.0f}")
            )
            ax.xaxis.set_major_formatter(
                FuncFormatter(format_param_ticks)
            )

    # Shared labels
    fig.supxlabel(
        r"Model parameters ($\times 1$B, log scale)",
        fontsize=10,
        y=0.060,
    )

    fig.supylabel(
        "Relative difference in score (%)",
        fontsize=10,
        x=0.05,
    )

    # No legend for now.

    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        bottom=0.145,
        top=0.835,
        wspace=0.12,
        hspace=0.58,
    )

    # Row subtitles
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
            fontsize=10,
            fontweight="bold",
        )

    base = "societal_parameter_vs_delta_GenderIdentity_SexualOrientation_combined_nature_style"
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

    attribute_types = ["Gender Identity", "Sexual Orientation"]

    attribute_type_to_application_to_model_to_delta = {}

    for attribute_type in attribute_types:
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

        attribute_type_to_application_to_model_to_delta[attribute_type] = application_to_model_to_delta

    draw_combined_societal_scatter_panels(
        attribute_type_to_application_to_model_to_delta=attribute_type_to_application_to_model_to_delta,
        model_to_parameter_count=model_to_parameter_count,
        attribute_types=attribute_types,
        model_names=model_names,
    )