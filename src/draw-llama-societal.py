import json
import math
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}


# ============================================================
# Statistical helpers
# ============================================================

def p_to_stars(p):
    """
    Convert p-value to significance stars.
    """
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


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


def compute_results(attribute_type, file_name):
    """
    Compute mean scores and 95% CIs for minority and majority groups.

    The p-value tests:
        H0: minority and majority score distributions are the same
        H1: minority and majority score distributions are different

    Test: two-sided Mann-Whitney U test.
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

    minority_mean, minority_ci_low, minority_ci_high = t_ci(minority_scores)
    majority_mean, majority_ci_low, majority_ci_high = t_ci(majority_scores)

    stat, p_value = stats.mannwhitneyu(
        minority_scores,
        majority_scores,
        alternative="two-sided",
    )

    return {
        "minority": {
            "mean": minority_mean,
            "ci_low": minority_ci_low,
            "ci_high": minority_ci_high,
        },
        "majority": {
            "mean": majority_mean,
            "ci_low": majority_ci_low,
            "ci_high": majority_ci_high,
        },
        "p_value": p_value,
    }


# ============================================================
# Nature-style plotting helpers
# ============================================================

def set_nature_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        # Keep text editable in Illustrator / Inkscape
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "figure.dpi": 150,
        "savefig.dpi": 600,

        "axes.linewidth": 0.7,
        "axes.edgecolor": "0.15",
        "axes.spines.top": False,
        "axes.spines.right": False,

        "axes.titlesize": 8.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.5,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": 1.25,
    })


def pretty_model_name(model_key):
    mapping = {
        "Llama-3.1-8B": "Llama-3.1-8B",
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    }
    return mapping.get(model_key, model_key)


def add_sig_bracket(
    ax,
    x1,
    x2,
    y,
    text,
    bar_height,
    text_offset,
    linewidth=0.8,
):
    """
    Add a significance bracket between two x positions.
    """

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + bar_height, y + bar_height, y],
        color="black",
        linewidth=linewidth,
        clip_on=False,
        zorder=5,
    )

    ax.text(
        (x1 + x2) / 2,
        y + bar_height + text_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="black",
        clip_on=False,
        zorder=6,
    )


# ============================================================
# Plotting
# ============================================================

def plot_societal_panel(
    ax,
    application,
    model_name,
    attribute_types,
    attribute_to_color,
):
    """
    Draw one panel:
        one application x one model.

    X-axis:
        Gender Identity, Sexual Orientation

    For each attribute:
        minority and majority scores with 95% CI.
    """

    minority_marker = "o"
    majority_marker = "s"

    x_base = np.arange(len(attribute_types), dtype=float)
    dodge = 0.13

    x_minority = x_base - dodge
    x_majority = x_base + dodge

    panel_low_values = []
    panel_high_values = []
    bracket_info = []

    for i, attribute_type in enumerate(attribute_types):
        file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"

        if not os.path.exists(file_name):
            print(f"[Warning] Missing file: {file_name}")
            continue

        res = compute_results(attribute_type, file_name)

        if res is None:
            print(f"[Warning] No valid result: {application}, {attribute_type}, {model_name}")
            continue

        color = attribute_to_color[attribute_type]

        min_res = res["minority"]
        maj_res = res["majority"]

        # Minority
        min_mean = float(min_res["mean"])
        min_low = float(min_res["ci_low"])
        min_high = float(min_res["ci_high"])

        ax.errorbar(
            x_minority[i],
            min_mean,
            yerr=[[min_mean - min_low], [min_high - min_mean]],
            marker=minority_marker,
            markersize=4.8,
            capsize=2.4,
            capthick=0.8,
            elinewidth=0.8,
            linestyle="",
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=3,
        )

        # Majority
        maj_mean = float(maj_res["mean"])
        maj_low = float(maj_res["ci_low"])
        maj_high = float(maj_res["ci_high"])

        ax.errorbar(
            x_majority[i],
            maj_mean,
            yerr=[[maj_mean - maj_low], [maj_high - maj_mean]],
            marker=majority_marker,
            markersize=4.8,
            capsize=2.4,
            capthick=0.8,
            elinewidth=0.8,
            linestyle="",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.0,
            zorder=3,
        )

        panel_low_values.extend([min_low, maj_low])
        panel_high_values.extend([min_high, maj_high])

        p_value = float(res["p_value"])
        stars = p_to_stars(p_value)

        y_pair_top = max(min_high, maj_high)
        bracket_info.append({
            "x1": x_minority[i],
            "x2": x_majority[i],
            "y_top": y_pair_top,
            "stars": stars,
            "p_value": p_value,
            "attribute_type": attribute_type,
        })

        print(
            f"{application} | {model_name} | {attribute_type}: "
            f"minority vs majority, two-sided P={p_value:.4g}"
        )

    # Panel-specific y-limits with space for brackets
    if panel_high_values:
        data_min = min(panel_low_values)
        data_max = max(panel_high_values)
    else:
        data_min, data_max = 0.0, 1.0

    data_span = max(data_max - data_min, 0.5)

    bracket_offset = 0.08 * data_span
    bracket_height = 0.035 * data_span
    text_offset = 0.020 * data_span

    bracket_tops = []

    for b in bracket_info:
        y_bracket = b["y_top"] + bracket_offset

        add_sig_bracket(
            ax=ax,
            x1=b["x1"],
            x2=b["x2"],
            y=y_bracket,
            text=b["stars"],
            bar_height=bracket_height,
            text_offset=text_offset,
        )

        bracket_tops.append(y_bracket + bracket_height + text_offset)

    y_lower = data_min - 0.12 * data_span
    y_upper = max([data_max] + bracket_tops) + 0.14 * data_span

    ax.set_ylim(y_lower, y_upper)

    # Axis formatting
    ax.set_xticks(x_base)
    ax.set_xticklabels(attribute_types)

    for tick_label, attribute_type in zip(ax.get_xticklabels(), attribute_types):
        tick_label.set_color(attribute_to_color[attribute_type])
        # tick_label.set_fontweight("bold")

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))

    ax.tick_params(
        axis="both",
        direction="out",
        length=3.0,
        width=0.7,
        color="black",
        labelcolor="black",
    )

    # Re-apply x tick colors after tick_params
    for tick_label, attribute_type in zip(ax.get_xticklabels(), attribute_types):
        tick_label.set_color(attribute_to_color[attribute_type])
        # tick_label.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis="y",
        color="0.88",
        linewidth=0.6,
        linestyle="-",
        zorder=0,
    )
    ax.set_axisbelow(True)


def draw_all_applications_societal(
    applications,
    attribute_types,
    model_names,
    output_dir="outputs/llama",
):
    """
    Draw one Nature-style 3 x 2 figure.

    Rows:
        applications

    Columns:
        Llama-3.1-8B and Llama-3.1-8B-Instruct
    """

    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship application",
    }

    # Professional, colorblind-safe colors
    attribute_to_color = {
        "Gender Identity": "#0072B2",
        "Sexual Orientation": "#D55E00",
    }

    fig, axes = plt.subplots(
        len(applications),
        len(model_names),
        figsize=(7.45, 7.9),
        sharex=False,
        sharey=False,
    )

    axes = np.asarray(axes)

    for row_idx, application in enumerate(applications):
        for col_idx, model_name in enumerate(model_names):
            ax = axes[row_idx, col_idx]

            plot_societal_panel(
                ax=ax,
                application=application,
                model_name=model_name,
                attribute_types=attribute_types,
                attribute_to_color=attribute_to_color,
            )

            # Column titles: model names
            ax.set_title(
                pretty_model_name(model_name),
                fontsize=9.0,
                # fontweight="bold",
                pad=5,
            )

    # Shared labels
    fig.supylabel(
        "Mean score",
        fontsize=9.2,
        x=0.035,
    )

    # One shared legend below the figure
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=5.0,
            label="Minority",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=5.0,
            label="Majority",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    fig.subplots_adjust(
        left=0.125,
        right=0.995,
        bottom=0.125,
        top=0.910,
        wspace=0.24,
        hspace=0.56,
    )

    # Row titles: simply the application name
    row_title_offset = 0.032

    for row_idx, application in enumerate(applications):
        pos_left = axes[row_idx, 0].get_position()
        pos_right = axes[row_idx, len(model_names) - 1].get_position()

        row_x_center = (pos_left.x0 + pos_right.x1) / 2
        row_y_top = pos_left.y1

        fig.text(
            row_x_center,
            row_y_top + row_title_offset,
            application_title_map.get(application, application),
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
        )

    base = "all_applications_societal_llama"
    pdf_path = os.path.join(output_dir, base + ".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {pdf_path}")

    plt.close(fig)


if __name__ == "__main__":
    attribute_types = [
        "Gender Identity",
        "Sexual Orientation",
    ]

    model_names = [
        "Llama-3.1-8B",
        "Llama-3.1-8B-Instruct",
    ]

    applications = ["hiring", "loan", "edu"]

    draw_all_applications_societal(
        applications=applications,
        attribute_types=attribute_types,
        model_names=model_names,
        output_dir="outputs/llama",
    )