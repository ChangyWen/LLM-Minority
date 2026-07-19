import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy import stats
from scipy.stats import pearsonr, t


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

APPLICATIONS = ["hiring", "loan", "edu"]

PANEL_TITLES = {
    "hiring": "Hiring",
    "loan": "Loan approval",
    "edu": "Scholarship allocation",
}

TYPE_TO_MINORITY_ATTRIBUTES = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}


# -----------------------------------------------------------------------------
# Basic statistics and data processing
# -----------------------------------------------------------------------------

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


def compute_contextual_results(file_name, attribute_type, max_n_trials=1_000_000):
    attr_value_to_results = defaultdict(
        lambda: {
            "same_attr_count_to_count": defaultdict(int),
            "same_attr_count_to_hit_count": defaultdict(int),
        }
    )
    n_trials = 0

    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            attributes = item["attributes"]

            if "Asian" in attributes:
                continue

            if n_trials >= max_n_trials:
                break
            n_trials += 1

            suggested_candidate_id = item["suggested_candidate_id"]

            for inner_idx, attr_value in enumerate(attributes):
                same_attr_count = attributes.count(attr_value) - 1
                attr_value_to_results[attr_value]["same_attr_count_to_count"][
                    same_attr_count
                ] += 1
                attr_value_to_results[attr_value]["same_attr_count_to_hit_count"][
                    same_attr_count
                ] += int(inner_idx == suggested_candidate_id)

    attr_counts_a = None
    attr_counts_b = None

    for attr_value, attr_value_results in attr_value_to_results.items():
        attr_counts = {}
        same_attr_count_to_count = dict(
            sorted(
                attr_value_results["same_attr_count_to_count"].items(),
                key=lambda x: x[0],
            )
        )

        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][
                same_attr_count
            ]
            attr_counts[same_attr_count] = (hit_count, count)

        if attr_value in {"Black", "Female"}:
            attr_counts_a = attr_counts
        else:
            attr_counts_b = attr_counts

    if attr_counts_a is None or attr_counts_b is None:
        raise ValueError(
            f"Could not identify the two comparison groups in {file_name} "
            f"for attribute type {attribute_type}."
        )

    deltas = {}
    for c in sorted(set(attr_counts_a) & set(attr_counts_b)):
        hits_a, count_a = attr_counts_a[c]
        hits_b, count_b = attr_counts_b[c]
        p_a = hits_a / count_a
        p_b = hits_b / count_b
        deltas[c] = p_a - p_b

    # # The original Figure 7 analysis uses c = 1.
    # if 1 not in deltas:
    #     raise ValueError(
    #         f"Candidate-pool composition index c=1 is unavailable in {file_name}."
    #     )

    # return abs(deltas[1])

    # Average over the two compositions in which the target group is
    # numerically underrepresented: q = 0.2 and q = 0.4.
    # Because q = (c + 1) / 5, these correspond to c = 0 and c = 1.
    target_cs = [0, 1, 2, 3]

    missing_cs = [c for c in target_cs if c not in deltas]
    if missing_cs:
        raise ValueError(
            f"Required composition indices are unavailable: {missing_cs}"
        )

    return float(np.mean([abs(deltas[c]) for c in target_cs]))


def compute_societal_results(file_name, attribute_type):
    """
    Compute the relative score difference:

        (minority_mean - majority_mean) / majority_mean

    Positive values indicate higher average scores for societal minorities.
    Negative values indicate lower average scores for societal minorities.
    """
    minority_scores = []
    majority_scores = []

    minority_attributes = TYPE_TO_MINORITY_ATTRIBUTES[attribute_type]

    with open(file_name, "r", encoding="utf-8") as f:
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

    return (minority_mean - majority_mean) / majority_mean


def fit_linear_with_ci(x, y, alpha=0.05):
    """Fit y = a*x + b and return the fitted line and its 95% mean CI."""
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


# -----------------------------------------------------------------------------
# Multiple-testing correction
# -----------------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """
    Return Benjamini-Hochberg adjusted P values.

    Non-finite entries are retained as NaN and are excluded from the number of
    tested hypotheses. The implementation is dependency-free and equivalent to
    an FDR-BH adjustment.
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)

    finite_mask = np.isfinite(p_values)
    finite_p = p_values[finite_mask]

    if finite_p.size == 0:
        return adjusted

    if np.any((finite_p < 0) | (finite_p > 1)):
        raise ValueError("All finite P values must lie between 0 and 1.")

    m = finite_p.size
    order = np.argsort(finite_p)
    ranked_p = finite_p[order]

    # Initial BH values: p_(i) * m / i.
    ranked_adjusted = ranked_p * m / np.arange(1, m + 1)

    # Enforce monotonicity from the largest rank toward the smallest rank.
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)

    # Return values to their original order.
    finite_adjusted = np.empty(m, dtype=float)
    finite_adjusted[order] = ranked_adjusted
    adjusted[finite_mask] = finite_adjusted

    return adjusted


def compute_correlation_family(
    attribute_type_to_application_to_model_to_delta,
    panel_to_x_values,
    attribute_types,
    model_names,
    family_name,
):
    """
    Compute all Pearson correlations in one multiplicity family and apply BH.

    For Figure 7, this function is called twice:
      1. societal minority bias: panels a and b (12 tests), and
      2. contextual minority bias: panels c and d (12 tests).

    Returns a dictionary keyed by (panel_letter, attribute_type, application).
    """
    correlation_results = {}
    valid_test_keys = []
    raw_p_values = []

    for panel_letter, model_to_x_value in panel_to_x_values.items():
        for attribute_type in attribute_types:
            application_to_model_to_delta = (
                attribute_type_to_application_to_model_to_delta[attribute_type]
            )

            for application in APPLICATIONS:
                xs_raw = np.asarray(
                    [model_to_x_value[m] for m in model_names], dtype=float
                )
                ys = np.asarray(
                    [
                        application_to_model_to_delta[application][m]
                        for m in model_names
                    ],
                    dtype=float,
                )

                valid_mask = np.isfinite(xs_raw) & np.isfinite(ys) & (xs_raw > 0)
                xs_valid = np.log10(xs_raw[valid_mask])
                ys_valid = ys[valid_mask]

                key = (panel_letter, attribute_type, application)

                if (
                    len(xs_valid) >= 3
                    and np.std(xs_valid) > 0
                    and np.std(ys_valid) > 0
                ):
                    r_value, raw_p = pearsonr(xs_valid, ys_valid)
                    r_value = float(r_value)
                    raw_p = float(raw_p)

                    valid_test_keys.append(key)
                    raw_p_values.append(raw_p)
                else:
                    r_value = np.nan
                    raw_p = np.nan

                correlation_results[key] = {
                    "family": family_name,
                    "n": int(len(xs_valid)),
                    "r": r_value,
                    "p_raw": raw_p,
                    "p_adjusted": np.nan,
                }

    adjusted_p_values = benjamini_hochberg(raw_p_values)
    for key, adjusted_p in zip(valid_test_keys, adjusted_p_values):
        correlation_results[key]["p_adjusted"] = float(adjusted_p)

    return correlation_results


def save_correlation_statistics(correlation_results, output_path):
    """Save raw and BH-adjusted correlation statistics for reproducibility."""
    scale_measure_by_panel = {
        "a": "Model parameters",
        "b": "Training compute",
        "c": "Model parameters",
        "d": "Training compute",
    }

    application_labels = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship allocation",
    }

    fieldnames = [
        "family",
        "panel",
        "scale_measure",
        "attribute",
        "application",
        "n_models",
        "pearson_r",
        "p_raw_two_sided",
        "p_bh_adjusted",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for key in sorted(correlation_results):
            panel_letter, attribute_type, application = key
            result = correlation_results[key]

            writer.writerow(
                {
                    "family": result["family"],
                    "panel": panel_letter,
                    "scale_measure": scale_measure_by_panel[panel_letter],
                    "attribute": attribute_type,
                    "application": application_labels[application],
                    "n_models": result["n"],
                    "pearson_r": result["r"],
                    "p_raw_two_sided": result["p_raw"],
                    "p_bh_adjusted": result["p_adjusted"],
                }
            )


def print_correlation_statistics(correlation_results):
    """Print a compact audit table of raw and adjusted P values."""
    print("\nPearson correlations with BH-adjusted P values")
    print("Adjustment families: societal (panels a+b) and contextual (panels c+d)")
    print("-" * 105)
    print(
        f"{'Panel':<7}{'Attribute':<22}{'Application':<25}"
        f"{'n':>4}{'r':>10}{'P raw':>14}{'P adjusted':>16}"
    )
    print("-" * 105)

    for key in sorted(correlation_results):
        panel_letter, attribute_type, application = key
        result = correlation_results[key]

        r_text = "NA" if not np.isfinite(result["r"]) else f"{result['r']:.4f}"
        raw_text = (
            "NA"
            if not np.isfinite(result["p_raw"])
            else f"{result['p_raw']:.6f}"
        )
        adj_text = (
            "NA"
            if not np.isfinite(result["p_adjusted"])
            else f"{result['p_adjusted']:.6f}"
        )

        print(
            f"{panel_letter:<7}{attribute_type:<22}{PANEL_TITLES[application]:<25}"
            f"{result['n']:>4}{r_text:>10}{raw_text:>14}{adj_text:>16}"
        )

    print("-" * 105)


def format_adjusted_p_value(p_value):
    """Format an adjusted P value for compact in-panel reporting."""
    if not np.isfinite(p_value):
        return r"\mathrm{NA}"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.2f}"


# -----------------------------------------------------------------------------
# Figure styling and plotting
# -----------------------------------------------------------------------------

def set_nature_style():
    plt.rcParams.update(
        {
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
        }
    )


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
    correlation_results,
    panel_letter,
    block_title,
    xlabel,
    ylabel=None,
):
    """
    Draw one 2 x 3 block.

    Rows are attributes, columns are applications, and the x-axis is log10
    transformed. Each subplot displays the Pearson r and the BH-adjusted,
    two-sided P value from its precomputed multiplicity family.
    """
    inner_gs = outer_spec.subgridspec(2, 3, wspace=0.12, hspace=1.2)
    axes = np.empty((2, 3), dtype=object)

    all_x = [model_to_x_value[m] for m in model_names if model_to_x_value[m] > 0]
    all_x_log = np.log10(all_x)
    x_min = min(all_x_log) - 0.08
    x_max = max(all_x_log) + 0.08

    for row_idx, attribute_type in enumerate(attribute_types):
        application_to_model_to_delta = (
            attribute_type_to_application_to_model_to_delta[attribute_type]
        )

        for col_idx, application in enumerate(APPLICATIONS):
            ax = fig.add_subplot(inner_gs[row_idx, col_idx])
            axes[row_idx, col_idx] = ax

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            xs_raw = np.asarray(
                [model_to_x_value[m] for m in model_names], dtype=float
            )
            ys = np.asarray(
                [
                    application_to_model_to_delta[application][m]
                    for m in model_names
                ],
                dtype=float,
            )

            valid_mask = np.isfinite(xs_raw) & np.isfinite(ys) & (xs_raw > 0)
            xs_valid = np.log10(xs_raw[valid_mask])
            ys_valid = ys[valid_mask]
            valid_models = [
                model for model, is_valid in zip(model_names, valid_mask) if is_valid
            ]

            for model, x_value, y_value in zip(
                valid_models, xs_valid, ys_valid
            ):
                ax.scatter(
                    x_value,
                    y_value,
                    s=98,
                    color=model_to_color[model],
                    edgecolors="none",
                    linewidths=0,
                    alpha=0.95,
                    zorder=3,
                    clip_on=False,
                )

            correlation = correlation_results[
                (panel_letter, attribute_type, application)
            ]
            r_value = correlation["r"]
            adjusted_p = correlation["p_adjusted"]

            if np.isfinite(r_value):
                xg, yg, yl, yu = fit_linear_with_ci(xs_valid, ys_valid)

                ax.plot(xg, yg, color="0.15", linewidth=1.35, zorder=2)
                ax.fill_between(
                    xg,
                    yl,
                    yu,
                    color="0.35",
                    alpha=0.12,
                    zorder=1,
                )

                p_text = format_adjusted_p_value(adjusted_p)
                corr_text = (
                    rf"$r={r_value:.2f}$"
                    + "\n"
                    # + rf"$P_{{\mathrm{{adj}}}}={p_text}$"
                    + rf"$P={p_text}$"
                )
            else:
                corr_text = (
                    r"$r=\mathrm{NA}$"
                    + "\n"
                    # + r"$P_{\mathrm{adj}}=\mathrm{NA}$"
                    + r"$P=\mathrm{NA}$"
                )

            ax.set_title(PANEL_TITLES[application], pad=8, fontsize=18)

            ax.text(
                0.03,
                0.96,
                corr_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=18,
                color="0.15",
            )

            ax.set_xlim(x_min, x_max)
            ax.set_axisbelow(True)

            ax.tick_params(
                axis="both",
                which="major",
                length=0.0,
                width=0.0,
                labelcolor="black",
                bottom=True,
                left=True,
                top=False,
                right=False,
            )

            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, pos: f"{value * 100:.0f}")
            )
            ax.xaxis.set_major_formatter(FuncFormatter(format_log_ticks))

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
        fontsize=18,
        fontweight="bold",
    )

    fig.text(
        block_x0 + 0.005,
        block_y1 + 0.060,
        block_title,
        ha="left",
        va="bottom",
        fontsize=18,
        fontweight="bold",
    )

    row_title_offset = 0.03
    for row_idx, attribute_type in enumerate(attribute_types):
        pos_row_left = axes[row_idx, 0].get_position()
        pos_row_right = axes[row_idx, 2].get_position()
        row_x_center = (pos_row_left.x0 + pos_row_right.x1) / 2
        row_y = pos_row_left.y1 + row_title_offset

        if attribute_type == "Gender Identity":
            attribute_type = "Gender identity"
        if attribute_type == "Sexual Orientation":
            attribute_type = "Sexual orientation"
        fig.text(
            row_x_center,
            row_y,
            attribute_type,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )

    fig.text(
        block_x_center,
        block_y0 - 0.043,
        xlabel,
        ha="center",
        va="top",
        fontsize=18,
    )

    if ylabel is not None:
        fig.text(
            block_x0 - 0.04,
            block_y_center,
            ylabel,
            ha="center",
            va="center",
            rotation=90,
            fontsize=18,
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
    Draw Figure 7 using two BH correction families:

      - societal minority bias: panels a and b (12 tests),
      - contextual minority bias: panels c and d (12 tests).
    """
    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    # Compute all raw correlations first, then perform BH correction within the
    # two pre-specified outcome families before any P values are plotted.
    societal_correlation_results = compute_correlation_family(
        attribute_type_to_application_to_model_to_delta=(
            societal_attribute_type_to_application_to_model_to_delta
        ),
        panel_to_x_values={
            "a": model_to_parameter_count,
            "b": model_to_training_compute,
        },
        attribute_types=societal_attribute_types,
        model_names=model_names,
        family_name="Societal minority bias (panels a and b)",
    )

    contextual_correlation_results = compute_correlation_family(
        attribute_type_to_application_to_model_to_delta=(
            contextual_attribute_type_to_application_to_model_to_delta
        ),
        panel_to_x_values={
            "c": model_to_parameter_count,
            "d": model_to_training_compute,
        },
        attribute_types=contextual_attribute_types,
        model_names=model_names,
        family_name="Contextual minority bias (panels c and d)",
    )

    correlation_results = {
        **societal_correlation_results,
        **contextual_correlation_results,
    }

    print_correlation_statistics(correlation_results)

    statistics_path = os.path.join(
        output_dir, "figure7_correlation_statistics_bh_adjusted.csv"
    )
    save_correlation_statistics(correlation_results, statistics_path)
    print(f"Saved: {statistics_path}")

    model_to_color = get_model_colors(model_names)
    fig = plt.figure(figsize=(15.0, 13.4))

    outer_gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.995,
        bottom=0.22,
        top=0.910,
        wspace=0.08,
        hspace=0.8,
    )

    # a. Societal minority bias versus model parameters
    draw_scale_block(
        fig=fig,
        outer_spec=outer_gs[0, 0],
        attribute_type_to_application_to_model_to_delta=(
            societal_attribute_type_to_application_to_model_to_delta
        ),
        model_to_x_value=model_to_parameter_count,
        attribute_types=societal_attribute_types,
        model_names=model_names,
        model_to_color=model_to_color,
        correlation_results=correlation_results,
        panel_letter="a",
        block_title="Societal minority bias vs. model parameters",
        xlabel=r"$\log_{10}$(Model parameters / $10^9$)",
        ylabel="Relative score difference (%)",
    )

    # b. Societal minority bias versus training compute
    draw_scale_block(
        fig=fig,
        outer_spec=outer_gs[0, 1],
        attribute_type_to_application_to_model_to_delta=(
            societal_attribute_type_to_application_to_model_to_delta
        ),
        model_to_x_value=model_to_training_compute,
        attribute_types=societal_attribute_types,
        model_names=model_names,
        model_to_color=model_to_color,
        correlation_results=correlation_results,
        panel_letter="b",
        block_title="Societal minority bias vs. training compute",
        xlabel=r"$\log_{10}$(Training compute / $10^{23}$ FLOPs)",
        ylabel=None,
    )

    # c. Contextual minority bias versus model parameters
    draw_scale_block(
        fig=fig,
        outer_spec=outer_gs[1, 0],
        attribute_type_to_application_to_model_to_delta=(
            contextual_attribute_type_to_application_to_model_to_delta
        ),
        model_to_x_value=model_to_parameter_count,
        attribute_types=contextual_attribute_types,
        model_names=model_names,
        model_to_color=model_to_color,
        correlation_results=correlation_results,
        panel_letter="c",
        block_title="Contextual minority bias vs. model parameters",
        xlabel=r"$\log_{10}$(Model parameters / $10^9$)",
        ylabel="Mean absolute candidate-level selection-rate difference (pp)",
    )

    # d. Contextual minority bias versus training compute
    draw_scale_block(
        fig=fig,
        outer_spec=outer_gs[1, 1],
        attribute_type_to_application_to_model_to_delta=(
            contextual_attribute_type_to_application_to_model_to_delta
        ),
        model_to_x_value=model_to_training_compute,
        attribute_types=contextual_attribute_types,
        model_names=model_names,
        model_to_color=model_to_color,
        correlation_results=correlation_results,
        panel_letter="d",
        block_title="Contextual minority bias vs. training compute",
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
            markerfacecolor=model_to_color[model],
            markeredgecolor="none",
            markersize=12.0,
            label=pretty_model_name(model),
            color=model_to_color[model],
        )
        for model in model_names
    ]

    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=4,
        frameon=False,
        fontsize=18,
        handletextpad=0.45,
        columnspacing=1.35,
    )

    for text, handle in zip(legend.get_texts(), legend_handles):
        text.set_color(handle.get_markerfacecolor())

    base = "scale_super_figure_societal_contextual_parameter_compute_nature_style"
    pdf_path = os.path.join(output_dir, base + ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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
        "msra-gpt-4o": 3.8e2,
        "gpt-oss-120b": 4.94e1,
        "Qwen3-235B-A22B-Instruct-2507": 4.752e1,
        "Qwen3-Next-80B-A3B-Instruct": 2.7e0,
        "GLM-4.5-Air": 1.656e1,
        "gemma-3-27b-it": 2.268e1,
        "Llama-3.3-70B-Instruct": 6.86498e1,
        "NVIDIA-Nemotron-Nano-12B-v2": 1.5192e1,
    }

    # -------------------------------------------------------------------------
    # Load contextual results
    # -------------------------------------------------------------------------
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

        contextual_attribute_type_to_application_to_model_to_delta[
            attribute_type
        ] = application_to_model_to_delta

    # -------------------------------------------------------------------------
    # Load societal results
    # -------------------------------------------------------------------------
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

        societal_attribute_type_to_application_to_model_to_delta[
            attribute_type
        ] = application_to_model_to_delta

    # -------------------------------------------------------------------------
    # Draw Figure 7
    # -------------------------------------------------------------------------
    draw_super_scale_figure(
        contextual_attribute_type_to_application_to_model_to_delta=(
            contextual_attribute_type_to_application_to_model_to_delta
        ),
        societal_attribute_type_to_application_to_model_to_delta=(
            societal_attribute_type_to_application_to_model_to_delta
        ),
        model_to_parameter_count=model_to_parameter_count,
        model_to_training_compute=model_to_training_compute,
        contextual_attribute_types=contextual_attribute_types,
        societal_attribute_types=societal_attribute_types,
        model_names=model_names,
        output_dir="outputs/parameter",
    )
