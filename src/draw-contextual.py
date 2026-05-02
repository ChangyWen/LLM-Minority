import json
import os
import math
import re
from collections import defaultdict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy.stats import chi2_contingency, norm
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D


# ============================================================
# Global style
# ============================================================

DELTA_COLOR = "#e05759"  # dark gray, more professional than pure blue
FONT_SIZE = 9.5
LABEL_SIZE = 8.0

def set_nature_style():
    """
    Compact, clean plotting style suitable for Nature-style multi-panel figures.
    """
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

        "axes.titlesize": 8.5,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "legend.fontsize": 8.5,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": 1.25,
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
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def p_to_stars(p):
    if p is None or math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def format_percent_tick(v, pos):
    return f"{v * 100:.0f}"


def format_signed_percent_tick(v, pos):
    return f"{v * 100:.0f}"


def get_attribute_style(attribute_type):
    """
    Consistent Nature-style color mapping.
    For Gender: Female - Male
    For Race: Black - White
    """
    if attribute_type == "Gender":
        return {
            "order": ["Female", "Male"],
            "colors": {
                "Female": "#D55E00",  # vermillion
                "Male": "#0072B2",    # blue
            },
            "delta_label": r"Selection-rate difference (Female - Male)",
            "right_ylabel": r"Selection-rate difference",
        }

    if attribute_type == "Race":
        return {
            "order": ["Black", "White"],
            "colors": {
                "Black": "#D55E00",   # vermillion
                "White": "#0072B2",   # blue
            },
            "delta_label": r"Selection-rate difference (Black - White)",
            "right_ylabel": r"Selection-rate difference",
        }

    raise ValueError(f"Unknown attribute_type: {attribute_type}")


# ============================================================
# Statistical helpers
# ============================================================

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0

    p = k / n
    denominator = 1 + (z ** 2) / n
    centre = p + (z ** 2) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))

    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return lower, upper


def chi2_test_same_attr_effect(attr_counts):
    table = []
    levels = sorted(attr_counts.keys())

    for c in levels:
        hit, total = attr_counts[c]
        miss = total - hit
        table.append([hit, miss])

    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, dof, levels


def cochran_armitage_trend(attr_counts):
    levels = sorted(attr_counts.keys())
    if not levels:
        return float("nan"), float("nan"), float("nan"), float("nan")

    scores = [float(c) for c in levels]
    hits = [attr_counts[c][0] for c in levels]
    totals = [attr_counts[c][1] for c in levels]

    N = sum(totals)
    X = sum(hits)

    if N == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    p_hat = X / N

    T = sum(w * (x - p_hat * n) for w, x, n in zip(scores, hits, totals))

    sum_nw = sum(n * w for n, w in zip(totals, scores))
    sum_nw2 = sum(n * (w ** 2) for n, w in zip(totals, scores))

    var_T = p_hat * (1 - p_hat) * (sum_nw2 - (sum_nw ** 2) / N)

    if var_T <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    z = T / math.sqrt(var_T)

    p_two = 2 * (1 - norm.cdf(abs(z)))
    p_inc = 1 - norm.cdf(z)
    p_dec = norm.cdf(z)

    return z, p_two, p_inc, p_dec


def trend_test_delta_counts(attr_counts_A, attr_counts_B):
    """
    Tests whether delta(c) = pA(c) - pB(c) changes systematically with c
    using a logistic regression with interaction term group*c.
    """

    levels = sorted(set(attr_counts_A) & set(attr_counts_B))
    rows = []

    for c in levels:
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]

        rows.append([1, c, hA, nA])
        rows.append([0, c, hB, nB])

    rows = np.array(rows, dtype=float)

    group = rows[:, 0]
    cvals = rows[:, 1]
    hits = rows[:, 2]
    totals = rows[:, 3]

    y = hits / totals
    w = totals

    X = np.column_stack([
        np.ones_like(group),
        group,
        cvals,
        group * cvals,
    ])

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=w,
    )
    result = model.fit()

    beta3 = result.params[3]
    se3 = result.bse[3]

    if se3 == 0 or np.isnan(se3):
        return float("nan"), float("nan"), float("nan"), float("nan")

    z = beta3 / se3

    p_two = 2 * (1 - norm.cdf(abs(z)))
    p_inc = 1 - norm.cdf(z)
    p_dec = norm.cdf(z)

    return z, p_two, p_inc, p_dec


# ============================================================
# Data computation
# ============================================================

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
    significance = {}

    attr_counts_A = None
    attr_counts_B = None

    for attr_value, attr_value_results in attr_value_to_results.items():
        results[attr_value] = {}
        significance[attr_value] = {}

        attr_counts = {}

        same_attr_count_to_count = dict(
            sorted(
                attr_value_results["same_attr_count_to_count"].items(),
                key=lambda x: x[0],
            )
        )

        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
            hit_rate = hit_count / count

            ci_low, ci_high = wilson_ci(hit_count, count)

            results[attr_value][same_attr_count] = {
                "hit_rate": hit_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

            attr_counts[same_attr_count] = (hit_count, count)

        chi2, p_global, dof, levels = chi2_test_same_attr_effect(attr_counts)
        significance[attr_value]["global_test_p_value"] = p_global

        z, p_two, p_inc, p_dec = cochran_armitage_trend(attr_counts)
        significance[attr_value]["p_value_two_sided"] = p_two
        significance[attr_value]["p_value_one_inc"] = p_inc
        significance[attr_value]["p_value_one_dec"] = p_dec

        if attr_value in ["Black", "Female"]:
            attr_counts_A = attr_counts
        else:
            attr_counts_B = attr_counts

    if attr_counts_A is None or attr_counts_B is None:
        return results, significance, n_trials

    z, p_two, p_inc, p_dec = trend_test_delta_counts(attr_counts_A, attr_counts_B)

    significance["delta"] = {
        "p_value_two_sided": p_two,
        "p_value_one_inc": p_inc,
        "p_value_one_dec": p_dec,
    }

    results["delta"] = {}

    for c in sorted(set(attr_counts_A) & set(attr_counts_B)):
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]

        pA = hA / nA
        pB = hB / nB

        delta = pA - pB

        ciA_low, ciA_high = wilson_ci(hA, nA)
        ciB_low, ciB_high = wilson_ci(hB, nB)

        ci_low = ciA_low - ciB_high
        ci_high = ciA_high - ciB_low

        results["delta"][c] = {
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return results, significance, n_trials


# ============================================================
# Panel plotting
# ============================================================

def plot_model_panel(
    ax_main,
    attribute_type,
    all_results,
    significance,
    model_name,
    show_left_ticks=True,
    show_right_ticks=True,
):
    """
    Draw one model panel.

    Left y-axis:
        selection rates for the two attribute values.

    Right y-axis:
        delta in selection rate.
    """

    attr_style = get_attribute_style(attribute_type)
    attribute_order = attr_style["order"]
    attribute_colors = attr_style["colors"]

    all_barlines = []
    xticks = []

    # ------------------------------------------------------------
    # Main selection-rate curves
    # ------------------------------------------------------------
    for attribute_value in attribute_order:
        if attribute_value not in all_results:
            continue

        res = all_results[attribute_value]

        xs = sorted(res.keys())
        xticks = xs

        ys = [res[x]["hit_rate"] for x in xs]
        lower_err = [res[x]["hit_rate"] - res[x]["ci_low"] for x in xs]
        upper_err = [res[x]["ci_high"] - res[x]["hit_rate"] for x in xs]

        color = attribute_colors[attribute_value]

        line, caplines, barlines = ax_main.errorbar(
            xs,
            ys,
            yerr=[lower_err, upper_err],
            marker="o",
            markersize=4.0,
            linewidth=1.15,
            linestyle="-",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.9,
            capsize=2.0,
            capthick=0.75,
            elinewidth=0.75,
            zorder=3,
        )

        all_barlines.extend(barlines)

    for bar in all_barlines:
        bar.set_linestyle("-")
        bar.set_linewidth(0.75)

    if len(xticks) > 0:
        ax_main.set_xticks(
            xticks,
            labels=[f"{(c + 1) / len(xticks) * 100:.0f}" for c in xticks],
        )
        ax_main.set_xlim(-0.12, len(xticks) - 1 + 0.12)

    ax_main.set_axisbelow(True)

    ax_main.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_main.yaxis.set_major_formatter(FuncFormatter(format_percent_tick))

    ax_main.tick_params(
        axis="x",
        direction="out",
        length=3.0,
        width=0.7,
        color="black",
        labelcolor="black",
        labelsize=LABEL_SIZE,   # x-axis tick-number size
        bottom=True,
        labelbottom=True,
    )

    ax_main.tick_params(
        axis="y",
        direction="out",
        length=3.0,
        width=0.7,
        color="black",
        labelcolor="black",
        labelsize=LABEL_SIZE,   # main y-axis tick-number size
        left=True,
        labelleft=True,
    )
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    for spine in ["left", "bottom"]:
        ax_main.spines[spine].set_visible(True)
        ax_main.spines[spine].set_linewidth(0.7)
        ax_main.spines[spine].set_color("0.15")

    # ------------------------------------------------------------
    # Delta curve on right y-axis
    # ------------------------------------------------------------
    if "delta" in all_results:
        ax_delta = ax_main.twinx()

        delta_res = all_results["delta"]

        xs_delta = sorted(delta_res.keys())
        ys_delta = [delta_res[c]["delta"] for c in xs_delta]

        lower_err_delta = [
            ys_delta[i] - delta_res[c]["ci_low"]
            for i, c in enumerate(xs_delta)
        ]
        upper_err_delta = [
            delta_res[c]["ci_high"] - ys_delta[i]
            for i, c in enumerate(xs_delta)
        ]

        ax_delta.errorbar(
            xs_delta,
            ys_delta,
            yerr=[lower_err_delta, upper_err_delta],
            marker="^",
            markersize=4.0,
            linestyle="--",
            linewidth=1.10,
            color=DELTA_COLOR,
            markerfacecolor=DELTA_COLOR,
            markeredgecolor=DELTA_COLOR,
            markeredgewidth=0.8,
            capsize=2.0,
            capthick=0.75,
            elinewidth=0.75,
            zorder=4,
        )

        ax_delta.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax_delta.yaxis.set_major_formatter(FuncFormatter(format_signed_percent_tick))

        ax_delta.tick_params(
            axis="y",
            which="major",
            direction="out",
            length=3.0,
            width=0.7,
            color="black",
            labelcolor="black",
            labelsize=LABEL_SIZE,   # twin y-axis tick-number size
            right=True,
            labelright=True,
            left=False,
            labelleft=False,
        )

        ax_delta.spines["top"].set_visible(False)
        ax_delta.spines["right"].set_visible(True)
        ax_delta.spines["right"].set_color("black")
        ax_delta.spines["right"].set_linewidth(0.7)
        ax_delta.spines["left"].set_visible(False)
        ax_delta.spines["bottom"].set_visible(False)


# ============================================================
# Big figure drawing
# ============================================================

def draw_attribute_big_figure(
    attribute_type,
    model_names,
    application_to_pool_count,
    resume_count=5,
    max_n_trials=1000000,
    output_dir="outputs/contextual",
):
    """
    Draw one big figure for one attribute.

    Layout:
        a. Hiring
        b. Loan approval
        c. Scholarship application

    Each application block contains eight model panels arranged as 2 x 4.
    """

    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    applications = ["hiring", "loan", "edu"]

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship application",
    }

    attr_style = get_attribute_style(attribute_type)

    fig = plt.figure(figsize=(9.5, 10))

    outer_gs = fig.add_gridspec(
        3,
        1,
        left=0.125,    # more room for the main y-axis label
        right=0.875,   # more room for the twin y-axis label
        bottom=0.090,
        top=0.940,
        hspace=0.3,
    )

    all_axes = {}

    for app_idx, application in enumerate(applications):
        inner_gs = outer_gs[app_idx].subgridspec(
            2,
            4,
            wspace=0.32,    # larger horizontal gap between model panels
            hspace=0.36,
        )

        axes = np.empty((2, 4), dtype=object)
        all_axes[application] = axes

        pool_count = application_to_pool_count[application]

        for idx, model_name in enumerate(model_names):
            row = idx // 4
            col = idx % 4

            ax_main = fig.add_subplot(inner_gs[row, col])
            axes[row, col] = ax_main

            ax_main.set_title(
                pretty_model_name(model_name),
                loc="center",
                pad=4,
                fontsize=FONT_SIZE,
            )

            file_name = (
                f"outputs/{application}/contextual/"
                f"{attribute_type}/{model_name}_{resume_count}_{pool_count}.jsonl"
            )

            if not os.path.exists(file_name):
                print(f"[Warning] File not found, skipping: {file_name}")
                ax_main.set_visible(False)
                continue

            results, significance, n_trials = compute_results(
                file_name=file_name,
                attribute_type=attribute_type,
                max_n_trials=max_n_trials,
            )

            plot_model_panel(
                ax_main=ax_main,
                attribute_type=attribute_type,
                all_results=results,
                significance=significance,
                model_name=model_name,
                show_left_ticks=(col == 0),
                show_right_ticks=(col == 3),
            )

    # ------------------------------------------------------------
    # Shared labels
    # ------------------------------------------------------------
    fig.supxlabel(
        "Proportion of focal group in candidate pool (%)",
        fontsize=FONT_SIZE,
        y=0.047,
    )

    fig.supylabel(
        "Selection rate (%)",
        fontsize=FONT_SIZE,
        x=0.08,   # farther from the left edge of the panels
    )

    fig.text(
        0.915,     # farther from the right edge of the panels
        0.515,
        attr_style["right_ylabel"] + " (pp)",
        va="center",
        ha="center",
        rotation=270,
        fontsize=FONT_SIZE,
        color="black",
    )

    # ------------------------------------------------------------
    # Application row titles and Nature-style letters
    # ------------------------------------------------------------
    for application in applications:
        axes = all_axes[application]

        pos_left = axes[0, 0].get_position()
        pos_right = axes[0, 3].get_position()

        x0 = pos_left.x0
        x1 = pos_right.x1
        y1 = pos_left.y1

        fig.text(
            (x0 + x1) / 2,
            y1 + 0.022,
            application_title_map[application],
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
            fontweight="bold",
        )

    # ------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------
    legend_handles = []

    for attribute_value in attr_style["order"]:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=attr_style["colors"][attribute_value],
                marker="o",
                markerfacecolor=attr_style["colors"][attribute_value],
                markeredgecolor=attr_style["colors"][attribute_value],
                markeredgewidth=0.9,
                linewidth=1.20,
                markersize=4.0,
                label=f"Selection rate ({attribute_value})",
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=DELTA_COLOR,
            marker="^",
            markerfacecolor=DELTA_COLOR,
            markeredgecolor=DELTA_COLOR,
            linewidth=1.20,
            linestyle="--",
            markersize=4.0,
            label=attr_style["delta_label"],
        )
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.010),
        ncol=3,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.4,
        handletextpad=0.55,
        fontsize=FONT_SIZE,   # legend text size
    )

    base = f"{safe_slug(attribute_type)}_all_applications_contextual_nature_style"
    pdf_path = os.path.join(output_dir, base + ".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    max_n_trials = 1000000

    model_names_order = [
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

    application_to_pool_count = {
        "edu": 500,
        "hiring": 200,
        "loan": 500,
    }

    for attribute_type in attribute_types:
        draw_attribute_big_figure(
            attribute_type=attribute_type,
            model_names=model_names_order,
            application_to_pool_count=application_to_pool_count,
            resume_count=5,
            max_n_trials=max_n_trials,
            output_dir="outputs/contextual",
        )