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

DELTA_COLOR = "#ffa21c"  # dark gray, more professional than pure blue
RANDOM_RATE_COLOR = "0.35"

FONT_SIZE = 9.5
LABEL_SIZE = 8.0
CI_ALPHA = 0.35

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
        return "ns"


def format_percent_tick(v, pos):
    return f"{v * 100:.0f}"


def format_signed_percent_tick(v, pos):
    return f"{v * 100:.0f}"


def get_attribute_style(attribute_type):
    """
    Consistent Nature-style color mapping.

    Gender:
        x-axis = proportion of Female candidates in the pool.

    Race:
        x-axis = proportion of Black candidates in the pool.
    """
    if attribute_type == "Gender":
        return {
            "order": ["Female", "Male"],
            "focal": "Female",
            "reference": "Male",
            "colors": {
                "Female": "#ff2d86",
                "Male": "#7550ff",
            },
            "x_label": "Proportion of female in candidate pool (%)",
        }

    if attribute_type == "Race":
        return {
            "order": ["Black", "White"],
            "focal": "Black",
            "reference": "White",
            "colors": {
                "Black": "#ff2d86",
                "White": "#7550ff",
            },
            "x_label": "Proportion of Black in candidate pool (%)",
        }

    raise ValueError(f"Unknown attribute_type: {attribute_type}")


def trend_label_from_pvalues(p_inc, p_dec, show_ns=True):
    """
    Return a compact one-sided trend label.

    Examples:
        ↑*
        ↓**
        ↑***
        ns

    Important:
        If the trend is not statistically significant, return "ns"
        without any arrow.
    """
    if p_inc is None or p_dec is None:
        return "ns" if show_ns else ""

    if math.isnan(p_inc) or math.isnan(p_dec):
        return "ns" if show_ns else ""

    # Increasing trend is stronger
    if p_inc <= p_dec:
        if p_inc < 0.001:
            return "↑***"
        elif p_inc < 0.01:
            return "↑**"
        elif p_inc < 0.05:
            return "↑*"
        else:
            return "ns" if show_ns else ""

    # Decreasing trend is stronger
    else:
        if p_dec < 0.001:
            return "↓***"
        elif p_dec < 0.01:
            return "↓**"
        elif p_dec < 0.05:
            return "↓*"
        else:
            return "ns" if show_ns else ""


def make_trend_legend_handles(attribute_type, significance):
    """
    Build a compact in-panel legend showing the direction and significance
    of each group-specific selection-rate trend.

    The arrows always refer to the plotted x-axis:
        Gender: increasing proportion of Female candidates.
        Race:   increasing proportion of Black candidates.

    Examples:
        ↑*** means the target group's selection rate significantly increases
              as the proportion of Female/Black candidates increases.

        ↓*** means the target group's selection rate significantly decreases
              as the proportion of Female/Black candidates increases.
    """
    attr_style = get_attribute_style(attribute_type)
    handles = []

    for attribute_value in attr_style["order"]:
        color = attr_style["colors"][attribute_value]

        p_inc = significance.get(attribute_value, {}).get(
            "p_value_one_inc",
            float("nan"),
        )
        p_dec = significance.get(attribute_value, {}).get(
            "p_value_one_dec",
            float("nan"),
        )

        trend_text = trend_label_from_pvalues(
            p_inc=p_inc,
            p_dec=p_dec,
            show_ns=True,
        )

        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="^" if attribute_value in ["Female", "Black"] else "o",
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.9,
                linewidth=1.15,
                linestyle="-",
                markersize=4.0,
                label=trend_text,
            )
        )

    return handles


def expand_lower_ylim(ax, lower_frac=0.22, upper_frac=0.04):
    """
    Add extra empty space below the current y-range.
    Useful when an in-panel legend is placed near the bottom.
    """
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    ax.set_ylim(
        ymin - lower_frac * yrange,
        ymax + upper_frac * yrange,
    )


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


def two_proportion_z_test(h1, n1, h2, n2):
    """
    Two-sided z-test for equality of two proportions.

    H0: p1 = p2
    H1: p1 != p2
    """
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    p1 = h1 / n1
    p2 = h2 / n2
    p_pool = (h1 + h2) / (n1 + n2)

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        if p1 == p2:
            return float("nan"), 1.0
        return float("nan"), 0.0

    z = (p1 - p2) / se
    p_two = 2 * (1 - norm.cdf(abs(z)))

    return z, p_two


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


def add_pairwise_significance_stars(
    ax,
    attribute_type,
    plotted_results,
    significance,
    fontsize=7.2,
):
    """
    Add per-x significance stars for pairwise tests between the two
    attribute-specific selection rates.

    The x-axis is now the focal-group proportion:
        Gender: proportion of Female candidates
        Race: proportion of Black candidates
    """
    pairwise = significance.get("pairwise", {})
    if not pairwise:
        return

    attr_style = get_attribute_style(attribute_type)
    attribute_order = attr_style["order"]

    ymin, ymax = ax.get_ylim()
    yrange = max(ymax - ymin, 1e-12)

    star_offset = 0.055 * yrange
    extra_headroom = 0.080 * yrange

    max_star_y = ymax

    for x_percent in sorted(pairwise.keys()):
        p_value = pairwise[x_percent].get("p_value_two_sided", float("nan"))
        stars = p_to_stars(p_value)

        if not stars:
            continue

        y_candidates = []

        for attribute_value in attribute_order:
            if (
                attribute_value in plotted_results
                and x_percent in plotted_results[attribute_value]
            ):
                y_candidates.append(
                    plotted_results[attribute_value][x_percent]["ci_high"]
                )

        if not y_candidates:
            continue

        y_star = max(y_candidates) + star_offset

        ax.text(
            x_percent,
            y_star,
            stars,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="0.10",
            clip_on=False,
            zorder=8,
        )

        max_star_y = max(max_star_y, y_star + extra_headroom)

    if max_star_y > ymax:
        ax.set_ylim(ymin, max_star_y)


def build_focal_axis_counts(attr_counts, attribute_value, attr_style, pool_size):
    """
    Re-index target-group counts by the plotted x-axis.

    New plotted x-axis:
        Gender: proportion of Female candidates in the pool.
        Race:   proportion of Black candidates in the pool.

    For the focal group, e.g., Female/Black:
        raw same_attr_count increases as the plotted x-axis increases.

    For the reference group, e.g., Male/White:
        raw same_attr_count decreases as the plotted x-axis increases.
        Therefore, we transform the count to the corresponding focal-group
        proportion before applying the trend test.

    Returns:
        dict mapping x_percent -> (hit_count, total_count)
    """
    focal_axis_counts = {}

    for same_attr_count, count_pair in attr_counts.items():
        group_count = same_attr_count + 1

        if attribute_value == attr_style["focal"]:
            focal_count = group_count
        else:
            focal_count = pool_size - group_count

        x_percent = 100.0 * focal_count / pool_size
        focal_axis_counts[x_percent] = count_pair

    return dict(sorted(focal_axis_counts.items(), key=lambda item: item[0]))


# ============================================================
# Data computation
# ============================================================

def compute_results(file_name, attribute_type, max_n_trials=1000000, pool_size=None):
    attr_value_to_results = defaultdict(lambda: {
        "same_attr_count_to_count": defaultdict(int),
        "same_attr_count_to_hit_count": defaultdict(int),
    })

    n_trials = 0
    observed_max_same_attr_count = 0

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

            if pool_size is None:
                pool_size = len(attributes)

            for inner_idx, attr_value in enumerate(attributes):
                same_attr_count = attributes.count(attr_value) - 1
                observed_max_same_attr_count = max(
                    observed_max_same_attr_count,
                    same_attr_count,
                )

                attr_value_to_results[attr_value]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results[attr_value]["same_attr_count_to_hit_count"][same_attr_count] += (
                    1 if inner_idx == suggested_candidate_id else 0
                )

    if pool_size is None:
        pool_size = observed_max_same_attr_count + 1

    attr_style = get_attribute_style(attribute_type)
    focal_attr = attr_style["focal"]
    reference_attr = attr_style["reference"]

    results = {}
    significance = {}
    attr_value_to_counts = {}

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
                "hit_count": hit_count,
                "count": count,
            }

            attr_counts[same_attr_count] = (hit_count, count)

        attr_value_to_counts[attr_value] = attr_counts

        if len(attr_counts) >= 2:
            # Omnibus test: whether the target group's selection rate differs
            # across pool-composition settings.
            chi2, p_global, dof, levels = chi2_test_same_attr_effect(attr_counts)
            significance[attr_value]["global_test_p_value"] = p_global

            # Directional trend test on the plotted x-axis:
            #   Gender: Female proportion increases.
            #   Race:   Black proportion increases.
            focal_axis_counts = build_focal_axis_counts(
                attr_counts=attr_counts,
                attribute_value=attr_value,
                attr_style=attr_style,
                pool_size=pool_size,
            )

            z, p_two, p_inc, p_dec = cochran_armitage_trend(focal_axis_counts)

            significance[attr_value]["trend_z"] = z
            significance[attr_value]["p_value_two_sided"] = p_two
            significance[attr_value]["p_value_one_inc"] = p_inc
            significance[attr_value]["p_value_one_dec"] = p_dec
        else:
            significance[attr_value]["global_test_p_value"] = float("nan")
            significance[attr_value]["trend_z"] = float("nan")
            significance[attr_value]["p_value_two_sided"] = float("nan")
            significance[attr_value]["p_value_one_inc"] = float("nan")
            significance[attr_value]["p_value_one_dec"] = float("nan")

    # ------------------------------------------------------------
    # Pairwise tests at each focal-group proportion.
    #
    # Gender:
    #   compare Female when #Female = k
    #   with Male when #Female = k, i.e., #Male = pool_size - k.
    #
    # Race:
    #   compare Black when #Black = k
    #   with White when #Black = k, i.e., #White = pool_size - k.
    # ------------------------------------------------------------
    significance["pairwise"] = {}

    focal_counts = attr_value_to_counts.get(focal_attr, {})
    reference_counts = attr_value_to_counts.get(reference_attr, {})

    for focal_count in range(1, pool_size):
        focal_same_attr_count = focal_count - 1

        reference_count = pool_size - focal_count
        reference_same_attr_count = reference_count - 1

        if (
            focal_same_attr_count not in focal_counts
            or reference_same_attr_count not in reference_counts
        ):
            continue

        h_focal, n_focal = focal_counts[focal_same_attr_count]
        h_ref, n_ref = reference_counts[reference_same_attr_count]

        z_pair, p_pair = two_proportion_z_test(
            h_focal,
            n_focal,
            h_ref,
            n_ref,
        )

        x_percent = 100.0 * focal_count / pool_size

        significance["pairwise"][x_percent] = {
            "z": z_pair,
            "p_value_two_sided": p_pair,
        }

    return results, significance, n_trials

# ============================================================
# Panel plotting
# ============================================================

def _x_percent_for_attribute(attribute_value, same_attr_count, attr_style, pool_size):
    """
    Convert the original same-attribute count into the new x-axis value.

    New x-axis:
        Gender: proportion of Female candidates
        Race: proportion of Black candidates

    For the focal group itself, e.g., Female/Black:
        x = #focal / pool_size

    For the reference group, e.g., Male/White:
        x = #focal / pool_size = 1 - #reference / pool_size
    """
    group_count = same_attr_count + 1

    if attribute_value == attr_style["focal"]:
        focal_count = group_count
    else:
        focal_count = pool_size - group_count

    return 100.0 * focal_count / pool_size


def plot_model_panel(
    ax_main,
    attribute_type,
    all_results,
    significance,
    model_name,
    pool_size,
    show_left_ticks=True,
    show_right_ticks=True,
    application=None,
):
    """
    Draw one model panel.

    Revised design:
        - Show group-specific selection-rate curves.
        - Keep the uniform-random selection-rate baseline.
        - Do not show the selection-rate difference curve.
        - Add an in-panel trend legend for the two group-specific curves.

    Trend arrows:
        Gender: trends are tested as Female proportion increases.
        Race:   trends are tested as Black proportion increases.
    """
    attr_style = get_attribute_style(attribute_type)
    attribute_order = attr_style["order"]
    attribute_colors = attr_style["colors"]

    all_barlines = []
    plotted_results = defaultdict(dict)

    # ------------------------------------------------------------
    # Main selection-rate curves
    # ------------------------------------------------------------
    for attribute_value in attribute_order:
        if attribute_value not in all_results:
            continue

        res = all_results[attribute_value]
        color = attribute_colors[attribute_value]

        rows = []

        for same_attr_count in sorted(res.keys()):
            x_percent = _x_percent_for_attribute(
                attribute_value=attribute_value,
                same_attr_count=same_attr_count,
                attr_style=attr_style,
                pool_size=pool_size,
            )

            if x_percent < 0 or x_percent > 100:
                continue

            row = {
                "x_percent": x_percent,
                "hit_rate": res[same_attr_count]["hit_rate"],
                "ci_low": res[same_attr_count]["ci_low"],
                "ci_high": res[same_attr_count]["ci_high"],
            }

            rows.append(row)
            plotted_results[attribute_value][x_percent] = row

        rows = sorted(rows, key=lambda x: x["x_percent"])

        if not rows:
            continue

        xs = [r["x_percent"] for r in rows]
        ys = [r["hit_rate"] for r in rows]
        lower_err = [r["hit_rate"] - r["ci_low"] for r in rows]
        upper_err = [r["ci_high"] - r["hit_rate"] for r in rows]

        line, caplines, barlines = ax_main.errorbar(
            xs,
            ys,
            yerr=[lower_err, upper_err],
            marker="^" if attribute_value in ["Female", "Black"] else "o",
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

        for cap in caplines:
            cap.set_alpha(CI_ALPHA)

        for bar in barlines:
            bar.set_alpha(CI_ALPHA)

        all_barlines.extend(barlines)

    for bar in all_barlines:
        bar.set_linestyle("-")
        bar.set_linewidth(0.75)

    # ------------------------------------------------------------
    # Uniform-random selection-rate baseline
    # ------------------------------------------------------------
    uniform_random_rate = 1.0 / pool_size

    ax_main.axhline(
        uniform_random_rate,
        color=RANDOM_RATE_COLOR,
        linestyle="--",
        linewidth=0.95,
        alpha=0.85,
        zorder=1,
    )

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------
    x_ticks = [100.0 * k / pool_size for k in range(0, pool_size + 1)]
    x_step = 100.0 / pool_size

    ax_main.set_xticks(
        x_ticks,
        labels=[f"{x:.0f}" for x in x_ticks],
    )
    ax_main.set_xlim(-0.35 * x_step, 100 + 0.35 * x_step)

    ax_main.set_axisbelow(True)

    ax_main.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_main.yaxis.set_major_formatter(FuncFormatter(format_percent_tick))

    # ------------------------------------------------------------
    # Pairwise significance at each focal-group proportion
    # ------------------------------------------------------------
    add_pairwise_significance_stars(
        ax=ax_main,
        attribute_type=attribute_type,
        plotted_results=plotted_results,
        significance=significance,
        fontsize=7.2,
    )

    # ------------------------------------------------------------
    # Add lower space for the in-panel trend legend
    # ------------------------------------------------------------
    expand_lower_ylim(
        ax_main,
        lower_frac=0.20,
        upper_frac=0.04,
    )

    # ------------------------------------------------------------
    # Per-panel trend legend
    # ------------------------------------------------------------
    trend_handles = make_trend_legend_handles(
        attribute_type=attribute_type,
        significance=significance,
    )

    trend_legend = ax_main.legend(
        handles=trend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.75",
        fontsize=7.4,
        title_fontsize=7.4,
        borderpad=0.25,
        labelspacing=0.25,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.65,
        borderaxespad=0.2,
    )

    trend_legend.get_frame().set_linewidth(0.6)

    # ------------------------------------------------------------
    # Tick and spine styling
    # ------------------------------------------------------------
    ax_main.tick_params(
        axis="x",
        length=0.0,
        width=0.0,
        color="black",
        labelcolor="black",
        labelsize=LABEL_SIZE,
        bottom=True,
        labelbottom=True,
    )

    ax_main.tick_params(
        axis="y",
        length=0.0,
        width=0.0,
        color="black",
        labelcolor="black",
        labelsize=LABEL_SIZE,
        left=True,
        labelleft=True,
    )

    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    for spine in ["left", "bottom"]:
        ax_main.spines[spine].set_visible(True)
        ax_main.spines[spine].set_linewidth(0.7)
        ax_main.spines[spine].set_color("0.15")


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
        "edu": "Scholarship allocation",
    }

    attribute_panel_title_map = {
        "Gender": ("a", "Evaluation results by gender"),
        "Race": ("b", "Evaluation results by race"),
    }

    attr_style = get_attribute_style(attribute_type)

    fig = plt.figure(figsize=(9.5, 10))

    panel_letter, panel_title = attribute_panel_title_map[attribute_type]

    fig.text(
        0.07,
        0.970,
        panel_letter,
        ha="left",
        va="top",
        fontsize=FONT_SIZE + 1.0,
        fontweight="bold",
    )

    fig.text(
        0.07 + 0.025,
        0.970,
        panel_title,
        ha="left",
        va="top",
        fontsize=FONT_SIZE + 1.0,
        fontweight="bold",
    )

    outer_gs = fig.add_gridspec(
        3,
        1,
        left=0.110,
        right=0.965,
        bottom=0.090,
        top=0.915,
        hspace=0.3,
    )

    all_axes = {}

    for app_idx, application in enumerate(applications):
        inner_gs = outer_gs[app_idx].subgridspec(
            2,
            4,
            wspace=0.16,
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
                pool_size=resume_count,
            )

            plot_model_panel(
                ax_main=ax_main,
                attribute_type=attribute_type,
                all_results=results,
                significance=significance,
                model_name=model_name,
                pool_size=resume_count,
                show_left_ticks=(col == 0),
                show_right_ticks=False,
                application=application,
            )

    # ------------------------------------------------------------
    # Shared labels
    # ------------------------------------------------------------
    fig.supxlabel(
        attr_style["x_label"],
        fontsize=FONT_SIZE,
        y=0.047,
    )

    fig.supylabel(
        "Candidate-level selection rate (%)",
        fontsize=FONT_SIZE,
        x=0.070,
    )

    # ------------------------------------------------------------
    # Application row titles
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
                marker="^" if attribute_value in ["Female", "Black"] else "o",
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
            color=RANDOM_RATE_COLOR,
            linestyle="--",
            linewidth=1.20,
            label="Uniform-random selection rate (20%)",
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
        fontsize=FONT_SIZE,
    )

    base = f"{safe_slug(attribute_type)}_all_applications_contextual_selection_rate_random_baseline"
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