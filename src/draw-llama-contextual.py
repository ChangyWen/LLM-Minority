import json
import sys
from collections import defaultdict
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
from statistics import NormalDist

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

def se_diff_of_props(hA, nA, hB, nB):
    pA = hA / nA
    pB = hB / nB
    return math.sqrt(pA*(1-pA)/nA + pB*(1-pB)/nB)

def abs_diff(h1, n1, h2, n2):
    return abs((h1 / n1) - (h2 / n2))

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


def two_sided_p_for_delta_difference(d1, se1, d2, se2):
    """
    Two-sided normal-approximation test for:

        H0: delta_1 = delta_2
        H1: delta_1 != delta_2

    Here delta is the normalized absolute selection-rate difference.
    """
    se = math.sqrt(se1 ** 2 + se2 ** 2)

    if se == 0:
        return 1.0 if d1 == d2 else 0.0

    z = (d1 - d2) / se
    p = 2.0 * (1.0 - NormalDist().cdf(abs(z)))
    return p


def compute_model_pair_pvalues_by_ratio(
    delta_by_ratio_1,
    delta_by_ratio_2,
    ratios=("20%", "40%", "60%", "80%"),
):
    """
    Compute per-ratio two-sided p-values comparing two models.

    H1: the two models have different contextual bias.
    """
    pvalues = {}

    for r in ratios:
        if r not in delta_by_ratio_1 or r not in delta_by_ratio_2:
            continue

        d1 = float(delta_by_ratio_1[r]["delta"])
        d2 = float(delta_by_ratio_2[r]["delta"])

        se1 = float(delta_by_ratio_1[r]["se"])
        se2 = float(delta_by_ratio_2[r]["se"])

        pvalues[r] = two_sided_p_for_delta_difference(d1, se1, d2, se2)

    return pvalues

def compute_results(file_name, context_size, max_n_trials=1000000):
    """
    Returns: dict like { "20%": {"delta":..., "ci_low":..., "ci_high":...}, ...}
    delta is normalized by random rate (1/context_size), consistent with your original code.
    """
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

    attr_counts_A, attr_counts_B = None, None
    for attr_value, attr_value_results in attr_value_to_results.items():
        attr_counts = {}
        same_attr_count_to_count = dict(
            sorted(attr_value_results["same_attr_count_to_count"].items(), key=lambda x: x[0])
        )
        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
            attr_counts[same_attr_count] = (hit_count, count)

        # Keep unchanged from your code
        if attr_value == "Black" or attr_value == "Female":
            attr_counts_A = attr_counts
        else:
            attr_counts_B = attr_counts

    results_delta = {}
    random_selection_rate = 1 / context_size
    for c in sorted(set(attr_counts_A) & set(attr_counts_B)):
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]
        pA = hA / nA
        pB = hB / nB

        ciA_low, ciA_high = wilson_ci(hA, nA)
        ciB_low, ciB_high = wilson_ci(hB, nB)

        raw_se = se_diff_of_props(hA, nA, hB, nB)

        if pA > pB:
            delta = pA - pB
            ci_low = ciA_low - ciB_high
            ci_high = ciA_high - ciB_low
        else:
            delta = pB - pA
            ci_low = ciB_low - ciA_high
            ci_high = ciB_high - ciA_low

        ratio = (c + 1) / context_size
        ratio_str = f"{ratio * 100:.0f}%"
        if ratio_str in ["20%", "40%", "60%", "80%"]:
            results_delta[ratio_str] = {
                "random_selection_rate": random_selection_rate,
                "delta": delta / random_selection_rate,
                "ci_low": ci_low / random_selection_rate,
                "ci_high": ci_high / random_selection_rate,
                "se": raw_se / random_selection_rate,
            }

    return results_delta


# ---------------------------------------------------------------------
# Nature-style plotting helpers
# ---------------------------------------------------------------------
def set_nature_style():
    """
    Compact, clean plotting style suitable for Nature-style multi-panel figures.
    """
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
    """
    Shorter display names for compact multi-panel figures.
    """
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


def draw_all_applications_in_one_figure(
    app_attr_model_to_delta,
    applications,
    attribute_types,
    model_names,
    output_dir="outputs/llama",
):
    """
    Draw one Nature-style figure for all applications.

    Layout:
        rows = applications
        columns = attribute types

    Statistical test:
        For each application, attribute, and contextual ratio,
        compare Llama-3.1-8B and Llama-3.1-8B-Instruct.

        H0: the two models have equal contextual bias
        H1: the two models have different contextual bias

    Stars denote two-sided p-values:
        * P < 0.05, ** P < 0.01, *** P < 0.001; ns, not significant.
    """

    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    ratio_strs = ["20%", "40%", "60%", "80%"]
    ratio_x = np.array([20, 40, 60, 80], dtype=float)

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship application",
    }

    model_style = {
        "Llama-3.1-8B": {
            "color": "#0072B2",
            "marker": "o",
            "label": "Llama-3.1-8B",
        },
        "Llama-3.1-8B-Instruct": {
            "color": "#D55E00",
            "marker": "s",
            "label": "Llama-3.1-8B-Instruct",
        },
    }

    fallback_colors = ["#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
    fallback_markers = ["^", "D", "v", "P"]

    for idx, model_name in enumerate(model_names):
        if model_name not in model_style:
            model_style[model_name] = {
                "color": fallback_colors[idx % len(fallback_colors)],
                "marker": fallback_markers[idx % len(fallback_markers)],
                "label": pretty_model_name(model_name),
            }

    fig, axes = plt.subplots(
        len(applications),
        len(attribute_types),
        figsize=(7.45, 7.75),
        sharex=False,
        sharey=False,
    )

    axes = np.asarray(axes)

    for row_idx, application in enumerate(applications):
        for col_idx, attribute_type in enumerate(attribute_types):
            ax = axes[row_idx, col_idx]

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.axhline(0, color="0.30", linewidth=0.7, zorder=1)

            panel_ci_upper_values = []
            panel_upper_by_ratio = {r: 0.0 for r in ratio_strs}

            # ------------------------------------------------------------
            # Plot model curves
            # ------------------------------------------------------------
            for model_name in model_names:
                delta_by_ratio = (
                    app_attr_model_to_delta
                    .get(application, {})
                    .get(attribute_type, {})
                    .get(model_name, None)
                )

                if not delta_by_ratio:
                    continue

                xs, ys, yerr_low, yerr_high = [], [], [], []

                for rx, rstr in zip(ratio_x, ratio_strs):
                    if rstr not in delta_by_ratio:
                        continue

                    d = delta_by_ratio[rstr]
                    y = float(d["delta"])

                    lo = max(0.0, float(d["ci_low"]))
                    hi = max(0.0, float(d["ci_high"]))

                    xs.append(rx)
                    ys.append(y)
                    yerr_low.append(max(0.0, y - lo))
                    yerr_high.append(max(0.0, hi - y))

                    panel_ci_upper_values.append(hi)
                    panel_upper_by_ratio[rstr] = max(panel_upper_by_ratio[rstr], hi)

                if len(xs) == 0:
                    continue

                yerr = np.vstack([yerr_low, yerr_high])
                style = model_style[model_name]

                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    fmt=style["marker"] + "-",
                    color=style["color"],
                    markerfacecolor="white",
                    markeredgecolor=style["color"],
                    markeredgewidth=1.0,
                    markersize=4.2,
                    linewidth=1.20,
                    elinewidth=0.80,
                    capsize=2.0,
                    capthick=0.80,
                    zorder=3,
                )

            # ------------------------------------------------------------
            # Statistical test: Llama-3.1-8B vs Llama-3.1-8B-Instruct
            # ------------------------------------------------------------
            pvalues_by_ratio = {}

            if len(model_names) == 2:
                model_1, model_2 = model_names

                delta_by_ratio_1 = (
                    app_attr_model_to_delta
                    .get(application, {})
                    .get(attribute_type, {})
                    .get(model_1, {})
                )
                delta_by_ratio_2 = (
                    app_attr_model_to_delta
                    .get(application, {})
                    .get(attribute_type, {})
                    .get(model_2, {})
                )

                pvalues_by_ratio = compute_model_pair_pvalues_by_ratio(
                    delta_by_ratio_1=delta_by_ratio_1,
                    delta_by_ratio_2=delta_by_ratio_2,
                    ratios=ratio_strs,
                )

                for rstr, p in pvalues_by_ratio.items():
                    print(
                        f"{application} | {attribute_type} | {rstr}: "
                        f"{model_1} vs {model_2}, two-sided P={p:.4g}"
                    )

            # ------------------------------------------------------------
            # Panel-specific y-limit with headroom for significance stars
            # ------------------------------------------------------------
            if len(panel_ci_upper_values) == 0:
                panel_data_top = 1.0
            else:
                panel_data_top = max(panel_ci_upper_values)
                panel_data_top = max(panel_data_top, 0.05)

            star_positions = {}
            star_offset = 0.075 * panel_data_top

            for rx, rstr in zip(ratio_x, ratio_strs):
                if rstr not in pvalues_by_ratio:
                    continue

                stars = p_to_stars(pvalues_by_ratio[rstr])
                if stars:
                    y_star = panel_upper_by_ratio.get(rstr, 0.0) + star_offset
                    star_positions[rstr] = y_star

            if star_positions:
                ymax_i = max(
                    panel_data_top * 1.20,
                    max(star_positions.values()) + 0.15 * panel_data_top,
                )
            else:
                ymax_i = panel_data_top * 1.20

            ax.set_ylim(0.0, ymax_i)

            # ------------------------------------------------------------
            # Draw significance stars
            # ------------------------------------------------------------
            for rx, rstr in zip(ratio_x, ratio_strs):
                if rstr not in star_positions:
                    continue

                stars = p_to_stars(pvalues_by_ratio[rstr])

                ax.text(
                    rx,
                    star_positions[rstr],
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    fontweight="bold",
                    color="0.10",
                    zorder=4,
                )

            ax.set_xlim(12, 88)
            ax.set_xticks(ratio_x)
            ax.set_xticklabels(["20", "40", "60", "80"])

            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, pos: f"{v * 100:.0f}")
            )

            ax.tick_params(
                axis="both",
                direction="out",
                length=3.0,
                width=0.7,
                color="black",
                labelcolor="black",
            )

    # Shared labels
    fig.supxlabel(
        "Proportion of focal group in candidate pool (%)",
        fontsize=9.2,
        y=0.075,
    )

    fig.supylabel(
        "Normalized absolute selection-rate difference (%)",
        fontsize=9.2,
        x=0.03,
    )

    # Shared legend
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=model_style[m]["color"],
            marker=model_style[m]["marker"],
            markerfacecolor="white",
            markeredgecolor=model_style[m]["color"],
            markeredgewidth=1.0,
            linewidth=1.25,
            markersize=4.5,
            label=model_style[m]["label"],
        )
        for m in model_names
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(model_names),
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
        handletextpad=0.6,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.995,
        bottom=0.155,
        top=0.900,
        wspace=0.26,
        hspace=0.72,
    )

    # ------------------------------------------------------------
    # Row headers:
    # Application name above each row;
    # attribute names below the application name.
    # ------------------------------------------------------------
    app_title_offset = 0.055
    attr_title_offset = 0.027

    for row_idx, application in enumerate(applications):
        pos_left = axes[row_idx, 0].get_position()
        pos_right = axes[row_idx, len(attribute_types) - 1].get_position()

        row_x_center = (pos_left.x0 + pos_right.x1) / 2
        row_y_top = pos_left.y1

        fig.text(
            row_x_center,
            row_y_top + app_title_offset,
            application_title_map.get(application, application),
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
        )

        for col_idx, attribute_type in enumerate(attribute_types):
            pos = axes[row_idx, col_idx].get_position()

            fig.text(
                (pos.x0 + pos.x1) / 2,
                row_y_top + attr_title_offset,
                attribute_type,
                ha="center",
                va="bottom",
                fontsize=9.0,
                fontweight="bold",
            )

    base = "all_applications_Gender_Race_llama_contextual"
    pdf_path = os.path.join(output_dir, base + ".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")

    plt.close(fig)


if __name__ == "__main__":
    applications = ["loan", "hiring", "edu"]
    model_names = ["Llama-3.1-8B", "Llama-3.1-8B-Instruct"]
    attribute_types = ["Gender", "Race"]

    # Fixed default context size
    context_size = 5

    # application -> attribute_type -> model -> delta_by_ratio
    app_attr_model_to_delta = defaultdict(lambda: defaultdict(dict))

    for application in applications:
        for attribute_type in attribute_types:
            for model_name in model_names:
                file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_{context_size}_500.jsonl"
                if not os.path.exists(file_name):
                    file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_{context_size}_200.jsonl"
                if not os.path.exists(file_name):
                    continue

                delta = compute_results(file_name, context_size=context_size)
                app_attr_model_to_delta[application][attribute_type][model_name] = delta

    draw_all_applications_in_one_figure(
        app_attr_model_to_delta=app_attr_model_to_delta,
        applications=applications,
        attribute_types=attribute_types,
        model_names=model_names,
    )