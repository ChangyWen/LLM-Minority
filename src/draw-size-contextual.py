import csv
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
from scipy.stats import chi2_contingency, norm
from statistics import NormalDist
from matplotlib.lines import Line2D
import re
import string


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


def p_to_stars(p):
    """
    Convert p-value to significance stars.
    """
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def se_diff_of_props(hA, nA, hB, nB):
    pA = hA / nA
    pB = hB / nB
    return math.sqrt(pA*(1-pA)/nA + pB*(1-pB)/nB)


def abs_diff(h1, n1, h2, n2):
    return abs((h1 / n1) - (h2 / n2))


def norm_delta_and_se(raw_data, context_size):
    """
    raw_data = (hA, nA, hB, nB) for the two groups at a fixed ratio.
    Returns:
      d_norm = context_size * |pA - pB|
      se_norm = context_size * SE(|pA - pB|)  (same SE as signed diff)
    """
    hA, nA, hB, nB = raw_data
    d = abs_diff(hA, nA, hB, nB)
    se = se_diff_of_props(hA, nA, hB, nB)
    return context_size * d, context_size * se


def one_sided_p(delta10, se10, delta5, se5):
    d = delta10 - delta5
    se = math.sqrt(se10**2 + se5**2)
    if se == 0:
        # Degenerate: if d>0 then p ~ 0 else p ~ 1
        return 0.0 if d > 0 else 1.0
    z = d / se
    return 1.0 - NormalDist().cdf(z)


# ---------------------------------------------------------------------
# Robust p-value computation
# ---------------------------------------------------------------------
def iut_pvalue_by_ratio(counts_size5, counts_size10, ratios=("20%", "40%", "60%", "80%")):
    """
    Per-ratio one-sided tests for whether the normalized disparity
    is larger under context size 10 than under context size 5.

    Returns a dictionary:
        {"20%": p, "40%": p, ...}

    If either context size is missing, returns an empty dictionary.
    """
    if counts_size5 is None or counts_size10 is None:
        return {}

    per_ratio_p = {}

    for r in ratios:
        if r not in counts_size5 or r not in counts_size10:
            continue

        d5, se5 = norm_delta_and_se(counts_size5[r], context_size=5)
        d10, se10 = norm_delta_and_se(counts_size10[r], context_size=10)

        p = one_sided_p(d10, se10, d5, se5)
        per_ratio_p[r] = p

    return per_ratio_p


# ---------------------------------------------------------------------
# Benjamini-Hochberg correction
# ---------------------------------------------------------------------
def benjamini_hochberg(p_values):
    """
    Return Benjamini-Hochberg-adjusted P values in the original order.

    The procedure controls the false discovery rate across the supplied
    family of tests. All values must be finite numbers in [0, 1].
    """
    p_values = np.asarray(p_values, dtype=float)

    if p_values.ndim != 1:
        raise ValueError("p_values must be a one-dimensional sequence.")
    if p_values.size == 0:
        return p_values.copy()
    if np.any(~np.isfinite(p_values)):
        raise ValueError("p_values must contain only finite values.")
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("Every P value must lie in [0, 1].")

    n_tests = p_values.size
    order = np.argsort(p_values)
    ranked_p = p_values[order]

    adjusted_ranked = ranked_p * n_tests / np.arange(1, n_tests + 1, dtype=float)

    # Enforce monotonicity from the largest rank back to the smallest.
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def adjust_figure5_pvalues_bh(
    raw_pvalues,
    attribute_types,
    applications,
    model_names,
    ratios=("20%", "40%", "60%", "80%"),
):
    """
    Apply one BH correction family per application/Figure 5.

    With the default configuration, each application contains
        2 attributes x 8 models x 4 target-group proportions = 64 tests.

    Missing tests are omitted from that application's correction family.
    The returned object has the same nested structure as raw_pvalues.
    """
    adjusted_pvalues = {
        attribute_type: {
            application: {model_name: {} for model_name in model_names}
            for application in applications
        }
        for attribute_type in attribute_types
    }

    for application in applications:
        test_keys = []
        test_pvalues = []

        for attribute_type in attribute_types:
            for model_name in model_names:
                per_ratio_p = (
                    raw_pvalues
                    .get(attribute_type, {})
                    .get(application, {})
                    .get(model_name, {})
                )

                for ratio in ratios:
                    p = per_ratio_p.get(ratio)
                    if p is None or not math.isfinite(p):
                        continue
                    test_keys.append((attribute_type, model_name, ratio))
                    test_pvalues.append(float(p))

        adjusted_values = benjamini_hochberg(test_pvalues)

        for (attribute_type, model_name, ratio), adjusted_p in zip(
            test_keys, adjusted_values
        ):
            adjusted_pvalues[attribute_type][application][model_name][ratio] = float(
                adjusted_p
            )

        print(
            f"BH correction for application={application}: "
            f"adjusted {len(test_pvalues)} P values jointly."
        )

    return adjusted_pvalues


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


def safe_slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def save_pvalue_summary_csv(
    raw_pvalues,
    adjusted_pvalues,
    attribute_types,
    applications,
    model_names,
    ratios=("20%", "40%", "60%", "80%"),
    output_dir="outputs/size",
):
    """Save raw and BH-adjusted P values used for Figure 5."""
    os.makedirs(output_dir, exist_ok=True)

    for application in applications:
        csv_path = os.path.join(
            output_dir,
            f"{safe_slug(application)}_Figure5_BH_adjusted_pvalues.csv",
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "application",
                    "attribute",
                    "model",
                    "target_group_proportion",
                    "raw_p_value",
                    "bh_adjusted_p_value",
                    "significance",
                ],
            )
            writer.writeheader()

            for attribute_type in attribute_types:
                for model_name in model_names:
                    raw_per_ratio = (
                        raw_pvalues
                        .get(attribute_type, {})
                        .get(application, {})
                        .get(model_name, {})
                    )
                    adjusted_per_ratio = (
                        adjusted_pvalues
                        .get(attribute_type, {})
                        .get(application, {})
                        .get(model_name, {})
                    )

                    for ratio in ratios:
                        raw_p = raw_per_ratio.get(ratio)
                        adjusted_p = adjusted_per_ratio.get(ratio)
                        if raw_p is None or adjusted_p is None:
                            continue

                        writer.writerow({
                            "application": application,
                            "attribute": attribute_type,
                            "model": pretty_model_name(model_name),
                            "target_group_proportion": ratio,
                            "raw_p_value": f"{raw_p:.12g}",
                            "bh_adjusted_p_value": f"{adjusted_p:.12g}",
                            "significance": p_to_stars(adjusted_p),
                        })

        print(f"Saved: {csv_path}")


def compute_results(file_name, context_size, max_n_trials=1000000):

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
    random_selection_rate = 1 / context_size
    for c in sorted(set(attr_counts_A) & set(attr_counts_B)):
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]
        pA = hA / nA
        pB = hB / nB
        if pA > pB:
            delta = pA - pB
            ciA_low, ciA_high = wilson_ci(hA, nA)
            ciB_low, ciB_high = wilson_ci(hB, nB)
            ci_low = ciA_low - ciB_high
            ci_high = ciA_high - ciB_low
            raw_data = (hA, nA, hB, nB)
        else:
            delta = pB - pA
            ciA_low, ciA_high = wilson_ci(hA, nA)
            ciB_low, ciB_high = wilson_ci(hB, nB)
            ci_low = ciB_low - ciA_high
            ci_high = ciB_high - ciA_low
            raw_data = (hB, nB, hA, nA)
        ratio = (c + 1) / context_size
        ratio_str = f"{ratio * 100:.0f}%"
        if ratio_str in ["20%", "40%", "60%", "80%"]:
            results["delta"][ratio_str] = {
                "raw_delta": delta,
                "random_selection_rate": random_selection_rate,
                "delta": delta / random_selection_rate,
                "ci_low": ci_low / random_selection_rate,
                "ci_high": ci_high / random_selection_rate,
                "raw_data": raw_data,
            }

    return results["delta"]


def get_global_ylim(application, application_to_model_to_delta, model_names, context_sizes):
    """
    Compute a common y-axis limit across all model panels within one figure.
    This improves comparability across subplots.
    """
    upper_values = []

    for model_key in model_names:
        for cs in context_sizes:
            delta_by_ratio = (
                application_to_model_to_delta
                .get(application, {})
                .get(model_key, {})
                .get(cs, {})
            )

            for _, d in delta_by_ratio.items():
                upper_values.append(max(0.0, float(d["ci_high"])))

    if not upper_values:
        return 0.0, 1.0

    ymax = max(upper_values)
    ymax = max(ymax, 0.05)

    # Add headroom for significance stars
    return 0.0, ymax * 1.25


def draw_combined_gender_race_by_application(
    attribute_type_to_application_to_model_to_delta,
    attribute_type_to_application_to_model_to_adjusted_pvalue,
    attribute_types,
    model_names,
    context_sizes,
    output_dir="outputs/size",
):
    """
    Nature-style combined multi-panel figure.

    For each application, draw one large figure:
        - Top block: Gender, arranged as 2 x 4 panels
        - Bottom block: Race, arranged as 2 x 4 panels

    Significance stars are based on Benjamini-Hochberg-adjusted P values.

    This version uses nested GridSpec:
        - outer_hspace controls the gap between Gender and Race
        - inner_hspace controls the row spacing within each block
    """

    set_nature_style()
    os.makedirs(output_dir, exist_ok=True)

    context_sizes = list(sorted(context_sizes))

    ratio_strs = ["20%", "40%", "60%", "80%"]
    ratio_x = np.array([20, 40, 60, 80], dtype=float)

    # Colorblind-safe Okabe-Ito palette
    context_style = {
        5: {
            "color": "#aad79d",
            "marker": "^",
            "label": "Pool size 5",
        },
        10: {
            "color": "#44a05c",
            "marker": "o",
            "label": "Pool size 10",
        },
    }

    fallback_colors = ["#009E73", "#CC79A7", "#56B4E9", "#E69F00"]

    for idx, cs in enumerate(context_sizes):
        if cs not in context_style:
            context_style[cs] = {
                "color": fallback_colors[idx % len(fallback_colors)],
                "marker": "o",
                "label": f"Pool size {cs}",
            }

    applications = sorted({
        application
        for attribute_type in attribute_types
        for application in attribute_type_to_application_to_model_to_delta[attribute_type].keys()
    })

    for application in applications:

        # ------------------------------------------------------------------
        # Nested GridSpec layout
        # ------------------------------------------------------------------
        fig = plt.figure(figsize=(7.45, 6))

        outer_gs = fig.add_gridspec(
            2, 1,
            left=0.095,
            right=0.995,
            bottom=0.105,
            top=0.935,
            hspace=0.25,   # larger gap between Gender and Race blocks
        )

        gender_gs = outer_gs[0].subgridspec(
            2, 4,
            wspace=0.30,
            hspace=0.35,   # smaller row spacing within Gender block
        )

        race_gs = outer_gs[1].subgridspec(
            2, 4,
            wspace=0.30,
            hspace=0.35,   # smaller row spacing within Race block
        )

        axes = np.empty((4, 4), dtype=object)

        for r in range(2):
            for c in range(4):
                axes[r, c] = fig.add_subplot(gender_gs[r, c])
                axes[r + 2, c] = fig.add_subplot(race_gs[r, c])

        # ------------------------------------------------------------------
        # Draw panels
        # ------------------------------------------------------------------
        for attr_idx, attribute_type in enumerate(attribute_types):
            row_offset = attr_idx * 2

            application_to_model_to_delta = attribute_type_to_application_to_model_to_delta[attribute_type]
            application_to_model_to_adjusted_pvalue = (
                attribute_type_to_application_to_model_to_adjusted_pvalue[attribute_type]
            )

            for i, model_key in enumerate(model_names):
                row = row_offset + (i // 4)
                col = i % 4
                ax = axes[row, col]
                ax.set_box_aspect(0.65)

                ymin = 0.0

                ax.axhline(0, color="0.30", linewidth=0.7, zorder=1)

                panel_upper_by_ratio = {r: 0.0 for r in ratio_strs}
                panel_ci_upper_values = []

                for cs in context_sizes:
                    delta_by_ratio = (
                        application_to_model_to_delta
                        .get(application, {})
                        .get(model_key, {})
                        .get(cs, {})
                    )

                    if not delta_by_ratio:
                        continue

                    xs, ys, yerr_low, yerr_high = [], [], [], []

                    for rx, rstr in zip(ratio_x, ratio_strs):
                        if rstr not in delta_by_ratio:
                            continue

                        d = delta_by_ratio[rstr]

                        y = float(d["delta"])

                        # Keep the non-negative display convention.
                        # If you want to show CIs crossing below 0, remove max(0.0, ...).
                        lo = max(0.0, float(d["ci_low"]))
                        hi = max(0.0, float(d["ci_high"]))

                        xs.append(rx)
                        ys.append(y)
                        yerr_low.append(max(0.0, y - lo))
                        yerr_high.append(max(0.0, hi - y))

                        panel_upper_by_ratio[rstr] = max(panel_upper_by_ratio[rstr], hi)
                        panel_ci_upper_values.append(hi)

                    if len(xs) == 0:
                        continue

                    yerr = np.vstack([yerr_low, yerr_high])
                    style = context_style[cs]

                    ax.errorbar(
                        xs,
                        ys,
                        yerr=yerr,
                        fmt=style["marker"] + "-",
                        color=style["color"],
                        markerfacecolor=style["color"],
                        markeredgecolor=style["color"],
                        markeredgewidth=1.0,
                        markersize=4.0,
                        linewidth=1.20,
                        elinewidth=0.80,
                        capsize=2.0,
                        capthick=0.80,
                        zorder=3,
                    )

                # ----------------------------------------------------------
                # Panel-specific y-axis upper limit
                # ----------------------------------------------------------
                if len(panel_ci_upper_values) == 0:
                    panel_data_top = 1.0
                else:
                    panel_data_top = max(panel_ci_upper_values)
                    panel_data_top = max(panel_data_top, 0.05)

                # Significance labels are based on BH-adjusted P values.
                per_ratio_adjusted_p = (
                    application_to_model_to_adjusted_pvalue
                    .get(application, {})
                    .get(model_key, {})
                )

                star_positions = {}
                star_offset = 0.075 * panel_data_top

                for rx, rstr in zip(ratio_x, ratio_strs):
                    stars = p_to_stars(per_ratio_adjusted_p.get(rstr, float("nan")))
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

                ax.set_ylim(ymin, ymax_i)

                for rx, rstr in zip(ratio_x, ratio_strs):
                    stars = p_to_stars(per_ratio_adjusted_p.get(rstr, float("nan")))

                    if stars:
                        ax.text(
                            rx,
                            star_positions[rstr],
                            stars,
                            ha="center",
                            va="bottom",
                            fontsize=7.0,
                            # fontweight="bold",
                            color="0.10",
                            zorder=4,
                        )

                ax.set_title(
                    pretty_model_name(model_key),
                    loc="center",
                    pad=4,
                    fontsize=8.0,
                    # fontweight="bold",
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
                    # direction="out",
                    length=0.0,
                    width=0.0,
                    # color="black",
                )

        # ------------------------------------------------------------------
        # Shared label and legend
        # ------------------------------------------------------------------

        # Keep this removed if you do not want the shared x-label.
        fig.supxlabel(
            "Target-group proportion in candidate pool (%)",
            fontsize=9.2,
            y=0.04,
        )

        fig.supylabel(
            "Normalized absolute candidate-level selection-rate difference (%)",
            fontsize=9.2,
            x=0.033,
        )

        legend_handles = [
            Line2D(
                [0], [0],
                color=context_style[cs]["color"],
                marker=context_style[cs]["marker"],
                markerfacecolor=context_style[cs]["color"],
                markeredgecolor=context_style[cs]["color"],
                markeredgewidth=1.0,
                linewidth=1.25,
                markersize=4.5,
                label=context_style[cs]["label"],
            )
            for cs in context_sizes
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=len(context_sizes),
            frameon=False,
            handlelength=1.8,
            columnspacing=1.4,
        )

        # ------------------------------------------------------------------
        # Block subtitles: Gender and Race
        # ------------------------------------------------------------------
        for attr_idx, attribute_type in enumerate(attribute_types):
            row_offset = attr_idx * 2

            pos_top_left = axes[row_offset, 0].get_position()

            x_center = 0.5   # center with respect to the whole figure
            y_text = pos_top_left.y1 + 0.03

            fig.text(
                x_center,
                y_text,
                attribute_type,
                ha="center",
                va="bottom",
                fontsize=10.0,
                fontweight="bold",
            )

        base = f"{safe_slug(application)}_Gender_Race_size_contextual"
        pdf_path = os.path.join(output_dir, base + ".pdf")

        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

        plt.close(fig)


if __name__ == "__main__":
    applications = ["loan"]

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

    context_sizes = [5, 10]

    attribute_types = ["Gender", "Race"]

    attribute_type_to_application_to_model_to_delta = {}
    attribute_type_to_application_to_model_to_raw_pvalue = {}

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(dict))
        application_to_model_to_pvalue = defaultdict(lambda: defaultdict(dict))

        for application in applications:
            for model_name in model_names:
                counts_size5 = None
                counts_size10 = None

                for context_size in context_sizes:
                    if "no_thinking" in model_name:
                        file_name = (
                            f"outputs/{application}/contextual/{attribute_type}/"
                            f"{model_name[:-12]}_{context_size}_500_no_thinking.jsonl"
                        )
                        if not os.path.exists(file_name):
                            file_name = (
                                f"outputs/{application}/contextual/{attribute_type}/"
                                f"{model_name[:-12]}_{context_size}_200_no_thinking.jsonl"
                            )
                    else:
                        file_name = (
                            f"outputs/{application}/contextual/{attribute_type}/"
                            f"{model_name}_{context_size}_500.jsonl"
                        )
                        if not os.path.exists(file_name):
                            file_name = (
                                f"outputs/{application}/contextual/{attribute_type}/"
                                f"{model_name}_{context_size}_200.jsonl"
                            )

                    if not os.path.exists(file_name):
                        continue

                    delta = compute_results(file_name, context_size)
                    raw_data = {
                        k: v["raw_data"]
                        for k, v in delta.items()
                    }

                    if context_size == 5:
                        counts_size5 = raw_data
                    else:
                        counts_size10 = raw_data

                    application_to_model_to_delta[application][model_name][context_size] = delta

                per_ratio_p = iut_pvalue_by_ratio(counts_size5, counts_size10)
                application_to_model_to_pvalue[application][model_name] = per_ratio_p

                print(
                    f"attribute_type: {attribute_type}, "
                    f"application: {application}, "
                    f"model_name: {model_name}, "
                    f"per_ratio_p: {per_ratio_p}"
                )

        attribute_type_to_application_to_model_to_delta[attribute_type] = application_to_model_to_delta
        attribute_type_to_application_to_model_to_raw_pvalue[attribute_type] = application_to_model_to_pvalue

    # Adjust all Figure 5 tests jointly within each application.
    # With the default settings, this is one family of 64 tests:
    # 2 attributes x 8 models x 4 target-group proportions.
    attribute_type_to_application_to_model_to_adjusted_pvalue = (
        adjust_figure5_pvalues_bh(
            raw_pvalues=attribute_type_to_application_to_model_to_raw_pvalue,
            attribute_types=attribute_types,
            applications=applications,
            model_names=model_names,
        )
    )

    save_pvalue_summary_csv(
        raw_pvalues=attribute_type_to_application_to_model_to_raw_pvalue,
        adjusted_pvalues=attribute_type_to_application_to_model_to_adjusted_pvalue,
        attribute_types=attribute_types,
        applications=applications,
        model_names=model_names,
        output_dir="outputs/size",
    )

    draw_combined_gender_race_by_application(
        attribute_type_to_application_to_model_to_delta=attribute_type_to_application_to_model_to_delta,
        attribute_type_to_application_to_model_to_adjusted_pvalue=(
            attribute_type_to_application_to_model_to_adjusted_pvalue
        ),
        attribute_types=attribute_types,
        model_names=model_names,
        context_sizes=context_sizes,
    )
