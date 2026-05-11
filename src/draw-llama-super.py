import json
import math
import os
import numpy as np
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import NormalDist


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}
FIG_FONT_SIZE = 12
MARKER_SIZE = 4.0

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



def compute_societal_results(attribute_type, file_name):
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


def compute_contextual_results(file_name, context_size, max_n_trials=1000000):
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
        fontsize=FIG_FONT_SIZE,
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

    minority_marker = "^"
    majority_marker = "o"

    x_gap = 0.40
    x_base = np.arange(len(attribute_types), dtype=float) * x_gap
    dodge = 0.065

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

        res = compute_societal_results(attribute_type, file_name)

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
            markersize=MARKER_SIZE,
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

        # Majority
        maj_mean = float(maj_res["mean"])
        maj_low = float(maj_res["ci_low"])
        maj_high = float(maj_res["ci_high"])

        ax.errorbar(
            x_majority[i],
            maj_mean,
            yerr=[[maj_mean - maj_low], [maj_high - maj_mean]],
            marker=majority_marker,
            markersize=MARKER_SIZE,
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
    attribute_tick_labels = {
        "Gender Identity": "GI",
        "Sexual Orientation": "SO",
    }
    ax.set_xticklabels(
        [attribute_tick_labels.get(a, a) for a in attribute_types]
    )
    ax.set_xlim(x_base[0] - 0.15, x_base[-1] + 0.15)

    for tick_label, attribute_type in zip(ax.get_xticklabels(), attribute_types):
        tick_label.set_color(attribute_to_color[attribute_type])
        # tick_label.set_fontweight("bold")

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))

    ax.tick_params(
        axis="both",
        # direction="out",
        length=0.0,
        width=0.0,
        # color="black",
        labelcolor="black",
    )

    # Re-apply x tick colors after tick_params
    for tick_label, attribute_type in zip(ax.get_xticklabels(), attribute_types):
        tick_label.set_color(attribute_to_color[attribute_type])
        # tick_label.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.grid(
    #     axis="y",
    #     color="0.88",
    #     linewidth=0.6,
    #     linestyle="-",
    #     zorder=0,
    # )
    ax.set_axisbelow(True)




# ============================================================
# Combined plotting: Societal + Contextual Llama results
# ============================================================

def draw_combined_llama_figure(
    applications,
    societal_attribute_types,
    contextual_attribute_types,
    model_names,
    context_size=5,
    output_dir="outputs/llama",
):
    """
    Draw one big Nature-style figure:

        a. Societal minority bias      b. Contextual minority bias

    Left block:
        rows = applications
        columns = models

    Right block:
        rows = applications
        columns = contextual attributes, e.g., Gender and Race
    """

    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship",
    }

    # Use different palettes for the two conceptual parts
    societal_attr_color = {
        "Gender Identity": "#BE0E23",        # green
        "Sexual Orientation": "#0634a0",     # purple
    }

    contextual_model_style = {
        "Llama-3.1-8B": {
            "color": "#f296ac",              # blue
            "marker": "^",
            "label": "Llama-3.1-8B",
        },
        "Llama-3.1-8B-Instruct": {
            "color": "#b53289",              # orange
            "marker": "o",
            "label": "Llama-3.1-8B-Instruct",
        },
    }

    fig = plt.figure(figsize=(9, 7))

    outer_gs = fig.add_gridspec(
        1,
        2,
        left=0.075,
        right=0.985,
        bottom=0.175,
        top=0.870,
        wspace=0.18,
    )

    societal_gs = outer_gs[0, 0].subgridspec(
        len(applications),
        len(model_names),
        wspace=0.16,
        hspace=0.65,
    )

    contextual_gs = outer_gs[0, 1].subgridspec(
        len(applications),
        len(contextual_attribute_types),
        wspace=0.16,
        hspace=0.65,
    )

    societal_axes = np.empty((len(applications), len(model_names)), dtype=object)
    contextual_axes = np.empty((len(applications), len(contextual_attribute_types)), dtype=object)

    # ============================================================
    # a. Societal minority bias
    # ============================================================
    for row_idx, application in enumerate(applications):
        for col_idx, model_name in enumerate(model_names):
            ax = fig.add_subplot(societal_gs[row_idx, col_idx])
            societal_axes[row_idx, col_idx] = ax

            plot_societal_panel(
                ax=ax,
                application=application,
                model_name=model_name,
                attribute_types=societal_attribute_types,
                attribute_to_color=societal_attr_color,
            )

            # ax.set_title(
            #     pretty_model_name(model_name),
            #     fontsize=FIG_FONT_SIZE,
            #     pad=5,
            # )

    # ============================================================
    # b. Contextual minority bias
    # ============================================================
    ratio_strs = ["20%", "40%", "60%", "80%"]
    ratio_step = 0.65
    ratio_x = np.arange(len(ratio_strs), dtype=float) * ratio_step

    # Load contextual results once
    app_attr_model_to_delta = defaultdict(lambda: defaultdict(dict))

    for application in applications:
        for attribute_type in contextual_attribute_types:
            for model_name in model_names:
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
                    print(f"[Warning] Missing file: {file_name}")
                    continue

                delta = compute_contextual_results(
                    file_name,
                    context_size=context_size,
                )
                app_attr_model_to_delta[application][attribute_type][model_name] = delta

    for row_idx, application in enumerate(applications):
        for col_idx, attribute_type in enumerate(contextual_attribute_types):
            ax = fig.add_subplot(contextual_gs[row_idx, col_idx])
            contextual_axes[row_idx, col_idx] = ax

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.axhline(0, color="0.30", linewidth=0.7, zorder=1)

            panel_ci_upper_values = []
            panel_upper_by_ratio = {r: 0.0 for r in ratio_strs}

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
                style = contextual_model_style[model_name]

                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    fmt=style["marker"] + "-",
                    color=style["color"],
                    markerfacecolor=style["color"],
                    markeredgecolor=style["color"],
                    markeredgewidth=1.0,
                    markersize=MARKER_SIZE,
                    linewidth=1.20,
                    elinewidth=0.80,
                    capsize=2.0,
                    capthick=0.80,
                    zorder=3,
                )

            # Statistical test: Llama-3.1-8B vs Llama-3.1-8B-Instruct
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

            for rx, rstr in zip(ratio_x, ratio_strs):
                if rstr not in star_positions:
                    continue

                ax.text(
                    rx,
                    star_positions[rstr],
                    p_to_stars(pvalues_by_ratio[rstr]),
                    ha="center",
                    va="bottom",
                    fontsize=FIG_FONT_SIZE,
                    color="0.10",
                    zorder=4,
                )

            ax.set_xlim(ratio_x[0] - 0.15, ratio_x[-1] + 0.15)
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
                labelcolor="black",
            )

    # ============================================================
    # Block-level titles and labels
    # ============================================================

    # Left block geometry
    soc_top_left = societal_axes[0, 0].get_position()
    soc_top_right = societal_axes[0, len(model_names) - 1].get_position()
    soc_bottom_left = societal_axes[-1, 0].get_position()

    soc_x0 = soc_top_left.x0
    soc_x1 = soc_top_right.x1
    soc_y1 = soc_top_left.y1
    soc_y0 = soc_bottom_left.y0
    soc_x_center = (soc_x0 + soc_x1) / 2
    soc_y_center = (soc_y0 + soc_y1) / 2

    # Right block geometry
    ctx_top_left = contextual_axes[0, 0].get_position()
    ctx_top_right = contextual_axes[0, len(contextual_attribute_types) - 1].get_position()
    ctx_bottom_left = contextual_axes[-1, 0].get_position()

    ctx_x0 = ctx_top_left.x0
    ctx_x1 = ctx_top_right.x1
    ctx_y1 = ctx_top_left.y1
    ctx_y0 = ctx_bottom_left.y0
    ctx_x_center = (ctx_x0 + ctx_x1) / 2
    ctx_y_center = (ctx_y0 + ctx_y1) / 2

    title_y_offset = 0.080
    letter_y_offset = 0.080

    # Panel letters and block titles
    fig.text(
        soc_x0 - 0.030,
        soc_y1 + letter_y_offset,
        "a",
        ha="left",
        va="bottom",
        fontsize=FIG_FONT_SIZE,
        fontweight="bold",
    )

    fig.text(
        soc_x0 + 0.005,
        soc_y1 + title_y_offset,
        "Societal minority bias",
        ha="left",
        va="bottom",
        fontsize=FIG_FONT_SIZE,
        fontweight="bold",
    )

    fig.text(
        ctx_x0 - 0.030,
        ctx_y1 + letter_y_offset,
        "b",
        ha="left",
        va="bottom",
        fontsize=FIG_FONT_SIZE,
        fontweight="bold",
    )

    fig.text(
        ctx_x0 + 0.005,
        ctx_y1 + title_y_offset,
        "Contextual minority bias",
        ha="left",
        va="bottom",
        fontsize=FIG_FONT_SIZE,
        fontweight="bold",
    )

    ylabel_offset = 0.045

    # Block y-axis labels
    fig.text(
        soc_x0 - ylabel_offset,
        soc_y_center,
        "Score",
        ha="center",
        va="center",
        rotation=90,
        fontsize=FIG_FONT_SIZE,
    )

    fig.text(
        ctx_x0 - ylabel_offset,
        ctx_y_center,
        "Normalized absolute selection-rate difference (%)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=FIG_FONT_SIZE,
    )

    shared_x_label_y = min(soc_y0, ctx_y0) - 0.045

    # Societal x-axis label
    fig.text(
        soc_x_center,
        shared_x_label_y,
        "Attribute",
        ha="center",
        va="top",
        fontsize=FIG_FONT_SIZE,
    )

    # Contextual x-axis label
    fig.text(
        ctx_x_center,
        shared_x_label_y,
        "Proportion of focal group in candidate pool (%)",
        ha="center",
        va="top",
        fontsize=FIG_FONT_SIZE,
    )

    # ============================================================
    # Row labels
    # ============================================================

    row_app_offset = 0.038
    column_label_offset = 0.008   # shared by Llama labels and Gender/Race labels

    for row_idx, application in enumerate(applications):
        app_name = application_title_map.get(application, application)

        # ------------------------------------------------------------
        # Societal block: application name
        # ------------------------------------------------------------
        soc_pos_left = societal_axes[row_idx, 0].get_position()
        soc_pos_right = societal_axes[row_idx, len(model_names) - 1].get_position()

        soc_row_x_center = (soc_pos_left.x0 + soc_pos_right.x1) / 2
        soc_row_y_top = soc_pos_left.y1

        fig.text(
            soc_row_x_center,
            soc_row_y_top + row_app_offset,
            app_name,
            ha="center",
            va="bottom",
            fontsize=FIG_FONT_SIZE,
            fontweight="bold",
        )

        # ------------------------------------------------------------
        # Societal block: model labels
        # Use the same offset as Gender/Race labels.
        # ------------------------------------------------------------
        for col_idx, model_name in enumerate(model_names):
            pos = societal_axes[row_idx, col_idx].get_position()

            fig.text(
                (pos.x0 + pos.x1) / 2,
                pos.y1 + column_label_offset,
                pretty_model_name(model_name),
                ha="center",
                va="bottom",
                fontsize=FIG_FONT_SIZE,
            )

        # ------------------------------------------------------------
        # Contextual block: application name
        # ------------------------------------------------------------
        ctx_pos_left = contextual_axes[row_idx, 0].get_position()
        ctx_pos_right = contextual_axes[row_idx, len(contextual_attribute_types) - 1].get_position()

        ctx_row_x_center = (ctx_pos_left.x0 + ctx_pos_right.x1) / 2
        ctx_row_y_top = ctx_pos_left.y1

        fig.text(
            ctx_row_x_center,
            ctx_row_y_top + row_app_offset,
            app_name,
            ha="center",
            va="bottom",
            fontsize=FIG_FONT_SIZE,
            fontweight="bold",
        )

        # ------------------------------------------------------------
        # Contextual block: attribute labels
        # Same offset as the Llama model labels.
        # ------------------------------------------------------------
        for col_idx, attribute_type in enumerate(contextual_attribute_types):
            pos = contextual_axes[row_idx, col_idx].get_position()

            fig.text(
                (pos.x0 + pos.x1) / 2,
                pos.y1 + column_label_offset,
                attribute_type,
                ha="center",
                va="bottom",
                fontsize=FIG_FONT_SIZE,
            )

    # -----------------------------
    # Legend for panel a: societal
    # -----------------------------
    societal_legend_handles = [
        # text-only entries for model names
        Line2D(
            [0], [0],
            linestyle="",
            marker=None,
            markersize=0.0,
            color="#BE0E23",
            label="GI: Gender identity",
        ),
        Line2D(
            [0], [0],
            linestyle="",
            marker=None,
            markersize=0.0,
            color="#0634a0",
            label="SO: Sexual orientation",
        ),
        Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=5.0,
            label="Societal minority",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=5.0,
            label="Societal majority",
        ),
    ]

    shared_legend_y = shared_x_label_y - 0.08

    leg = fig.legend(
        handles=societal_legend_handles,
        loc="lower center",
        bbox_to_anchor=(soc_x_center - 0.03, shared_legend_y - 0.035),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.40,
        fontsize=FIG_FONT_SIZE,
    )

    # Hide the first two dummy handles
    for h in leg.legend_handles[:2]:
        h.set_visible(False)

    # Color the first two legend texts
    legend_texts = leg.get_texts()
    legend_texts[0].set_color("#BE0E23")   # GI text
    legend_texts[1].set_color("#0634a0")   # SO text

    # -----------------------------
    # Legend for panel b: contextual
    # -----------------------------
    contextual_legend_handles = [
        Line2D(
            [0],
            [0],
            color=contextual_model_style["Llama-3.1-8B"]["color"],
            marker=contextual_model_style["Llama-3.1-8B"]["marker"],
            markerfacecolor=contextual_model_style["Llama-3.1-8B"]["color"],
            markeredgecolor=contextual_model_style["Llama-3.1-8B"]["color"],
            markeredgewidth=1.0,
            linewidth=1.25,
            markersize=5.0,
            label="Llama-3.1-8B",
        ),
        Line2D(
            [0],
            [0],
            color=contextual_model_style["Llama-3.1-8B-Instruct"]["color"],
            marker=contextual_model_style["Llama-3.1-8B-Instruct"]["marker"],
            markerfacecolor=contextual_model_style["Llama-3.1-8B-Instruct"]["color"],
            markeredgecolor=contextual_model_style["Llama-3.1-8B-Instruct"]["color"],
            markeredgewidth=1.0,
            linewidth=1.25,
            markersize=5.0,
            label="Llama-3.1-8B-Instruct",
        ),
    ]

    fig.legend(
        handles=contextual_legend_handles,
        loc="lower center",
        bbox_to_anchor=(ctx_x_center, shared_legend_y - 0.02),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.40,
        fontsize=FIG_FONT_SIZE,
    )

    # Save
    base = "llama_societal_contextual_combined"
    pdf_path = os.path.join(output_dir, base + ".pdf")

    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {pdf_path}")

    plt.close(fig)


if __name__ == "__main__":
    applications = ["hiring", "loan", "edu"]

    societal_attribute_types = [
        "Gender Identity",
        "Sexual Orientation",
    ]

    contextual_attribute_types = [
        "Gender",
        "Race",
    ]

    model_names = [
        "Llama-3.1-8B",
        "Llama-3.1-8B-Instruct",
    ]

    draw_combined_llama_figure(
        applications=applications,
        societal_attribute_types=societal_attribute_types,
        contextual_attribute_types=contextual_attribute_types,
        model_names=model_names,
        context_size=5,
        output_dir="outputs/llama",
    )