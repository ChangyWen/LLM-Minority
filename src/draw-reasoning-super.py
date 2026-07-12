import csv
import json
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy import stats


def se_diff_of_props(hA, nA, hB, nB):
    """
    Standard error for the difference between two proportions.
    Used for contextual minority bias.
    """
    pA = hA / nA
    pB = hB / nB
    return math.sqrt(pA * (1 - pA) / nA + pB * (1 - pB) / nB)


def p_to_stars(p):
    """
    Convert a p-value to significance stars.
    """
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p_value(p):
    """
    Optional exact p-value formatting.
    """
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return r"$P<0.001$"
    return rf"$P={p:.2f}$"


def reasoning_lower_p_value(reasoning_result, nonreasoning_result):
    """
    One-sided test for:

        H1: bias_reasoning < bias_nonreasoning

    For contextual results:
        Uses a normal approximation based on the SE of the absolute
        difference between two proportions.

    For societal results:
        Uses bootstrap distributions of the relative score difference.
    """

    # Societal case: bootstrap distributions are available
    if (
        "boot_deltas" in reasoning_result
        and "boot_deltas" in nonreasoning_result
    ):
        boot_r = np.asarray(reasoning_result["boot_deltas"], dtype=float)
        boot_nr = np.asarray(nonreasoning_result["boot_deltas"], dtype=float)

        boot_r = boot_r[np.isfinite(boot_r)]
        boot_nr = boot_nr[np.isfinite(boot_nr)]

        n = min(len(boot_r), len(boot_nr))
        if n == 0:
            return np.nan

        diff = boot_r[:n] - boot_nr[:n]

        # One-sided p-value for reasoning not being lower.
        # Smaller p supports reasoning < non-reasoning.
        p = (np.sum(diff >= 0) + 1) / (n + 1)
        return float(p)

    # Contextual case: SEs are available
    if "se" in reasoning_result and "se" in nonreasoning_result:
        delta_r = float(reasoning_result["delta"])
        delta_nr = float(nonreasoning_result["delta"])

        se_r = float(reasoning_result["se"])
        se_nr = float(nonreasoning_result["se"])

        se = math.sqrt(se_r ** 2 + se_nr ** 2)

        if se == 0:
            return 0.0 if delta_r < delta_nr else 1.0

        z = (delta_r - delta_nr) / se

        # Left-tailed test: reasoning bias is smaller.
        p = stats.norm.cdf(z)
        return float(p)

    return np.nan




def benjamini_hochberg(p_values):
    """
    Adjust a sequence of P values using the Benjamini--Hochberg procedure.

    Non-finite entries are preserved as NaN and are excluded from the
    adjustment family. In this Figure 8 analysis, all 24 P values are expected
    to be finite.
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)

    valid_mask = np.isfinite(p_values)
    valid_p = p_values[valid_mask]

    if valid_p.size == 0:
        return adjusted

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    ranks = np.arange(1, valid_p.size + 1, dtype=float)

    ranked_adjusted = ranked_p * valid_p.size / ranks

    # Enforce the BH monotonicity constraint from largest to smallest rank.
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)

    valid_adjusted = np.empty_like(ranked_adjusted)
    valid_adjusted[order] = ranked_adjusted
    adjusted[valid_mask] = valid_adjusted

    return adjusted


REASONING_PAIR_DEFS = [
    ("GLM-4.5-Air", "GLM-4.5-Air_no_thinking", "GLM"),
    (
        "NVIDIA-Nemotron-Nano-12B-v2",
        "NVIDIA-Nemotron-Nano-12B-v2_no_thinking",
        "Nemotron",
    ),
]


def compute_reasoning_pvalue_results(
    societal_results,
    contextual_results,
    applications,
    alpha=0.05,
):
    """
    Compute all 24 one-sided reasoning-versus-non-reasoning tests and apply
    one Benjamini--Hochberg correction jointly across the complete Figure 8
    family:

        2 bias types x 2 attributes x 3 scenarios x 2 models = 24 tests.

    Returns
    -------
    lookup : dict
        Maps (bias_type, attribute_type, application, short_model_name) to a
        result record containing raw and BH-adjusted P values.
    records : list[dict]
        Flat records suitable for printing or CSV export.
    """
    test_blocks = [
        (
            "societal",
            societal_results,
            ["Gender Identity", "Sexual Orientation"],
        ),
        (
            "contextual",
            contextual_results,
            ["Gender", "Race"],
        ),
    ]

    records = []

    for bias_type, results, attribute_types in test_blocks:
        for attribute_type in attribute_types:
            for application in applications:
                for reasoning_name, nonreasoning_name, short_label in REASONING_PAIR_DEFS:
                    reasoning_result = results[attribute_type][application][reasoning_name]
                    nonreasoning_result = results[attribute_type][application][nonreasoning_name]

                    raw_p = reasoning_lower_p_value(
                        reasoning_result=reasoning_result,
                        nonreasoning_result=nonreasoning_result,
                    )

                    records.append({
                        "bias_type": bias_type,
                        "attribute": attribute_type,
                        "application": application,
                        "model": short_label,
                        "reasoning_model": reasoning_name,
                        "nonreasoning_model": nonreasoning_name,
                        "bias_reasoning": float(reasoning_result["delta"]),
                        "bias_nonreasoning": float(nonreasoning_result["delta"]),
                        "p_raw": float(raw_p),
                    })

    expected_n_tests = 24
    if len(records) != expected_n_tests:
        raise RuntimeError(
            f"Expected {expected_n_tests} Figure 8 tests, but constructed "
            f"{len(records)} tests."
        )

    raw_pvalues = np.asarray([record["p_raw"] for record in records], dtype=float)
    if not np.all(np.isfinite(raw_pvalues)):
        bad_indices = np.where(~np.isfinite(raw_pvalues))[0].tolist()
        raise RuntimeError(
            "All 24 raw P values must be finite before BH adjustment. "
            f"Non-finite P values occurred at record indices: {bad_indices}"
        )

    adjusted_pvalues = benjamini_hochberg(raw_pvalues)

    for record, adjusted_p in zip(records, adjusted_pvalues):
        record["p_adjusted"] = float(adjusted_p)
        record["significance"] = p_to_stars(adjusted_p)
        record["fdr_alpha"] = float(alpha)
        record["n_tests_in_bh_family"] = expected_n_tests
        record["multiple_testing_method"] = (
            "Benjamini-Hochberg; one family across all 24 Figure 8 tests"
        )

    lookup = {
        (
            record["bias_type"],
            record["attribute"],
            record["application"],
            record["model"],
        ): record
        for record in records
    }

    return lookup, records


def save_reasoning_pvalue_results(records, output_file):
    """Save raw and BH-adjusted Figure 8 statistics to a CSV file."""
    fieldnames = [
        "bias_type",
        "attribute",
        "application",
        "model",
        "reasoning_model",
        "nonreasoning_model",
        "bias_reasoning",
        "bias_nonreasoning",
        "p_raw",
        "p_adjusted",
        "significance",
        "fdr_alpha",
        "n_tests_in_bh_family",
        "multiple_testing_method",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


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
        zorder=4,
    )

    ax.text(
        (x1 + x2) / 2,
        y + bar_height + text_offset,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        clip_on=False,
        zorder=5,
    )


# ============================================================
# Shared style
# ============================================================

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

        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": 1.2,
    })


# ============================================================
# Contextual minority computation
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


def compute_contextual_results(file_name, attribute_type, max_n_trials=1000000):
    """
    Compute absolute selection-rate difference at the minimum contextual ratio.
    Returns:
        {
            "delta": ...,
            "ci_low": ...,
            "ci_high": ...
        }
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

    attr_counts_A = None
    attr_counts_B = None

    for attr_value, attr_value_results in attr_value_to_results.items():
        attr_counts = {}

        same_attr_count_to_count = dict(
            sorted(
                attr_value_results["same_attr_count_to_count"].items(),
                key=lambda x: x[0],
            )
        )

        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
            attr_counts[same_attr_count] = (hit_count, count)

        if attr_value in ["Black", "Female"]:
            attr_counts_A = attr_counts
        else:
            attr_counts_B = attr_counts

    if attr_counts_A is None or attr_counts_B is None:
        return None

    results = {}

    for c in sorted(set(attr_counts_A) & set(attr_counts_B)):
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]

        pA = hA / nA
        pB = hB / nB

        ciA_low, ciA_high = wilson_ci(hA, nA)
        ciB_low, ciB_high = wilson_ci(hB, nB)

        if pA > pB:
            delta = pA - pB
            ci_low = ciA_low - ciB_high
            ci_high = ciA_high - ciB_low
        else:
            delta = pB - pA
            ci_low = ciB_low - ciA_high
            ci_high = ciB_high - ciA_low

        se = se_diff_of_props(hA, nA, hB, nB)

        results[c] = {
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "se": se,
        }

    if 1 not in results:
        return None

    return results[1]


# ============================================================
# Societal minority computation
# ============================================================

type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}


def bootstrap_relative_diff_ci(
    minority_scores,
    majority_scores,
    n_bootstrap=5000,
    alpha=0.05,
    seed=42,
):
    rng = np.random.default_rng(seed)

    minority_scores = np.asarray(minority_scores, dtype=float)
    majority_scores = np.asarray(majority_scores, dtype=float)

    n_min = len(minority_scores)
    n_maj = len(majority_scores)

    deltas = []

    for _ in range(n_bootstrap):
        min_sample = rng.choice(minority_scores, size=n_min, replace=True)
        maj_sample = rng.choice(majority_scores, size=n_maj, replace=True)

        min_mean = np.mean(min_sample)
        maj_mean = np.mean(maj_sample)

        if maj_mean == 0:
            continue

        delta = (min_mean - maj_mean) / maj_mean
        deltas.append(delta)

    deltas = np.asarray(deltas, dtype=float)

    if len(deltas) == 0:
        return None

    lower = np.percentile(deltas, 100 * alpha / 2)
    upper = np.percentile(deltas, 100 * (1 - alpha / 2))

    return {
        "delta": float(np.mean(deltas)),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "boot_deltas": deltas,
    }


def compute_societal_results(file_name, attribute_type):
    """
    Compute relative score difference:
        (minority_mean - majority_mean) / majority_mean
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

    if len(minority_scores) == 0 or len(majority_scores) == 0:
        return None

    return bootstrap_relative_diff_ci(minority_scores, majority_scores)


# ============================================================
# Plotting helpers
# ============================================================

def get_file_contextual(application, attribute_type, model_name):
    if "no_thinking" in model_name:
        base_model = model_name[:-12]
        file_name = (
            f"outputs/{application}/contextual/{attribute_type}/"
            f"{base_model}_5_500_no_thinking.jsonl"
        )
        if not os.path.exists(file_name):
            file_name = (
                f"outputs/{application}/contextual/{attribute_type}/"
                f"{base_model}_5_200_no_thinking.jsonl"
            )
    else:
        file_name = (
            f"outputs/{application}/contextual/{attribute_type}/"
            f"{model_name}_5_500.jsonl"
        )
        if not os.path.exists(file_name):
            file_name = (
                f"outputs/{application}/contextual/{attribute_type}/"
                f"{model_name}_5_200.jsonl"
            )

    return file_name


def get_file_societal(application, attribute_type, model_name):
    return f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"


def collect_results(
    applications,
    attribute_types,
    model_names,
    result_type,
):
    """
    result_type: "contextual" or "societal"
    """

    attribute_type_to_application_to_model_to_delta = {}

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(dict))

        for application in applications:
            for model_name in model_names:
                if result_type == "contextual":
                    file_name = get_file_contextual(application, attribute_type, model_name)
                    compute_fn = compute_contextual_results
                elif result_type == "societal":
                    file_name = get_file_societal(application, attribute_type, model_name)
                    compute_fn = compute_societal_results
                else:
                    raise ValueError(f"Unknown result_type: {result_type}")

                if not os.path.exists(file_name):
                    raise FileNotFoundError(
                        f"File not found: {application} {attribute_type} {model_name}\n{file_name}"
                    )

                delta = compute_fn(file_name, attribute_type)

                if delta is None:
                    delta = {
                        "delta": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                    }

                application_to_model_to_delta[application][model_name] = delta

        attribute_type_to_application_to_model_to_delta[attribute_type] = application_to_model_to_delta

    return attribute_type_to_application_to_model_to_delta


def draw_reasoning_block(
    fig,
    outer_spec,
    attribute_type_to_application_to_model_to_delta,
    attribute_types,
    applications,
    bias_type,
    adjusted_pvalue_lookup,
    panel_letter,
    block_title,
    ylabel,
    base_to_color,
    mode_to_marker,
):
    """
    Draw one 2 x 3 block:
        rows = attribute types
        columns = applications
    """

    panel_titles = {
        "edu": "Scholarship allocation",
        "hiring": "Hiring",
        "loan": "Loan approval",
    }

    pair_defs = REASONING_PAIR_DEFS
    model_order = [p[2] for p in pair_defs]

    # Smaller value = GLM and Nemotron closer together
    model_gap = 0.58
    model_to_x = {m: i * model_gap for i, m in enumerate(model_order)}

    # Keep reasoning/non-reasoning points separated within each model
    dodge = 0.11
    mode_to_dx = {
        "non-reasoning": -dodge,
        "reasoning": +dodge,
    }

    inner_gs = outer_spec.subgridspec(
        2,
        3,
        wspace=0.18,
        hspace=0.70,
    )

    axes = np.empty((2, 3), dtype=object)

    for row_idx, attribute_type in enumerate(attribute_types):
        application_to_model_to_delta = attribute_type_to_application_to_model_to_delta[attribute_type]

        for col_idx, application in enumerate(applications):
            ax = fig.add_subplot(inner_gs[row_idx, col_idx])
            # ax.set_box_aspect(0.65)
            axes[row_idx, col_idx] = ax

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            rows = []

            for base_name, no_think_name, short_label in pair_defs:
                d = application_to_model_to_delta[application][base_name]
                rows.append({
                    "model": short_label,
                    "mode": "reasoning",
                    "delta": float(d["delta"]),
                    "ci_low": float(d["ci_low"]),
                    "ci_high": float(d["ci_high"]),
                })

                d = application_to_model_to_delta[application][no_think_name]
                rows.append({
                    "model": short_label,
                    "mode": "non-reasoning",
                    "delta": float(d["delta"]),
                    "ci_low": float(d["ci_low"]),
                    "ci_high": float(d["ci_high"]),
                })

            # Store upper bounds and globally BH-adjusted P values for brackets.
            pair_upper = {m: -np.inf for m in model_order}
            pair_pvalues = {}

            for _, _, short_label in pair_defs:
                p_record = adjusted_pvalue_lookup[(
                    bias_type,
                    attribute_type,
                    application,
                    short_label,
                )]
                pair_pvalues[short_label] = p_record["p_adjusted"]

            all_ci_lows = []
            all_ci_highs = []

            for r in rows:
                x0 = model_to_x[r["model"]]
                x = x0 + mode_to_dx[r["mode"]]
                y = r["delta"]

                ci_low = r["ci_low"]
                ci_high = r["ci_high"]

                yerr_low = max(0.0, y - ci_low)
                yerr_high = max(0.0, ci_high - y)
                yerr = np.array([[yerr_low], [yerr_high]])

                pair_upper[r["model"]] = max(pair_upper[r["model"]], ci_high)
                all_ci_lows.append(ci_low)
                all_ci_highs.append(ci_high)

                ax.errorbar(
                    [x],
                    [y],
                    yerr=yerr,
                    fmt=mode_to_marker[r["mode"]],
                    markersize=4.5,
                    capsize=3.2,
                    elinewidth=1.0,
                    markeredgecolor=base_to_color[r["model"]],
                    markerfacecolor=base_to_color[r["model"]],
                    markeredgewidth=0.7,
                    color=base_to_color[r["model"]],
                    zorder=3,
                )

            # -------------------------------------------------------
            # Significance brackets
            # H1: reasoning has lower bias than non-reasoning
            # -------------------------------------------------------
            data_min = min(all_ci_lows)
            data_max = max(all_ci_highs)
            data_span = max(data_max - data_min, 0.02)

            bracket_offset = 0.08 * data_span
            bracket_height = 0.025 * data_span
            text_offset = 0.015 * data_span

            bracket_tops = []

            for model_short in model_order:
                x_left = model_to_x[model_short] + mode_to_dx["non-reasoning"]
                x_right = model_to_x[model_short] + mode_to_dx["reasoning"]

                y_bracket = pair_upper[model_short] + bracket_offset
                p_val = pair_pvalues[model_short]

                # Figure stars are based on the BH-adjusted P value.
                p_text = p_to_stars(p_val)

                add_sig_bracket(
                    ax=ax,
                    x1=x_left,
                    x2=x_right,
                    y=y_bracket,
                    text=p_text,
                    bar_height=bracket_height,
                    text_offset=text_offset,
                )

                bracket_tops.append(y_bracket + bracket_height + text_offset)

            # Ensure brackets are visible
            y_lower = min(0.0, data_min - 0.12 * data_span)
            y_upper = max(max(bracket_tops), data_max) + 0.18 * data_span
            ax.set_ylim(y_lower, y_upper)

            # if row_idx == 0:
            ax.set_title(
                panel_titles[application],
                fontsize=9,
                pad=2,
            )

            ax.set_xticks([model_to_x[m] for m in model_order])
            ax.set_xticklabels(model_order)

            ax.set_xlim(
                model_to_x[model_order[0]] - 0.28,
                model_to_x[model_order[-1]] + 0.28,
            )

            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v * 100:.0f}"))

            ax.tick_params(
                axis="both",
                which="major",
                # direction="out",
                length=0.0,
                width=0.0,
                # color="black",
                bottom=True,
                left=True,
                top=False,
                right=False,
            )
            ax.tick_params(axis="x", labelsize=9)

            for tick_label in ax.get_xticklabels():
                model_name = tick_label.get_text()
                tick_label.set_color(base_to_color[model_name])

            # ax.grid(
            #     axis="y",
            #     color="0.88",
            #     linewidth=0.6,
            #     linestyle="-",
            #     zorder=0,
            # )
            ax.set_axisbelow(True)

    # Geometry
    pos_top_left = axes[0, 0].get_position()
    pos_top_right = axes[0, 2].get_position()
    pos_bottom_left = axes[1, 0].get_position()

    block_x0 = pos_top_left.x0
    block_x1 = pos_top_right.x1
    block_y1 = pos_top_left.y1
    block_y0 = pos_bottom_left.y0
    block_x_center = (block_x0 + block_x1) / 2
    block_y_center = (block_y0 + block_y1) / 2

    # Panel letter and block title
    fig.text(
        block_x0 - 0.030,
        block_y1 + 0.055,
        panel_letter,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    fig.text(
        block_x0 + 0.005,
        block_y1 + 0.055,
        block_title,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    # Row subtitles
    row_title_offset = 0.026

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
            fontsize=9,
            fontweight="bold",
        )

    fig.text(
        block_x0 - 0.05,
        block_y_center,
        ylabel,
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
    )

    return axes


def draw_reasoning_super_figure(
    contextual_results,
    societal_results,
    output_dir="outputs/reasoning",
):
    set_nature_style()
    sns.set_theme(style="white")
    os.makedirs(output_dir, exist_ok=True)

    applications = ["hiring", "loan", "edu"]

    contextual_attribute_types = ["Gender", "Race"]
    societal_attribute_types = ["Gender Identity", "Sexual Orientation"]

    # Compute the 24 raw one-sided P values, then apply one joint BH correction
    # across the entire Figure 8 family. All significance stars below use the
    # adjusted P values.
    adjusted_pvalue_lookup, pvalue_records = compute_reasoning_pvalue_results(
        societal_results=societal_results,
        contextual_results=contextual_results,
        applications=applications,
        alpha=0.05,
    )

    statistics_path = os.path.join(
        output_dir,
        "figure8_reasoning_statistics_bh_adjusted.csv",
    )
    save_reasoning_pvalue_results(pvalue_records, statistics_path)

    print(
        "Benjamini-Hochberg adjustment applied jointly across all 24 "
        "Figure 8 tests (FDR = 0.05)."
    )
    for record in pvalue_records:
        print(
            f"{record['bias_type']} | {record['attribute']} | "
            f"{record['application']} | {record['model']}: "
            f"raw P={record['p_raw']:.4g}, "
            f"BH-adjusted P={record['p_adjusted']:.4g}, "
            f"{record['significance']}"
        )
    print(f"Saved: {statistics_path}")

    # Professional, colorblind-safe colors
    base_to_color = {
        "GLM": "#FB6542",
        "Nemotron": "#2673EF",
    }

    mode_to_marker = {
        "reasoning": "o",
        "non-reasoning": "^",
    }

    fig = plt.figure(figsize=(7.45, 6.5))

    outer_gs = fig.add_gridspec(
        2,
        1,
        left=0.115,
        right=0.995,
        bottom=0.165,
        top=0.925,
        hspace=0.38,
    )

    draw_reasoning_block(
        fig=fig,
        outer_spec=outer_gs[0],
        attribute_type_to_application_to_model_to_delta=societal_results,
        attribute_types=societal_attribute_types,
        applications=applications,
        bias_type="societal",
        adjusted_pvalue_lookup=adjusted_pvalue_lookup,
        panel_letter="a",
        block_title="Societal minority bias vs. reasoning",
        ylabel="Relative score difference (%)",
        base_to_color=base_to_color,
        mode_to_marker=mode_to_marker,
    )

    draw_reasoning_block(
        fig=fig,
        outer_spec=outer_gs[1],
        attribute_type_to_application_to_model_to_delta=contextual_results,
        attribute_types=contextual_attribute_types,
        applications=applications,
        bias_type="contextual",
        adjusted_pvalue_lookup=adjusted_pvalue_lookup,
        panel_letter="b",
        block_title="Contextual minority bias vs. reasoning",
        ylabel="Absolute candidate-level selection-rate difference (pp)",
        base_to_color=base_to_color,
        mode_to_marker=mode_to_marker,
    )

    # One shared legend below the whole figure
    legend_handles = [
        # text-only entries for model names
        Line2D(
            [0], [0],
            linestyle="",
            marker=None,
            color="#f599a1",
            label="GLM: GLM-4.5-Air",
        ),
        Line2D(
            [0], [0],
            linestyle="",
            marker=None,
            color="#a577ad",
            label="Nemotron: Nemotron-Nano-12B-v2",
        ),

        # marker entries for reasoning modes
        Line2D(
            [0], [0],
            marker="^",
            linestyle="",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6.0,
            label="Non-reasoning",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6.0,
            label="Reasoning",
        ),
    ]

    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=4,
        frameon=False,
        fontsize=9,
        handletextpad=0.45,
        columnspacing=1.35,
    )

    # Hide the first two dummy handles
    for h in leg.legend_handles[:2]:
        h.set_visible(False)

    # Color the first two legend texts
    legend_texts = leg.get_texts()
    legend_texts[0].set_color("#FB6542")   # GLM text
    legend_texts[1].set_color("#2673EF")   # Nemotron text

    pdf_path = os.path.join(output_dir, "reasoning_contextual_societal_super_figure.pdf")

    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {pdf_path}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    applications = ["edu", "hiring", "loan"]

    model_names = [
        "GLM-4.5-Air",
        "GLM-4.5-Air_no_thinking",
        "NVIDIA-Nemotron-Nano-12B-v2",
        "NVIDIA-Nemotron-Nano-12B-v2_no_thinking",
    ]

    contextual_attribute_types = ["Gender", "Race"]
    societal_attribute_types = ["Gender Identity", "Sexual Orientation"]

    contextual_results = collect_results(
        applications=applications,
        attribute_types=contextual_attribute_types,
        model_names=model_names,
        result_type="contextual",
    )

    societal_results = collect_results(
        applications=applications,
        attribute_types=societal_attribute_types,
        model_names=model_names,
        result_type="societal",
    )

    draw_reasoning_super_figure(
        contextual_results=contextual_results,
        societal_results=societal_results,
        output_dir="outputs/reasoning",
    )