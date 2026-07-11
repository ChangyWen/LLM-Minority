import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import binomtest, norm


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}

FONT_SIZE = 9.5
LABEL_SIZE = 8.0
MARKER_SIZE = 5.0
LINE_WIDTH = 0.95
LINE_ALPHA = 0.82


# ============================================================
# Shared style
# ============================================================

def set_nature_style():
    """
    Compact, clean plotting style.
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
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.4,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": LINE_WIDTH,
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
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_")


def format_percentage_point_tick(v, pos):
    return f"{v:.0f}"


# ============================================================
# Statistical helpers
# ============================================================

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)

    p = k / n
    denominator = 1 + (z ** 2) / n
    centre = p + (z ** 2) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return (lower, upper)


# ============================================================
# Data computation
# ============================================================

def compute_results(file_name, attribute_type, max_n_trials=100000):
    """
    Compute overall selection rates for societal minority and societal majority.

    The candidate pool is balanced between societal minority and societal majority.
    Therefore, each selected candidate is classified as either Minority or Majority.

    The binomial test evaluates:

        H0: P(select Minority) = 0.5
        H1: P(select Minority) != 0.5
    """
    attr_value_to_hit_count = defaultdict(int)
    n_trials = 0

    with open(file_name, "r") as f:
        for line in f:
            item = json.loads(line)
            attributes = item["attributes"]

            # Keep this guard for compatibility with your previous processing.
            if "Asian" in attributes:
                continue
            if isinstance(attributes, dict) and "Asian" in attributes.values():
                continue

            if n_trials >= max_n_trials:
                break

            suggested_candidate_id = item["suggested_candidate_id"]
            suggested_candidate_attr_value = attributes[suggested_candidate_id]

            minority_attr_values = type_to_minority_attributes[attribute_type]
            if suggested_candidate_attr_value in minority_attr_values:
                group = "Minority"
            else:
                group = "Majority"

            attr_value_to_hit_count[group] += 1
            n_trials += 1

    if n_trials == 0:
        raise ValueError(f"No valid trials found in file: {file_name}")

    minority_count = attr_value_to_hit_count["Minority"]
    majority_count = attr_value_to_hit_count["Majority"]

    results = {}

    for group, hit_count in [
        ("Minority", minority_count),
        ("Majority", majority_count),
    ]:
        hit_rate = hit_count / n_trials
        ci_low, ci_high = wilson_ci(hit_count, n_trials)

        results[group] = {
            "hit_count": hit_count,
            "total_count": n_trials,
            "hit_rate": hit_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    test = binomtest(
        k=minority_count,
        n=n_trials,
        p=0.5,
        alternative="two-sided",
    )

    results["significance"] = {
        "p_value": float(test.pvalue),
        "minority_count": minority_count,
        "majority_count": majority_count,
        "total_count": n_trials,
    }

    return results


def compute_delta_results(file_name, attribute_type, max_n_trials=100000):
    """
    Compute Delta in selection rate:

        Delta = SelectionRate_Minority - SelectionRate_Majority

    Since Minority + Majority = 1 in this balanced binary comparison,

        Delta = 2 * SelectionRate_Minority - 1

    Returned values are fractions, not percentage points.
    """
    results = compute_results(
        file_name=file_name,
        attribute_type=attribute_type,
        max_n_trials=max_n_trials,
    )

    p_min = results["Minority"]["hit_rate"]
    ci_min_low = results["Minority"]["ci_low"]
    ci_min_high = results["Minority"]["ci_high"]

    delta = 2.0 * p_min - 1.0
    delta_ci_low = 2.0 * ci_min_low - 1.0
    delta_ci_high = 2.0 * ci_min_high - 1.0

    return {
        "delta": delta,
        "ci_low": delta_ci_low,
        "ci_high": delta_ci_high,
        "p_value": results["significance"]["p_value"],
        "minority_count": results["significance"]["minority_count"],
        "majority_count": results["significance"]["majority_count"],
        "total_count": results["significance"]["total_count"],
    }


def collect_delta_data(
    attribute_types,
    applications,
    model_names,
    resume_counts,
    application_to_pool_count,
    max_n_trials=1000000,
    strict=True,
):
    """
    Load all data needed for the 2 x 3 delta figure.

    Delta values and confidence intervals are stored in percentage points.
    Raw minority/majority selection counts are retained for the formal
    behavior-classification tests.
    """
    delta_data = {}

    for attribute_type in attribute_types:
        for application in applications:
            pool_count = application_to_pool_count[application]

            for model_name in model_names:
                deltas = []
                ci_lows = []
                ci_highs = []
                p_values = []
                minority_counts = []
                majority_counts = []
                total_counts = []

                for rc in resume_counts:
                    file_path = (
                        f"outputs/{application}/contextual/"
                        f"{attribute_type}/{model_name}_{rc}_{pool_count}.jsonl"
                    )

                    if not os.path.exists(file_path):
                        message = f"File not found: {file_path}"
                        if strict:
                            raise FileNotFoundError(message)
                        print(f"[Warning] {message}")

                        deltas.append(np.nan)
                        ci_lows.append(np.nan)
                        ci_highs.append(np.nan)
                        p_values.append(np.nan)
                        minority_counts.append(np.nan)
                        majority_counts.append(np.nan)
                        total_counts.append(np.nan)
                        continue

                    result = compute_delta_results(
                        file_name=file_path,
                        attribute_type=attribute_type,
                        max_n_trials=max_n_trials,
                    )

                    # Convert effect estimates to percentage points.
                    deltas.append(result["delta"] * 100.0)
                    ci_lows.append(result["ci_low"] * 100.0)
                    ci_highs.append(result["ci_high"] * 100.0)
                    p_values.append(result["p_value"])

                    # Retain counts for line-level statistical classification.
                    minority_counts.append(result["minority_count"])
                    majority_counts.append(result["majority_count"])
                    total_counts.append(result["total_count"])

                delta_data[(attribute_type, application, model_name)] = {
                    "x": np.asarray(resume_counts, dtype=float),
                    "delta": np.asarray(deltas, dtype=float),
                    "ci_low": np.asarray(ci_lows, dtype=float),
                    "ci_high": np.asarray(ci_highs, dtype=float),
                    "p_value": np.asarray(p_values, dtype=float),
                    "minority_count": np.asarray(minority_counts, dtype=float),
                    "majority_count": np.asarray(majority_counts, dtype=float),
                    "total_count": np.asarray(total_counts, dtype=float),
                }

    return delta_data

def compute_subplot_ylim(
    delta_data,
    attribute_type,
    application,
    model_names,
    include_ci=False,
    step=2.0,
    min_abs=2.0,
    pad_frac=0.12,
    symmetric=True,
):
    """
    Compute a y-axis range for one subplot only.

    If symmetric=True, the y-axis is centered at 0:
        (-upper, upper)

    This is recommended for Delta plots because 0 is the parity baseline.
    """
    values = []

    for model_name in model_names:
        item = delta_data[(attribute_type, application, model_name)]

        if include_ci:
            values.extend(item["ci_low"])
            values.extend(item["ci_high"])
        else:
            values.extend(item["delta"])

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (-min_abs, min_abs)

    if symmetric:
        abs_max = np.max(np.abs(values))
        abs_max = abs_max * (1.0 + pad_frac)
        upper = max(min_abs, math.ceil(abs_max / step) * step)
        return (-upper, upper)

    # Optional non-symmetric version.
    vmin = np.min(values)
    vmax = np.max(values)

    span = max(vmax - vmin, 1e-8)
    vmin = min(vmin - pad_frac * span, 0.0)
    vmax = max(vmax + pad_frac * span, 0.0)

    lower = math.floor(vmin / step) * step
    upper = math.ceil(vmax / step) * step

    return (lower, upper)


# ============================================================
# Plotting
# ============================================================

# Semantic colors for the statistically supported overall preference.
# Markers/linestyles identify models. Reversal is retained as a separate
# statistical result in the CSV but is not a color category.
BEHAVIOR_COLORS = {
    "Minority favored": "#1B9E77",
    "Majority favored": "#C9253C",
    "No statistically supported overall preference": "#9E9E9E",
}

BEHAVIOR_ORDER = [
    "Minority favored",
    "Majority favored",
    "No statistically supported overall preference",
]

CLASSIFICATION_ALPHA = 0.05

MINORITY_BG = "#EEF7F4"
MAJORITY_BG = "#FBF0F2"

# Manual y-axis limits for panels that need extra space for annotations.
# Key format: (attribute_type, application).
PANEL_YLIM_OVERRIDES = {
    ("Gender Identity", "hiring"): (-11.0, 11.0),
}


def build_model_styles(model_names):
    """
    Assign each model a distinct marker/linestyle.

    In the revised figure, color encodes the behavioral pattern rather than
    model identity. Model identity is encoded by marker and line style.
    """
    markers = ["^", "v", "<", ">", "o", "*", "d", "X"]
    linestyles = [
        "solid",
        "solid",
        "dotted",
        "dotted",
        "dashed",
        "dashdot",
        (0, (3, 1, 1, 1)), # densely dashdotted
        (0, (3,5,1,5,1,5)), # dashdotdotted
    ]

    model_to_style = {}
    for idx, model_name in enumerate(model_names):
        model_to_style[model_name] = {
            "marker": markers[idx % len(markers)],
            "linestyle": linestyles[idx % len(linestyles)],
        }

    return model_to_style


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg false-discovery-rate adjustment.

    NaN values are ignored and remain NaN in the returned array.
    """
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)

    valid_mask = np.isfinite(p_values)
    valid_p = p_values[valid_mask]

    if valid_p.size == 0:
        return adjusted

    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    m = ranked_p.size

    ranked_adjusted = np.empty(m, dtype=float)
    running_min = 1.0

    for idx in range(m - 1, -1, -1):
        rank = idx + 1
        candidate = ranked_p[idx] * m / rank
        running_min = min(running_min, candidate)
        ranked_adjusted[idx] = min(running_min, 1.0)

    valid_adjusted = np.empty(m, dtype=float)
    valid_adjusted[order] = ranked_adjusted
    adjusted[valid_mask] = valid_adjusted

    return adjusted


def cochran_armitage_trend_test(x, minority_count, total_count, alternative):
    """
    Cochran-Armitage score test for a monotonic trend in the probability of
    selecting a societal-minority candidate as pool size increases.

    The intercept is estimated under the null of no pool-size trend. This is
    equivalent to the score test for the slope in a grouped-binomial logistic
    regression with pool size entered as a continuous predictor.

    Parameters
    ----------
    alternative : {"greater", "less", "two-sided"}
        "greater" tests for an increasing minority-selection probability as
        pool size grows; "less" tests for a decreasing probability.
    """
    x = np.asarray(x, dtype=float)
    minority_count = np.asarray(minority_count, dtype=float)
    total_count = np.asarray(total_count, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(minority_count)
        & np.isfinite(total_count)
        & (total_count > 0)
    )

    x = x[valid]
    minority_count = minority_count[valid]
    total_count = total_count[valid]

    if x.size < 2 or np.allclose(x, x[0]):
        return {"z": 0.0, "p_value": 1.0}

    pooled_probability = minority_count.sum() / total_count.sum()
    weighted_x_mean = np.average(x, weights=total_count)

    centered_x = x - weighted_x_mean
    score = np.sum(centered_x * (minority_count - total_count * pooled_probability))
    variance = (
        pooled_probability
        * (1.0 - pooled_probability)
        * np.sum(total_count * centered_x ** 2)
    )

    if not np.isfinite(variance) or variance <= 0:
        return {"z": 0.0, "p_value": 1.0}

    z_value = score / math.sqrt(variance)

    if alternative == "greater":
        p_value = norm.sf(z_value)
    elif alternative == "less":
        p_value = norm.cdf(z_value)
    elif alternative == "two-sided":
        p_value = 2.0 * norm.sf(abs(z_value))
    else:
        raise ValueError(
            "alternative must be 'greater', 'less', or 'two-sided'"
        )

    return {
        "z": float(z_value),
        "p_value": float(min(max(p_value, 0.0), 1.0)),
    }


def equal_weight_direction_test(minority_count, total_count):
    """
    Test the equal-weight mean minority-selection probability across pool sizes.

    Each pool size contributes equally, regardless of its number of trials.
    Under the global parity null, p_N = 0.5 at every pool size. Independence
    across trials and pool-size settings gives

        Var(mean(p_hat_N)) = (1 / m^2) * sum_N 0.25 / n_N.

    The resulting score test is conservative because 0.25 is the maximum
    possible Bernoulli variance.

    Returns a two-sided p-value. The sign of the estimated average delta
    determines whether the supported direction favors the societal minority or
    societal majority.
    """
    minority_count = np.asarray(minority_count, dtype=float)
    total_count = np.asarray(total_count, dtype=float)

    valid = (
        np.isfinite(minority_count)
        & np.isfinite(total_count)
        & (total_count > 0)
    )

    minority_count = minority_count[valid]
    total_count = total_count[valid]

    if minority_count.size == 0:
        return {
            "mean_probability": np.nan,
            "mean_delta_pp": np.nan,
            "z": np.nan,
            "p_value": np.nan,
        }

    probabilities = minority_count / total_count
    mean_probability = float(np.mean(probabilities))
    m = probabilities.size

    standard_error_null = math.sqrt(
        np.sum(0.25 / total_count)
    ) / m

    if standard_error_null <= 0 or not np.isfinite(standard_error_null):
        z_value = 0.0
        p_value = 1.0
    else:
        z_value = (mean_probability - 0.5) / standard_error_null
        p_value = 2.0 * norm.sf(abs(z_value))

    return {
        "mean_probability": mean_probability,
        "mean_delta_pp": 100.0 * (2.0 * mean_probability - 1.0),
        "z": float(z_value),
        "p_value": float(min(max(p_value, 0.0), 1.0)),
    }


def compute_line_classification_statistics(item):
    """
    Compute the two prespecified line-level hypotheses.

    1. Reversal hypothesis
       A reversal requires:
       - opposite endpoint directions at the smallest and largest pool sizes;
       - an exact one-sided binomial test supporting each endpoint direction;
       - a one-sided Cochran-Armitage trend test supporting the corresponding
         direction of change.

       These are joint requirements. Their intersection-union p-value is the
       maximum of the three component p-values.

    2. Overall-direction hypothesis
       The equal-weight average minority-selection probability across pool
       sizes is tested against 0.5.

    Multiple-testing adjustment and final category assignment are performed
    later, jointly across all plotted model lines.
    """
    x = np.asarray(item["x"], dtype=float)
    minority_count = np.asarray(item["minority_count"], dtype=float)
    total_count = np.asarray(item["total_count"], dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(minority_count)
        & np.isfinite(total_count)
        & (total_count > 0)
    )

    x = x[valid]
    minority_count = minority_count[valid]
    total_count = total_count[valid]

    if x.size < 2:
        return {
            "reversal_direction": "none",
            "endpoint_start_delta_pp": np.nan,
            "endpoint_end_delta_pp": np.nan,
            "endpoint_start_p": np.nan,
            "endpoint_end_p": np.nan,
            "trend_z": np.nan,
            "trend_p": np.nan,
            "reversal_p_raw": np.nan,
            "mean_probability": np.nan,
            "mean_delta_pp": np.nan,
            "direction_z": np.nan,
            "direction_p_raw": np.nan,
        }

    order = np.argsort(x)
    x = x[order]
    minority_count = minority_count[order]
    total_count = total_count[order]

    probabilities = minority_count / total_count

    start_probability = float(probabilities[0])
    end_probability = float(probabilities[-1])
    start_delta_pp = 100.0 * (2.0 * start_probability - 1.0)
    end_delta_pp = 100.0 * (2.0 * end_probability - 1.0)

    reversal_direction = "none"
    endpoint_start_p = 1.0
    endpoint_end_p = 1.0
    trend_result = {"z": 0.0, "p_value": 1.0}
    reversal_p_raw = 1.0

    if start_probability > 0.5 and end_probability < 0.5:
        reversal_direction = "minority_to_majority"

        endpoint_start_p = binomtest(
            k=int(minority_count[0]),
            n=int(total_count[0]),
            p=0.5,
            alternative="greater",
        ).pvalue

        endpoint_end_p = binomtest(
            k=int(minority_count[-1]),
            n=int(total_count[-1]),
            p=0.5,
            alternative="less",
        ).pvalue

        trend_result = cochran_armitage_trend_test(
            x=x,
            minority_count=minority_count,
            total_count=total_count,
            alternative="less",
        )

        reversal_p_raw = max(
            float(endpoint_start_p),
            float(endpoint_end_p),
            float(trend_result["p_value"]),
        )

    elif start_probability < 0.5 and end_probability > 0.5:
        reversal_direction = "majority_to_minority"

        endpoint_start_p = binomtest(
            k=int(minority_count[0]),
            n=int(total_count[0]),
            p=0.5,
            alternative="less",
        ).pvalue

        endpoint_end_p = binomtest(
            k=int(minority_count[-1]),
            n=int(total_count[-1]),
            p=0.5,
            alternative="greater",
        ).pvalue

        trend_result = cochran_armitage_trend_test(
            x=x,
            minority_count=minority_count,
            total_count=total_count,
            alternative="greater",
        )

        reversal_p_raw = max(
            float(endpoint_start_p),
            float(endpoint_end_p),
            float(trend_result["p_value"]),
        )

    direction_result = equal_weight_direction_test(
        minority_count=minority_count,
        total_count=total_count,
    )

    return {
        "reversal_direction": reversal_direction,
        "endpoint_start_delta_pp": start_delta_pp,
        "endpoint_end_delta_pp": end_delta_pp,
        "endpoint_start_p": float(endpoint_start_p),
        "endpoint_end_p": float(endpoint_end_p),
        "trend_z": float(trend_result["z"]),
        "trend_p": float(trend_result["p_value"]),
        "reversal_p_raw": float(reversal_p_raw),
        "mean_probability": float(direction_result["mean_probability"]),
        "mean_delta_pp": float(direction_result["mean_delta_pp"]),
        "direction_z": float(direction_result["z"]),
        "direction_p_raw": float(direction_result["p_value"]),
    }


def classify_all_model_lines(
    delta_data,
    attribute_types,
    applications,
    model_names,
    alpha=CLASSIFICATION_ALPHA,
):
    """
    Classify all plotted model lines by their overall directional preference.

    The plotted behavior category is determined only by the equal-weight
    overall-direction test:

      - a significantly positive mean difference is "Minority favored";
      - a significantly negative mean difference is "Majority favored";
      - otherwise, the line is "No statistically supported overall preference".

    Benjamini-Hochberg adjustment is applied across all line-level
    overall-direction tests. With the default figure design, this family
    contains 48 tests (8 models x 3 applications x 2 attributes), provided
    that all tests return finite p-values.

    Reversal statistics are still computed, adjusted in their own 48-test
    family, and exported for supplementary interpretation. They do not affect
    the plotted behavior category or line color.

    No manually selected effect-size threshold is used.
    """
    classification_results = {}
    ordered_keys = []

    for attribute_type in attribute_types:
        for application in applications:
            for model_name in model_names:
                key = (attribute_type, application, model_name)
                ordered_keys.append(key)
                classification_results[key] = compute_line_classification_statistics(
                    delta_data[key]
                )

    # The reversal and overall-direction hypotheses answer different questions,
    # so their p-values are adjusted in separate Benjamini-Hochberg families.
    # Only the adjusted direction test is used for the behavior category.
    reversal_raw_p_values = np.asarray(
        [
            classification_results[key]["reversal_p_raw"]
            for key in ordered_keys
        ],
        dtype=float,
    )
    direction_raw_p_values = np.asarray(
        [
            classification_results[key]["direction_p_raw"]
            for key in ordered_keys
        ],
        dtype=float,
    )

    reversal_adjusted = benjamini_hochberg(reversal_raw_p_values)
    direction_adjusted = benjamini_hochberg(direction_raw_p_values)

    reversal_family_size = int(np.isfinite(reversal_raw_p_values).sum())
    direction_family_size = int(np.isfinite(direction_raw_p_values).sum())

    for idx, key in enumerate(ordered_keys):
        result = classification_results[key]

        result["reversal_p_adjusted"] = float(reversal_adjusted[idx])
        result["direction_p_adjusted"] = float(direction_adjusted[idx])

        # Retained for reporting only; this flag does not determine line color.
        reversal_significant = (
            result["reversal_direction"] != "none"
            and np.isfinite(result["reversal_p_adjusted"])
            and result["reversal_p_adjusted"] < alpha
        )

        direction_significant = (
            np.isfinite(result["direction_p_adjusted"])
            and result["direction_p_adjusted"] < alpha
        )

        # Assign the plotted category solely from the overall-direction test.
        if direction_significant and result["mean_delta_pp"] > 0:
            behavior = "Minority favored"
        elif direction_significant and result["mean_delta_pp"] < 0:
            behavior = "Majority favored"
        else:
            behavior = "No statistically supported overall preference"

        result["reversal_significant"] = bool(reversal_significant)
        result["direction_significant"] = bool(direction_significant)
        result["classification_alpha"] = float(alpha)
        result["classification_basis"] = (
            "Overall-direction test only; reversal does not determine class"
        )
        result["multiple_testing_method"] = (
            "Benjamini-Hochberg; separate reversal and direction families"
        )
        result["reversal_bh_family_size"] = reversal_family_size
        result["direction_bh_family_size"] = direction_family_size
        result["behavior"] = behavior

    return classification_results


def export_classification_results(
    classification_results,
    output_path,
):
    """
    Save all raw and adjusted statistics used for behavior assignment.
    """
    fieldnames = [
        "attribute_type",
        "application",
        "model_name",
        "behavior",
        "classification_alpha",
        "classification_basis",
        "multiple_testing_method",
        "reversal_bh_family_size",
        "direction_bh_family_size",
        "reversal_significant",
        "direction_significant",
        "mean_probability",
        "mean_delta_pp",
        "direction_z",
        "direction_p_raw",
        "direction_p_adjusted",
        "reversal_direction",
        "endpoint_start_delta_pp",
        "endpoint_end_delta_pp",
        "endpoint_start_p",
        "endpoint_end_p",
        "trend_z",
        "trend_p",
        "reversal_p_raw",
        "reversal_p_adjusted",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for (
            attribute_type,
            application,
            model_name,
        ), result in classification_results.items():
            row = {
                "attribute_type": attribute_type,
                "application": application,
                "model_name": model_name,
            }

            for field in fieldnames[3:]:
                row[field] = result.get(field, "")

            writer.writerow(row)

def compute_subplot_ylim_reviewer(
    delta_data,
    attribute_type,
    application,
    model_names,
    include_ci=False,
    step=2.5,
    min_pos=2.0,
    min_neg=2.0,
    pad_frac=0.12,
):
    """
    Compute reviewer-style, non-symmetric y-limits for one subplot.

    The original delta figure used symmetric y-limits around zero. The reviewer
    illustration uses non-symmetric ranges to save space, while keeping zero
    visible and leaving enough room for both positive and negative shaded zones.
    """
    values = []

    for model_name in model_names:
        item = delta_data[(attribute_type, application, model_name)]
        if include_ci:
            values.extend(item["ci_low"])
            values.extend(item["ci_high"])
        else:
            values.extend(item["delta"])

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return (-min_neg, min_pos)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, 1.0)

    lower = min(vmin - pad_frac * span, -min_neg)
    upper = max(vmax + pad_frac * span, min_pos)

    lower = math.floor(lower / step) * step
    upper = math.ceil(upper / step) * step

    if lower >= 0:
        lower = -min_neg
    if upper <= 0:
        upper = min_pos

    return (lower, upper)


def plot_delta_application_panel(
    ax,
    attribute_type,
    application,
    model_names,
    delta_data,
    model_to_style,
    resume_counts,
    ylim,
    show_errorbars=False,
    classification_results=None,
    annotate_favored_regions=False,
):
    """
    Draw one subplot: one attribute x one application.

    Each line corresponds to one model. Line color encodes the statistically
    supported overall preference; marker/linestyle encodes the model.
    """
    xs = np.asarray(resume_counts, dtype=float)

    ax.set_ylim(*ylim)

    # Positive/negative regions.
    ax.axhspan(0, ylim[1], color=MINORITY_BG, zorder=0)
    ax.axhspan(ylim[0], 0, color=MAJORITY_BG, zorder=0)

    # Horizontal parity line.
    ax.axhline(
        0,
        color="0.35",
        linewidth=0.8,
        linestyle=(0, (3.0, 2.2)),
        zorder=2,
    )

    for model_name in model_names:
        item = delta_data[(attribute_type, application, model_name)]

        y = item["delta"]
        ci_low = item["ci_low"]
        ci_high = item["ci_high"]
        valid = np.isfinite(y)

        if not np.any(valid):
            continue

        if classification_results is None:
            raise ValueError(
                "classification_results must be provided for statistical "
                "behavior assignment."
            )

        key = (attribute_type, application, model_name)
        behavior = classification_results[key]["behavior"]
        color = BEHAVIOR_COLORS[behavior]
        style = model_to_style[model_name]

        if show_errorbars:
            yerr = np.vstack([
                np.maximum(0.0, y - ci_low),
                np.maximum(0.0, ci_high - y),
            ])

            ax.errorbar(
                xs[valid],
                y[valid],
                yerr=yerr[:, valid],
                fmt=style["marker"],
                linestyle=style["linestyle"],
                color=color,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.35,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                elinewidth=0.45,
                capsize=1.6,
                capthick=0.45,
                alpha=0.95,
                zorder=3,
            )
        else:
            ax.plot(
                xs[valid],
                y[valid],
                linestyle=style["linestyle"],
                linewidth=LINE_WIDTH,
                color=color,
                marker=style["marker"],
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.35,
                markersize=MARKER_SIZE,
                alpha=0.95,
                zorder=3,
            )

    # Axes formatting.
    ax.set_xlim(xs[0] - 0.40, xs[-1] + 0.40)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(format_percentage_point_tick))

    ax.tick_params(
        axis="x",
        length=0.0,
        width=0.0,
        labelsize=LABEL_SIZE,
        bottom=True,
        labelbottom=True,
    )

    ax.tick_params(
        axis="y",
        length=0.0,
        width=0.0,
        labelsize=LABEL_SIZE,
        left=True,
        labelleft=True,
    )

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color("0.15")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_axisbelow(True)

    if annotate_favored_regions:
        ax.text(
            0.06,
            0.90,
            "Societal minority favored",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=LABEL_SIZE + 0.4,
            # fontweight="bold",
            fontstyle="italic",
            color=BEHAVIOR_COLORS["Minority favored"],
        )
        ax.text(
            0.06,
            0.09,
            "Societal majority favored",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=LABEL_SIZE + 0.4,
            # fontweight="bold",
            fontstyle="italic",
            color=BEHAVIOR_COLORS["Majority favored"],
        )


def draw_combined_delta_figure(
    model_names,
    applications,
    resume_counts,
    application_to_pool_count,
    max_n_trials=1000000,
    output_dir="outputs/societal",
    show_errorbars=False,
    classification_alpha=CLASSIFICATION_ALPHA,
):
    """
    Draw the reviewer-style revised Figure 3.

    Layout:
        Rows:    gender identity, sexual orientation
        Columns: hiring, loan approval, scholarship allocation

    Y-axis:
        Delta selection rate in percentage points:
        Delta = SelectionRate_Minority - SelectionRate_Majority

    Visual encoding:
        Color:       statistically supported overall preference
        Marker/line: model identity
        Background:  positive = societal minority favored;
                     negative = societal majority favored
    """
    set_nature_style()
    os.makedirs(output_dir, exist_ok=True)

    attribute_types = ["Gender Identity", "Sexual Orientation"]

    attribute_title_map = {
        "Gender Identity": "Gender identity",
        "Sexual Orientation": "Sexual orientation",
    }

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship allocation",
    }

    application_subtitle_map = {
        "hiring": "Mixed, model-dependent\npreferences",
        "loan": "Predominantly majority favored,\noften stronger in larger pools",
        "edu": "Predominantly minority favored,\ngenerally persistent across pool sizes",
    }

    model_to_style = build_model_styles(model_names)

    delta_data = collect_delta_data(
        attribute_types=attribute_types,
        applications=applications,
        model_names=model_names,
        resume_counts=resume_counts,
        application_to_pool_count=application_to_pool_count,
        max_n_trials=max_n_trials,
        strict=True,
    )

    classification_results = classify_all_model_lines(
        delta_data=delta_data,
        attribute_types=attribute_types,
        applications=applications,
        model_names=model_names,
        alpha=classification_alpha,
    )

    classification_csv_path = os.path.join(
        output_dir,
        "Figure3_behavior_classification_statistics.csv",
    )
    export_classification_results(
        classification_results=classification_results,
        output_path=classification_csv_path,
    )

    # Reserve four separate vertical bands below the panels:
    # plots -> shared x-axis label -> behavior legend -> model legend.
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.4, 7.0),
        sharex=True,
        sharey=False,
    )

    plt.subplots_adjust(
        left=0.125,
        right=0.985,
        top=0.820,
        bottom=0.340,
        wspace=0.12,
        hspace=0.28,
    )

    for row_idx, attribute_type in enumerate(attribute_types):
        for col_idx, application in enumerate(applications):
            ax = axes[row_idx, col_idx]

            subplot_ylim = compute_subplot_ylim_reviewer(
                delta_data=delta_data,
                attribute_type=attribute_type,
                application=application,
                model_names=model_names,
                include_ci=show_errorbars,
                step=2.5,
                min_pos=2.0,
                min_neg=2.0,
                pad_frac=0.12,
            )

            subplot_ylim = PANEL_YLIM_OVERRIDES.get(
                (attribute_type, application),
                subplot_ylim,
            )

            plot_delta_application_panel(
                ax=ax,
                attribute_type=attribute_type,
                application=application,
                model_names=model_names,
                delta_data=delta_data,
                model_to_style=model_to_style,
                resume_counts=resume_counts,
                ylim=subplot_ylim,
                show_errorbars=show_errorbars,
                classification_results=classification_results,
                annotate_favored_regions=(row_idx == 0 and col_idx == 0),
            )

            if row_idx == 0:
                ax.set_title(
                    application_title_map[application],
                    fontsize=FONT_SIZE + 2.0,
                    fontweight="bold",
                    pad=47,
                )

                ax.text(
                    0.5,
                    1.155,
                    application_subtitle_map[application],
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZE - 0.8,
                    fontstyle="italic",
                    color="0.35",
                    linespacing=1.05,
                )

    plot_left = min(ax.get_position().x0 for ax in axes.flat)
    plot_right = max(ax.get_position().x1 for ax in axes.flat)
    plot_center_x = 0.5 * (plot_left + plot_right)

    # Shared axis labels.
    fig.supxlabel(
        "Number of candidates in pool",
        fontsize=FONT_SIZE + 0.3,
        x=plot_center_x,
        y=0.285,
    )

    plot_bottom = min(ax.get_position().y0 for ax in axes.flat)
    plot_top = max(ax.get_position().y1 for ax in axes.flat)
    plot_center_y = 0.5 * (plot_bottom + plot_top)

    fig.supylabel(
        "Group-level selection-rate difference: minority − majority (pp)",
        fontsize=FONT_SIZE + 0.3,
        x=0.054,
        y=plot_center_y,
    )

    # Row labels.
    for row_idx, row_label in enumerate(["Gender identity", "Sexual orientation"]):
        row_pos = axes[row_idx, 0].get_position()
        row_center_y = 0.5 * (row_pos.y0 + row_pos.y1)
        fig.text(
            0.09,
            row_center_y,
            row_label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=FONT_SIZE + 0.5,
            fontweight="bold",
        )

    # Overall-preference legend: semantic color meaning.
    behavior_handles = [
        Line2D(
            [0],
            [0],
            color=BEHAVIOR_COLORS[name],
            linestyle="-",
            linewidth=2.2,
            label=name,
        )
        for name in BEHAVIOR_ORDER
    ]

    fig.legend(
        handles=behavior_handles,
        loc="lower center",
        bbox_to_anchor=(plot_center_x, 0.19),
        ncol=3,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.35,
        handletextpad=0.55,
        title="Overall preference (color)",
        title_fontsize=FONT_SIZE + 0.3,
        fontsize=FONT_SIZE - 0.3,
    )

    # Model legend: marker and line style meaning.
    model_handles = []
    for model_name in model_names:
        style = model_to_style[model_name]
        model_handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=LINE_WIDTH + 0.25,
                markersize=MARKER_SIZE + 2.0,
                markerfacecolor="0.35",
                markeredgecolor="white",
                markeredgewidth=0.35,
                label=pretty_model_name(model_name),
            )
        )

    fig.legend(
        handles=model_handles,
        loc="lower center",
        bbox_to_anchor=(plot_center_x, 0.09),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.35,
        handletextpad=0.55,
        title="Model (linestyle and marker)",
        title_fontsize=FONT_SIZE + 0.3,
        fontsize=FONT_SIZE - 0.3,
    )

    base = "Figure3_societal_cross_candidate_delta_reviewer_style"
    pdf_path = os.path.join(output_dir, base + ".pdf")
    png_path = os.path.join(output_dir, base + ".png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {classification_csv_path}")

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

    applications = ["hiring", "loan", "edu"]

    application_to_pool_count = {
        "hiring": 200,
        "loan": 500,
        "edu": 500,
    }

    resume_counts = [2, 4, 6, 8, 10]

    draw_combined_delta_figure(
        model_names=model_names_order,
        applications=applications,
        resume_counts=resume_counts,
        application_to_pool_count=application_to_pool_count,
        max_n_trials=max_n_trials,
        output_dir="outputs/societal",
        show_errorbars=False,
        classification_alpha=0.05,
    )
