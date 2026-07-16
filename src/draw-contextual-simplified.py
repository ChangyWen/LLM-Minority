#!/usr/bin/env python3
"""
Generate the reviewer-suggested redesign of Figure 4.

The output is a 2 x 3 summary figure:

    columns: Hiring | Loan approval | Scholarship allocation
    rows:    Gender | Race

Each panel contains:
    - thin group-colored lines for the individual LLMs;
    - thick lines with markers for the unweighted cross-model mean;
    - cross-model trend-test significance embedded in the explanatory
      annotations in every panel;
    - cross-model difference-test annotations at 20%, 40%, 60%, and 80%;
    - a dashed uniform-random candidate-level selection-rate baseline;
    - a shaded region indicating that the focal group is the contextual minority;
    - legends below the plots identifying the contextual-minority regions;
    - explanatory annotations in every panel showing each group's
      trend significance and favored/less-favored status under greater
      contextual underrepresentation.

The inferential unit for the added tests is the model, which matches the
unweighted cross-model mean shown by the thick curves:
    - trend tests use a one-sample t-test on the model-specific linear slopes;
    - difference tests use a paired, two-sided one-sample t-test on the
      model-specific focal-minus-reference differences at each composition.

The resulting P values are adjusted with the Benjamini--Hochberg procedure
in two prespecified families:
    - 24 cross-group difference tests (6 panels x 4 compositions);
    - 12 group-specific trend tests (6 panels x 2 groups).

Expected input path pattern:
    {data_root}/{application}/contextual/{attribute_type}/
        {model_name}_{pool_size}_{pool_count}.jsonl

With the default arguments, this matches the original project layout:
    outputs/hiring/contextual/Gender/msra-gpt-4o_5_200.jsonl
    outputs/loan/contextual/Race/msra-gpt-4o_5_500.jsonl
    ...

Run:
    python figure4_reviewer_redesign.py

Optional example:
    python figure4_reviewer_redesign.py \
        --data-root outputs \
        --output-dir outputs/contextual \
        --pool-size 5 \
        --max-trials 1000000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.stats import t as student_t


# ============================================================
# Configuration
# ============================================================

MODEL_NAMES = [
    "msra-gpt-4o",
    "gpt-oss-120b",
    "Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-Next-80B-A3B-Instruct",
    "GLM-4.5-Air",
    "gemma-3-27b-it",
    "Llama-3.3-70B-Instruct",
    "NVIDIA-Nemotron-Nano-12B-v2",
]

APPLICATIONS = ["hiring", "loan", "edu"]
ATTRIBUTE_TYPES = ["Gender", "Race"]

# Prespecified multiple-testing families.
DIFFERENCE_TEST_X_VALUES = (20.0, 40.0, 60.0, 80.0)
EXPECTED_DIFFERENCE_TEST_COUNT = (
    len(APPLICATIONS)
    * len(ATTRIBUTE_TYPES)
    * len(DIFFERENCE_TEST_X_VALUES)
)
EXPECTED_TREND_TEST_COUNT = (
    len(APPLICATIONS)
    * len(ATTRIBUTE_TYPES)
    * 2
)

DEFAULT_APPLICATION_TO_POOL_COUNT = {
    "hiring": 200,
    "loan": 500,
    "edu": 500,
}

APPLICATION_TEXT = {
    "hiring": {
        "title": "Hiring",
        "subtitle": (
            "Female/Black favored"
        ),
    },
    "loan": {
        "title": "Loan approval",
        "subtitle": (
            "Male/White favored"
        ),
    },
    "edu": {
        "title": "Scholarship allocation",
        "subtitle": (
            "Female/Black favored\n"
            "Pattern resembles hiring"
        ),
    },
}

ATTRIBUTE_STYLE = {
    "Gender": {
        "order": ["Female", "Male"],
        "focal": "Female",
        "reference": "Male",
    },
    "Race": {
        "order": ["Black", "White"],
        "focal": "Black",
        "reference": "White",
    },
}

# Which plotted group is favored overall in each application.
# "focal" denotes Female/Black; "reference" denotes Male/White.
APPLICATION_FAVORED_ROLE = {
    "hiring": "focal",
    "loan": "reference",
    "edu": "focal",
}

# Each of the six panels receives its own data-driven y-axis limits.
# The limits include all individual-model curves and the 20% random baseline.
#
# Optional manual overrides can be added for any individual panel:
#     ("Gender", "hiring"): (14.0, 27.0),
#     ("Race", "loan"): (5.0, 31.0),
#
# Leave the dictionary empty to use automatic panel-specific limits throughout.
PANEL_Y_AXIS_OVERRIDES: dict[tuple[str, str], tuple[float, float]] = {}

Y_AXIS_PADDING_FRACTION = 0.12
Y_AXIS_MIN_PADDING = 0.50
Y_AXIS_MIN_DATA_SPAN = 4.00
Y_AXIS_TARGET_TICKS = 5

# Dedicated empty bands above and below the curves in every panel. The
# explanatory annotations are drawn in these bands rather than on top of
# the model traces.
EXPLANATORY_ARROW_LOWER_MARGIN_FRACTION = 0.38
EXPLANATORY_ARROW_UPPER_MARGIN_FRACTION = 0.38

FOCAL_COLOR = "#ff2d86"
REFERENCE_COLOR = "#7550ff"
INDIVIDUAL_ALPHA = 0.27
MINORITY_SHADE_COLOR = "0.93"
BASELINE_COLOR = "0.05"

FIGURE_SIZE = (16.5, 12.0)
OUTPUT_BASENAME = "Figure4_contextual_bias_scholarship_trend_only"

# ============================================================
# Font sizes
#
# Adjust the values below to resize all textual elements consistently.
# ============================================================

BASE_FONT_SIZE = 10.5
AXIS_FONT_SIZE = 16
TICK_FONT_SIZE = 13.5
TITLE_FONT_SIZE = 16
SUBTITLE_FONT_SIZE = 16
ROW_LABEL_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
CONTEXT_LEGEND_FONT_SIZE = 13.5
SIGNIFICANCE_FONT_SIZE = 14.5
UNDERREPRESENTATION_FONT_SIZE = 11.5
UNDERREPRESENTATION_ARROW_LINEWIDTH = 1.6
UNDERREPRESENTATION_ARROW_MUTATION_SCALE = 14


# ============================================================
# Plot style
# ============================================================

def set_figure_style() -> None:
    """Set a clean, compact, publication-oriented Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_FONT_SIZE,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": 0.85,
            "axes.edgecolor": "0.10",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.titleweight": "bold",
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "legend.frameon": False,
        }
    )


# ============================================================
# Data loading and aggregation
# ============================================================

def get_attribute_style(attribute_type: str) -> Mapping[str, object]:
    try:
        return ATTRIBUTE_STYLE[attribute_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown attribute type {attribute_type!r}; "
            f"expected one of {list(ATTRIBUTE_STYLE)}."
        ) from exc


def _validate_selected_index(
    selected_index: int,
    pool_size: int,
    file_path: Path,
    line_number: int,
) -> None:
    if selected_index < 0 or selected_index >= pool_size:
        raise ValueError(
            f"{file_path}:{line_number}: suggested_candidate_id="
            f"{selected_index} is outside the valid zero-based range "
            f"[0, {pool_size - 1}]."
        )


def compute_selection_rates(
    file_path: Path,
    pool_size: int,
    max_n_trials: int,
) -> Dict[str, Dict[int, float]]:
    """
    Compute candidate-level selection rates for every group and composition.

    The returned mapping is:
        group -> same_attr_count -> selection_rate

    Here, same_attr_count excludes the focal candidate itself, matching the
    original Figure 4 computation. For example, same_attr_count=0 means the
    candidate is the only member of its group in a pool.
    """
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    counts: MutableMapping[str, MutableMapping[int, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    n_trials = 0

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if n_trials >= max_n_trials:
                break

            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{file_path}:{line_number}: invalid JSON."
                ) from exc

            try:
                attributes = list(item["attributes"])
                selected_index = int(item["suggested_candidate_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{file_path}:{line_number}: each record must contain "
                    "'attributes' and an integer 'suggested_candidate_id'."
                ) from exc

            # Preserve the behavior of the original script: race trials that
            # contain Asian candidates are excluded from the Black/White study.
            if "Asian" in attributes:
                continue

            if len(attributes) != pool_size:
                raise ValueError(
                    f"{file_path}:{line_number}: found {len(attributes)} "
                    f"candidates, but --pool-size is {pool_size}."
                )

            _validate_selected_index(
                selected_index=selected_index,
                pool_size=pool_size,
                file_path=file_path,
                line_number=line_number,
            )

            n_trials += 1
            group_sizes = Counter(attributes)

            for candidate_index, group in enumerate(attributes):
                same_attr_count = group_sizes[group] - 1
                counts[group][same_attr_count][1] += 1
                if candidate_index == selected_index:
                    counts[group][same_attr_count][0] += 1

    if n_trials == 0:
        raise ValueError(
            f"No usable trials were found in {file_path}. "
            "Check the file contents and filtering conditions."
        )

    rates: Dict[str, Dict[int, float]] = {}

    for group, group_counts in counts.items():
        rates[group] = {}
        for same_attr_count, (hit_count, total_count) in sorted(
            group_counts.items()
        ):
            if total_count <= 0:
                continue
            rates[group][same_attr_count] = hit_count / total_count

    return rates


def x_percent_for_group(
    group: str,
    same_attr_count: int,
    attribute_type: str,
    pool_size: int,
) -> float:
    """
    Express every curve on the same focal-group proportion axis.

    Gender:
        focal group = Female
        reference group = Male

    Race:
        focal group = Black
        reference group = White
    """
    style = get_attribute_style(attribute_type)
    group_count = same_attr_count + 1

    if group == style["focal"]:
        focal_count = group_count
    elif group == style["reference"]:
        focal_count = pool_size - group_count
    else:
        raise ValueError(
            f"Unexpected group {group!r} for attribute {attribute_type!r}."
        )

    return 100.0 * focal_count / pool_size


def rates_to_plot_curves(
    rates: Mapping[str, Mapping[int, float]],
    attribute_type: str,
    pool_size: int,
) -> Dict[str, Dict[float, float]]:
    """
    Convert raw rates to:
        group -> focal_group_percentage -> selection_rate_percentage
    """
    style = get_attribute_style(attribute_type)
    curves: Dict[str, Dict[float, float]] = {
        group: {} for group in style["order"]
    }

    for group in style["order"]:
        for same_attr_count, rate in rates.get(group, {}).items():
            x_percent = x_percent_for_group(
                group=group,
                same_attr_count=same_attr_count,
                attribute_type=attribute_type,
                pool_size=pool_size,
            )
            curves[group][x_percent] = 100.0 * rate

    return curves


def build_input_path(
    data_root: Path,
    application: str,
    attribute_type: str,
    model_name: str,
    pool_size: int,
    pool_count: int,
) -> Path:
    return (
        data_root
        / application
        / "contextual"
        / attribute_type
        / f"{model_name}_{pool_size}_{pool_count}.jsonl"
    )


def load_panel_model_curves(
    data_root: Path,
    application: str,
    attribute_type: str,
    model_names: Sequence[str],
    pool_size: int,
    pool_count: int,
    max_n_trials: int,
    strict_missing: bool,
) -> Dict[str, Dict[str, Dict[float, float]]]:
    """
    Load plot-ready curves for all available models in one panel.
    """
    model_to_curves: Dict[str, Dict[str, Dict[float, float]]] = {}
    missing_paths: list[Path] = []

    for model_name in model_names:
        file_path = build_input_path(
            data_root=data_root,
            application=application,
            attribute_type=attribute_type,
            model_name=model_name,
            pool_size=pool_size,
            pool_count=pool_count,
        )

        if not file_path.exists():
            missing_paths.append(file_path)
            continue

        rates = compute_selection_rates(
            file_path=file_path,
            pool_size=pool_size,
            max_n_trials=max_n_trials,
        )

        model_to_curves[model_name] = rates_to_plot_curves(
            rates=rates,
            attribute_type=attribute_type,
            pool_size=pool_size,
        )

    if missing_paths:
        message = "\n".join(f"  - {path}" for path in missing_paths)
        if strict_missing:
            raise FileNotFoundError(
                "The following expected model-output files are missing:\n"
                f"{message}"
            )
        print(
            "[Warning] Skipping missing model-output files:\n"
            f"{message}",
            file=sys.stderr,
        )

    if not model_to_curves:
        raise FileNotFoundError(
            "No model-output files were found for "
            f"application={application!r}, attribute={attribute_type!r}. "
            f"Expected files below {data_root.resolve()}."
        )

    return model_to_curves


def compute_cross_model_mean(
    model_to_curves: Mapping[
        str, Mapping[str, Mapping[float, float]]
    ],
    group: str,
) -> Dict[float, float]:
    """
    Compute the unweighted mean of model-level selection rates at each x.
    """
    x_to_values: MutableMapping[float, list[float]] = defaultdict(list)

    for curves in model_to_curves.values():
        for x_percent, selection_rate in curves.get(group, {}).items():
            if math.isfinite(selection_rate):
                x_to_values[x_percent].append(selection_rate)

    return {
        x_percent: float(np.mean(values))
        for x_percent, values in sorted(x_to_values.items())
        if values
    }


def p_to_stars(p_value: float) -> str:
    """Convert a p value to the significance labels used in Figure 9."""
    if not math.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def benjamini_hochberg(
    p_values: Sequence[float],
) -> list[float]:
    """
    Adjust a prespecified family of P values using Benjamini--Hochberg FDR.

    The returned adjusted P values preserve the input order. All values must
    be finite and lie in [0, 1], because the two test families used in this
    figure are expected to be complete.
    """
    p_array = np.asarray(p_values, dtype=float)

    if p_array.ndim != 1:
        raise ValueError("p_values must be a one-dimensional sequence.")
    if p_array.size == 0:
        return []
    if not np.all(np.isfinite(p_array)):
        raise ValueError(
            "Benjamini--Hochberg adjustment received a non-finite P value."
        )
    if np.any((p_array < 0.0) | (p_array > 1.0)):
        raise ValueError("All P values must lie between 0 and 1.")

    order = np.argsort(p_array, kind="mergesort")
    ranked_p = p_array[order]
    n_tests = ranked_p.size
    ranks = np.arange(1, n_tests + 1, dtype=float)

    adjusted_ranked = ranked_p * n_tests / ranks
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def _one_sample_t_test(
    values: Sequence[float],
    null_mean: float = 0.0,
) -> Dict[str, float]:
    """
    One-sample t-test with two-sided and directional p values.

    The directional alternatives are:
        p_value_one_inc: mean(values) > null_mean
        p_value_one_dec: mean(values) < null_mean
    """
    array = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=float,
    )
    n = int(array.size)

    if n < 2:
        return {
            "n_models": float(n),
            "mean": float("nan"),
            "t": float("nan"),
            "p_value_two_sided": float("nan"),
            "p_value_one_inc": float("nan"),
            "p_value_one_dec": float("nan"),
        }

    sample_mean = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    difference = sample_mean - null_mean

    # Handle a completely degenerate sample explicitly. This can occur, for
    # example, when all models have exactly the same endpoint rate.
    if sample_sd <= np.finfo(float).eps:
        if math.isclose(difference, 0.0, abs_tol=1e-12):
            t_value = 0.0
            p_two = 1.0
            p_inc = 0.5
            p_dec = 0.5
        elif difference > 0:
            t_value = float("inf")
            p_two = 0.0
            p_inc = 0.0
            p_dec = 1.0
        else:
            t_value = float("-inf")
            p_two = 0.0
            p_inc = 1.0
            p_dec = 0.0
    else:
        standard_error = sample_sd / math.sqrt(n)
        t_value = difference / standard_error
        degrees_of_freedom = n - 1
        p_two = 2.0 * student_t.sf(abs(t_value), degrees_of_freedom)
        p_inc = student_t.sf(t_value, degrees_of_freedom)
        p_dec = student_t.cdf(t_value, degrees_of_freedom)

    return {
        "n_models": float(n),
        "mean": sample_mean,
        "t": float(t_value),
        "p_value_two_sided": float(p_two),
        "p_value_one_inc": float(p_inc),
        "p_value_one_dec": float(p_dec),
    }


def trend_label_from_test(test_result: Mapping[str, float]) -> str:
    """
    Return a Figure-9-style arrow label using the BH-adjusted trend P value.

    The arrow direction is determined by the sign of the mean model-specific
    slope. The P value is the one-sided P value in that direction, adjusted
    across the 12 prespecified group-specific trend tests.
    """
    mean_slope = float(test_result.get("mean", float("nan")))
    p_adjusted = float(
        test_result.get(
            "p_value_directional_adjusted",
            test_result.get("p_value_directional_raw", float("nan")),
        )
    )

    if not math.isfinite(mean_slope) or not math.isfinite(p_adjusted):
        return "ns"

    stars = p_to_stars(p_adjusted)
    if stars == "ns" or math.isclose(mean_slope, 0.0, abs_tol=1e-12):
        return "ns"

    return f"↑{stars}" if mean_slope > 0.0 else f"↓{stars}"


def compute_cross_model_significance(
    model_to_curves: Mapping[
        str, Mapping[str, Mapping[float, float]]
    ],
    focal_group: str,
    reference_group: str,
) -> Dict[str, object]:
    """
    Test the thick cross-model mean curves using models as replicates.

    Trend tests
    -----------
    For each group, fit a linear slope to every model-specific curve and test
    whether the mean slope across models differs directionally from zero.

    Difference tests
    ----------------
    At each shared composition (20%, 40%, 60%, and 80% for pool size 5),
    compute the focal-minus-reference difference within each model and test
    whether the mean paired difference across models differs from zero.

    Using the model as the inferential unit is aligned with the plotted thick
    lines, which are unweighted means of model-level selection rates.
    """
    trend_results: Dict[str, Dict[str, float]] = {}

    for group in (focal_group, reference_group):
        slopes: list[float] = []

        for curves in model_to_curves.values():
            curve = curves.get(group, {})
            finite_points = sorted(
                (float(x), float(y))
                for x, y in curve.items()
                if math.isfinite(x) and math.isfinite(y)
            )

            if len(finite_points) < 2:
                continue

            xs = np.asarray([point[0] for point in finite_points], dtype=float)
            ys = np.asarray([point[1] for point in finite_points], dtype=float)

            if np.ptp(xs) <= np.finfo(float).eps:
                continue

            slope = float(np.polyfit(xs, ys, deg=1)[0])
            if math.isfinite(slope):
                slopes.append(slope)

        trend_result = _one_sample_t_test(slopes, null_mean=0.0)
        mean_slope = float(trend_result.get("mean", float("nan")))

        if not math.isfinite(mean_slope):
            directional_p = float("nan")
        elif mean_slope > 0.0:
            directional_p = float(trend_result["p_value_one_inc"])
        elif mean_slope < 0.0:
            directional_p = float(trend_result["p_value_one_dec"])
        else:
            directional_p = 0.5

        trend_result["p_value_directional_raw"] = directional_p
        trend_results[group] = trend_result

    pairwise_results: Dict[float, Dict[str, float]] = {}

    # The difference-test family is prespecified at these four shared pool
    # compositions for every panel.
    for x_percent in DIFFERENCE_TEST_X_VALUES:
        paired_differences: list[float] = []

        for curves in model_to_curves.values():
            focal_curve = curves.get(focal_group, {})
            reference_curve = curves.get(reference_group, {})

            if x_percent not in focal_curve or x_percent not in reference_curve:
                continue

            difference = (
                float(focal_curve[x_percent])
                - float(reference_curve[x_percent])
            )
            if math.isfinite(difference):
                paired_differences.append(difference)

        pairwise_results[x_percent] = _one_sample_t_test(
            paired_differences,
            null_mean=0.0,
        )

    return {
        "trend": trend_results,
        "pairwise": pairwise_results,
    }


def adjust_cross_model_p_values(
    panel_significance: MutableMapping[
        tuple[str, str], Dict[str, object]
    ],
) -> None:
    """
    Apply Benjamini--Hochberg correction to the two prespecified families.

    Family 1:
        24 two-sided cross-group difference tests
        = 6 panels x 4 pool compositions.

    Family 2:
        12 one-sided group-specific trend tests
        = 6 panels x 2 groups.

    The function mutates ``panel_significance`` by adding:
        pairwise[*]["p_value_two_sided_adjusted"]
        trend[*]["p_value_directional_adjusted"]
    """
    difference_refs: list[MutableMapping[str, float]] = []
    difference_raw_p_values: list[float] = []

    trend_refs: list[MutableMapping[str, float]] = []
    trend_raw_p_values: list[float] = []

    for attribute_type in ATTRIBUTE_TYPES:
        style = get_attribute_style(attribute_type)

        for application in APPLICATIONS:
            panel_key = (attribute_type, application)
            if panel_key not in panel_significance:
                raise ValueError(
                    f"Missing significance results for panel {panel_key}."
                )

            significance = panel_significance[panel_key]

            pairwise = significance.get("pairwise", {})
            if not isinstance(pairwise, MutableMapping):
                raise TypeError(
                    f"Pairwise results for panel {panel_key} are invalid."
                )

            for x_percent in DIFFERENCE_TEST_X_VALUES:
                test_result = pairwise.get(x_percent)
                if not isinstance(test_result, MutableMapping):
                    raise ValueError(
                        "Missing cross-group difference test for "
                        f"panel={panel_key}, x={x_percent:.0f}%."
                    )

                raw_p = float(
                    test_result.get(
                        "p_value_two_sided",
                        float("nan"),
                    )
                )
                difference_refs.append(test_result)
                difference_raw_p_values.append(raw_p)

            trend = significance.get("trend", {})
            if not isinstance(trend, MutableMapping):
                raise TypeError(
                    f"Trend results for panel {panel_key} are invalid."
                )

            for group in style["order"]:
                test_result = trend.get(group)
                if not isinstance(test_result, MutableMapping):
                    raise ValueError(
                        "Missing group-specific trend test for "
                        f"panel={panel_key}, group={group!r}."
                    )

                raw_p = float(
                    test_result.get(
                        "p_value_directional_raw",
                        float("nan"),
                    )
                )
                trend_refs.append(test_result)
                trend_raw_p_values.append(raw_p)

    if len(difference_raw_p_values) != EXPECTED_DIFFERENCE_TEST_COUNT:
        raise ValueError(
            "Expected "
            f"{EXPECTED_DIFFERENCE_TEST_COUNT} cross-group difference tests, "
            f"but found {len(difference_raw_p_values)}."
        )

    if len(trend_raw_p_values) != EXPECTED_TREND_TEST_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_TREND_TEST_COUNT} trend tests, "
            f"but found {len(trend_raw_p_values)}."
        )

    difference_adjusted = benjamini_hochberg(
        difference_raw_p_values
    )
    trend_adjusted = benjamini_hochberg(trend_raw_p_values)

    for test_result, adjusted_p in zip(
        difference_refs,
        difference_adjusted,
    ):
        test_result["p_value_two_sided_adjusted"] = float(adjusted_p)

    for test_result, adjusted_p in zip(
        trend_refs,
        trend_adjusted,
    ):
        test_result["p_value_directional_adjusted"] = float(adjusted_p)


# ============================================================
# Plotting helpers
# ============================================================

def _curve_xy(
    curve: Mapping[float, float],
) -> tuple[list[float], list[float]]:
    xs = sorted(curve)
    ys = [curve[x] for x in xs]
    return xs, ys


def _panel_y_limits(
    model_to_curves: Mapping[
        str, Mapping[str, Mapping[float, float]]
    ],
    attribute_type: str,
    application: str,
    focal_group: str,
    reference_group: str,
    random_rate_percent: float,
) -> tuple[float, float]:
    """
    Return y-axis limits for one panel.

    By default, the limits are calculated independently from all individual
    model curves in that panel plus the uniform-random baseline. A manual
    panel-specific override in PANEL_Y_AXIS_OVERRIDES takes precedence.
    """
    override = PANEL_Y_AXIS_OVERRIDES.get(
        (attribute_type, application)
    )
    if override is not None:
        ymin, ymax = override
        if not ymin < ymax:
            raise ValueError(
                "Each PANEL_Y_AXIS_OVERRIDES entry must satisfy ymin < ymax."
            )
        return float(ymin), float(ymax)

    values = [float(random_rate_percent)]

    for curves in model_to_curves.values():
        for group in (focal_group, reference_group):
            values.extend(
                float(value)
                for value in curves.get(group, {}).values()
                if math.isfinite(value)
            )

    if not values:
        raise ValueError(
            f"No finite y values found for {attribute_type}, {application}."
        )

    data_min = min(values)
    data_max = max(values)
    data_span = data_max - data_min

    # Avoid an excessively narrow axis when all curves are nearly flat.
    if data_span < Y_AXIS_MIN_DATA_SPAN:
        center = 0.5 * (data_min + data_max)
        data_min = center - 0.5 * Y_AXIS_MIN_DATA_SPAN
        data_max = center + 0.5 * Y_AXIS_MIN_DATA_SPAN
        data_span = Y_AXIS_MIN_DATA_SPAN

    padding = max(
        Y_AXIS_MIN_PADDING,
        Y_AXIS_PADDING_FRACTION * data_span,
    )
    raw_min = max(0.0, data_min - padding)
    raw_max = min(100.0, data_max + padding)

    locator = MaxNLocator(
        nbins=Y_AXIS_TARGET_TICKS,
        steps=[1, 2, 2.5, 5, 10],
        min_n_ticks=4,
    )
    nice_min, nice_max = locator.view_limits(raw_min, raw_max)

    nice_min = max(0.0, float(nice_min))
    nice_max = min(100.0, float(nice_max))

    if nice_max <= nice_min:
        nice_max = min(100.0, nice_min + Y_AXIS_MIN_DATA_SPAN)

    return nice_min, nice_max


def _add_cross_model_difference_labels(
    ax: plt.Axes,
    focal_mean: Mapping[float, float],
    reference_mean: Mapping[float, float],
    significance: Mapping[str, object],
) -> None:
    """
    Add BH-adjusted, two-sided difference-test labels above the means.

    For a five-candidate pool, the prespecified group-composition points are
    20%, 40%, 60%, and 80%, matching the annotations in Figure 9.
    """
    pairwise = significance.get("pairwise", {})
    if not isinstance(pairwise, Mapping) or not pairwise:
        return

    ymin, ymax = ax.get_ylim()
    y_span = max(ymax - ymin, 1e-12)
    # Keep the test labels clearly above the inline group names used near the
    # first and last mean-curve points.
    label_offset = 0.065 * y_span
    label_height = 0.045 * y_span

    labels_to_draw: list[tuple[float, float, str]] = []
    required_ymax = ymax

    for x_percent in sorted(pairwise):
        if x_percent not in focal_mean or x_percent not in reference_mean:
            continue

        test_result = pairwise[x_percent]
        if not isinstance(test_result, Mapping):
            continue

        p_value = float(
            test_result.get(
                "p_value_two_sided_adjusted",
                test_result.get("p_value_two_sided", float("nan")),
            )
        )
        label = p_to_stars(p_value)
        if not label:
            continue

        label_y = max(
            float(focal_mean[x_percent]),
            float(reference_mean[x_percent]),
        ) + label_offset

        labels_to_draw.append((float(x_percent), label_y, label))
        required_ymax = max(required_ymax, label_y + label_height)

    if required_ymax > ymax:
        ax.set_ylim(ymin, required_ymax)

    for x_percent, label_y, label in labels_to_draw:
        ax.text(
            x_percent,
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=SIGNIFICANCE_FONT_SIZE,
            color="0.08",
            clip_on=False,
            zorder=7,
        )


def _add_underrepresentation_arrows(
    ax: plt.Axes,
    application: str,
    attribute_type: str,
    focal_group: str,
    reference_group: str,
    focal_mean: Mapping[float, float],
    reference_mean: Mapping[float, float],
    significance: Mapping[str, object],
) -> None:
    """
    Add two explanatory annotations to one panel.

    Each annotation combines the group name, its BH-adjusted trend direction
    and significance, and whether that group is the more-favored or
    less-favored group in the current application.
    """
    if attribute_type not in {"Gender", "Race"}:
        raise ValueError(
            "Underrepresentation annotations are defined only for "
            f"Gender and Race, not {attribute_type!r}."
        )

    trend = significance.get("trend", {})
    if not isinstance(trend, Mapping):
        trend = {}

    focal_test = trend.get(focal_group, {})
    reference_test = trend.get(reference_group, {})

    focal_trend_label = trend_label_from_test(
        focal_test if isinstance(focal_test, Mapping) else {}
    )
    reference_trend_label = trend_label_from_test(
        reference_test if isinstance(reference_test, Mapping) else {}
    )

    try:
        favored_role = APPLICATION_FAVORED_ROLE[application]
    except KeyError as exc:
        raise ValueError(
            f"No favored-group role is configured for {application!r}."
        ) from exc

    if favored_role == "focal":
        focal_interpretation = "More favored when more underrepresented"
        reference_interpretation = "Less favored when more underrepresented"
    elif favored_role == "reference":
        focal_interpretation = "Less favored when more underrepresented"
        reference_interpretation = "More favored when more underrepresented"
    else:
        raise ValueError(
            "APPLICATION_FAVORED_ROLE values must be 'focal' or 'reference'."
        )

    if application == "edu":
        # Scholarship allocation follows the same broad preference pattern as
        # hiring, so keep these panel annotations compact and show only each
        # group's BH-adjusted trend direction and significance.
        text_for_attribute = {
            "focal": f"{focal_group} {focal_trend_label}",
            "reference": f"{reference_group} {reference_trend_label}",
        }
    else:
        text_for_attribute = {
            "focal": (
                f"{focal_group} {focal_trend_label}\n"
                f"{focal_interpretation}"
            ),
            "reference": (
                f"{reference_group} {reference_trend_label}\n"
                f"{reference_interpretation}"
            ),
        }

    ymin, ymax = ax.get_ylim()
    y_span = max(ymax - ymin, 1e-12)

    focal_values = [
        float(value)
        for value in focal_mean.values()
        if math.isfinite(value)
    ]
    reference_values = [
        float(value)
        for value in reference_mean.values()
        if math.isfinite(value)
    ]

    if not focal_values or not reference_values:
        return

    # The axis limits reserve dedicated annotation bands above and below
    # every plotted trace. Keep the arrows at fixed positions within those
    # bands so their text cannot collide with the individual-model lines.
    focal_arrow_y = ymax - 0.17 * y_span
    reference_arrow_y = ymin + 0.17 * y_span

    # By default, place the focal-group annotation in the upper band and
    # the reference-group annotation in the lower band. In the Gender × Loan
    # panel, reverse this order so Male appears above Female, as Male is the
    # favored group in loan approval.
    reverse_vertical_order = (
        application == "loan"
        and attribute_type == "Gender"
    )

    if reverse_vertical_order:
        upper_text = text_for_attribute["reference"]
        upper_color = REFERENCE_COLOR
        lower_text = text_for_attribute["focal"]
        lower_color = FOCAL_COLOR
    else:
        upper_text = text_for_attribute["focal"]
        upper_color = FOCAL_COLOR
        lower_text = text_for_attribute["reference"]
        lower_color = REFERENCE_COLOR

    ax.text(
        52.0,
        focal_arrow_y + 0.025 * y_span,
        upper_text,
        ha="center",
        va="bottom",
        fontsize=UNDERREPRESENTATION_FONT_SIZE,
        color=upper_color,
        linespacing=1.05,
        clip_on=False,
        zorder=8,
    )

    ax.text(
        48.0,
        reference_arrow_y - 0.025 * y_span,
        lower_text,
        ha="center",
        va="top",
        fontsize=UNDERREPRESENTATION_FONT_SIZE,
        color=lower_color,
        linespacing=1.05,
        clip_on=False,
        zorder=8,
    )


def plot_summary_panel(
    ax: plt.Axes,
    model_to_curves: Mapping[
        str, Mapping[str, Mapping[float, float]]
    ],
    cross_model_significance: Mapping[str, object],
    application: str,
    attribute_type: str,
    pool_size: int,
    show_column_header: bool,
    show_x_tick_labels: bool,
    add_explanatory_arrows: bool,
) -> None:
    style = get_attribute_style(attribute_type)
    focal_group = str(style["focal"])
    reference_group = str(style["reference"])

    # Region where the focal group is numerically underrepresented.
    ax.axvspan(
        0.0,
        50.0,
        facecolor=MINORITY_SHADE_COLOR,
        edgecolor="none",
        zorder=0,
    )

    random_rate_percent = 100.0 / pool_size
    ax.axhline(
        random_rate_percent,
        color=BASELINE_COLOR,
        linestyle=(0, (4, 2.4)),
        linewidth=1.25,
        zorder=1,
    )

    # Individual model traces.
    for curves in model_to_curves.values():
        focal_curve = curves.get(focal_group, {})
        reference_curve = curves.get(reference_group, {})

        if focal_curve:
            xs, ys = _curve_xy(focal_curve)
            ax.plot(
                xs,
                ys,
                color=FOCAL_COLOR,
                alpha=INDIVIDUAL_ALPHA,
                linewidth=2,
                solid_capstyle="round",
                zorder=2,
            )

        if reference_curve:
            xs, ys = _curve_xy(reference_curve)
            ax.plot(
                xs,
                ys,
                color=REFERENCE_COLOR,
                alpha=INDIVIDUAL_ALPHA,
                linewidth=2,
                solid_capstyle="round",
                zorder=2,
            )

    # Cross-model mean traces.
    focal_mean = compute_cross_model_mean(
        model_to_curves=model_to_curves,
        group=focal_group,
    )
    reference_mean = compute_cross_model_mean(
        model_to_curves=model_to_curves,
        group=reference_group,
    )

    if not focal_mean or not reference_mean:
        raise ValueError(
            f"Could not construct both group curves for {application}, "
            f"{attribute_type}."
        )

    focal_xs, focal_ys = _curve_xy(focal_mean)
    reference_xs, reference_ys = _curve_xy(reference_mean)

    ax.plot(
        focal_xs,
        focal_ys,
        color=FOCAL_COLOR,
        marker="X",
        markersize=8,
        markerfacecolor=FOCAL_COLOR,
        markeredgecolor=FOCAL_COLOR,
        linewidth=3.25,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        reference_xs,
        reference_ys,
        color=REFERENCE_COLOR,
        marker="o",
        markersize=8,
        markerfacecolor=REFERENCE_COLOR,
        markeredgecolor=REFERENCE_COLOR,
        linewidth=3.25,
        solid_capstyle="round",
        zorder=4,
    )

    # Axes. Each panel calculates its own y-axis limits from its own data.
    ax.set_xlim(-3.0, 103.0)
    ax.set_xticks([0, 20, 40, 60, 80, 100])

    panel_ymin, panel_ymax = _panel_y_limits(
        model_to_curves=model_to_curves,
        attribute_type=attribute_type,
        application=application,
        focal_group=focal_group,
        reference_group=reference_group,
        random_rate_percent=random_rate_percent,
    )

    # Preserve the manually selected base ranges for the two race panels.
    # For the hiring panel, the annotation margins are added *after* this
    # base range is set, so the arrows receive their own empty bands.
    # if attribute_type == "Race" and application == "hiring":
    #     panel_ymin, panel_ymax = 10.0, 28.0
    # elif attribute_type == "Race" and application == "edu":
    #     panel_ymin, panel_ymax = 13.0, 23.0

    # Every panel contains two explanatory annotations. Expand
    # both ends of its y-axis so the data occupy the middle of the panel and
    # the upper/lower annotations never sit on top of model traces.
    if add_explanatory_arrows:
        panel_span = max(panel_ymax - panel_ymin, 1e-12)
        panel_ymin = max(
            0.0,
            panel_ymin
            - EXPLANATORY_ARROW_LOWER_MARGIN_FRACTION * panel_span,
        )
        panel_ymax = min(
            100.0,
            panel_ymax
            + EXPLANATORY_ARROW_UPPER_MARGIN_FRACTION * panel_span,
        )

    ax.set_ylim(panel_ymin, panel_ymax)

    # Keep integer-valued ticks for the race panels while allowing their
    # limits to expand when explanatory arrows are present.
    if attribute_type == "Race":
        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=Y_AXIS_TARGET_TICKS,
                integer=True,
                min_n_ticks=4,
            )
        )
    else:
        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=Y_AXIS_TARGET_TICKS,
                steps=[1, 2, 2.5, 5, 10],
                min_n_ticks=4,
            )
        )

    # BH-adjusted two-sided tests of the cross-model mean difference.
    _add_cross_model_difference_labels(
        ax=ax,
        focal_mean=focal_mean,
        reference_mean=reference_mean,
        significance=cross_model_significance,
    )


    # Add the explanatory annotations to this panel.
    if add_explanatory_arrows:
        _add_underrepresentation_arrows(
            ax=ax,
            application=application,
            attribute_type=attribute_type,
            focal_group=focal_group,
            reference_group=reference_group,
            focal_mean=focal_mean,
            reference_mean=reference_mean,
            significance=cross_model_significance,
        )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=TICK_FONT_SIZE,
        width=0.85,
        length=0.0,
        pad=4.0,
    )
    ax.tick_params(
        axis="x",
        labelbottom=show_x_tick_labels,
    )

    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.spines["left"].set_color("0.10")
    ax.spines["bottom"].set_color("0.10")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_column_header:
        ax.text(
            0.5,
            1.34,
            APPLICATION_TEXT[application]["title"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            0.5,
            1.15,
            APPLICATION_TEXT[application]["subtitle"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=SUBTITLE_FONT_SIZE,
            fontstyle="italic",
            color="0.35",
            linespacing=1.08,
            clip_on=False,
        )



def _format_model_count(panel_counts: Iterable[int]) -> str:
    unique_counts = sorted(set(panel_counts))
    if not unique_counts:
        return "Individual models"
    if len(unique_counts) == 1:
        return f"Individual models (n = {unique_counts[0]})"
    return (
        "Individual models "
        f"(n = {unique_counts[0]}–{unique_counts[-1]})"
    )


def draw_figure(
    data_root: Path,
    output_dir: Path,
    model_names: Sequence[str],
    application_to_pool_count: Mapping[str, int],
    pool_size: int,
    max_n_trials: int,
    strict_missing: bool,
    png_dpi: int,
) -> tuple[Path, Path]:
    set_figure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all six panels before drawing, so missing-data errors occur early
    # and the legend can report the actual number of model traces.
    panel_data: Dict[
        tuple[str, str],
        Dict[str, Dict[str, Dict[float, float]]],
    ] = {}
    panel_model_counts: list[int] = []

    for attribute_type in ATTRIBUTE_TYPES:
        for application in APPLICATIONS:
            model_to_curves = load_panel_model_curves(
                data_root=data_root,
                application=application,
                attribute_type=attribute_type,
                model_names=model_names,
                pool_size=pool_size,
                pool_count=application_to_pool_count[application],
                max_n_trials=max_n_trials,
                strict_missing=strict_missing,
            )
            panel_data[(attribute_type, application)] = model_to_curves
            panel_model_counts.append(len(model_to_curves))

    # Compute all 36 raw tests first, then apply BH adjustment separately
    # to the 24 cross-group difference tests and the 12 trend tests.
    panel_significance: Dict[
        tuple[str, str],
        Dict[str, object],
    ] = {}

    for attribute_type in ATTRIBUTE_TYPES:
        style = get_attribute_style(attribute_type)
        focal_group = str(style["focal"])
        reference_group = str(style["reference"])

        for application in APPLICATIONS:
            panel_key = (attribute_type, application)
            panel_significance[panel_key] = (
                compute_cross_model_significance(
                    model_to_curves=panel_data[panel_key],
                    focal_group=focal_group,
                    reference_group=reference_group,
                )
            )

    adjust_cross_model_p_values(panel_significance)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=FIGURE_SIZE,
        squeeze=False,
        gridspec_kw={
            # The first row is slightly taller because it also carries the
            # column titles and subtitles.
            "height_ratios": [1.12, 1.0],
        },
    )

    # Reserve a dedicated footer below the plots for, in order:
    #   (1) the second-row shared x label;
    #   (2) the contextual-region legend;
    #   (3) the line/marker legend.
    # The larger inter-row gap similarly isolates the first-row shared x label.
    fig.subplots_adjust(
        left=0.135,
        right=0.988,
        bottom=0.305,
        top=0.835,
        wspace=0.13,
        hspace=0.58,
    )

    for row_index, attribute_type in enumerate(ATTRIBUTE_TYPES):
        for column_index, application in enumerate(APPLICATIONS):
            panel_key = (attribute_type, application)
            plot_summary_panel(
                ax=axes[row_index, column_index],
                model_to_curves=panel_data[panel_key],
                cross_model_significance=panel_significance[panel_key],
                application=application,
                attribute_type=attribute_type,
                pool_size=pool_size,
                show_column_header=(row_index == 0),
                show_x_tick_labels=True,
                add_explanatory_arrows=True,
            )

    # ------------------------------------------------------------
    # Shared axis labels
    #
    # Center both shared labels on the actual 2 x 3 plotting area,
    # rather than on the entire figure canvas.
    # ------------------------------------------------------------
    left_plot_edge = min(
        ax.get_position().x0
        for row_axes in axes
        for ax in row_axes
    )
    right_plot_edge = max(
        ax.get_position().x1
        for row_axes in axes
        for ax in row_axes
    )
    bottom_plot_edge = min(
        ax.get_position().y0
        for row_axes in axes
        for ax in row_axes
    )
    top_plot_edge = max(
        ax.get_position().y1
        for row_axes in axes
        for ax in row_axes
    )

    plots_center_x = (left_plot_edge + right_plot_edge) / 2.0
    plots_center_y = (bottom_plot_edge + top_plot_edge) / 2.0

    # Row-specific x labels, centered on the plotting area.
    first_row_positions = [axes[0, col].get_position() for col in range(3)]
    second_row_positions = [axes[1, col].get_position() for col in range(3)]

    first_row_bottom = min(pos.y0 for pos in first_row_positions)
    second_row_top = max(pos.y1 for pos in second_row_positions)
    second_row_bottom = min(pos.y0 for pos in second_row_positions)

    # First-row shared xlabel: place it in the gap between the two rows.
    first_row_xlabel_y = second_row_top + 0.5 * (first_row_bottom - second_row_top)

    # Stack the second-row shared x label and the two legends in three
    # separate footer bands. Their positions are derived from the actual
    # bottom edge of the plots, so changing subplot spacing remains safe.
    second_row_xlabel_y = second_row_bottom - 0.043
    context_legend_top_y = second_row_xlabel_y - 0.032
    series_legend_top_y = context_legend_top_y - 0.092

    if series_legend_top_y <= 0.02:
        raise RuntimeError(
            "The figure footer is too small for the x label and two legends. "
            "Increase the bottom margin in fig.subplots_adjust()."
        )

    fig.text(
        plots_center_x,
        first_row_xlabel_y,
        "Proportion of female in pool (%)",
        fontsize=AXIS_FONT_SIZE,
        ha="center",
        va="center",
    )

    fig.text(
        plots_center_x,
        second_row_xlabel_y,
        "Proportion of Black in pool (%)",
        fontsize=AXIS_FONT_SIZE,
        ha="center",
        va="center",
    )

    fig.supylabel(
        "Candidate-level selection rate (%)",
        fontsize=AXIS_FONT_SIZE,
        x=0.06,
        y=plots_center_y,
        va="center",
    )

    # Row-specific attribute labels are placed between the shared y label
    # and the plots and are centered vertically on each row.
    row_label_x = 0.093

    for row_index, attribute_type in enumerate(ATTRIBUTE_TYPES):
        first_axis_position = axes[row_index, 0].get_position()
        row_center_y = (
            first_axis_position.y0 + first_axis_position.y1
        ) / 2.0

        fig.text(
            row_label_x,
            row_center_y,
            attribute_type,
            ha="center",
            va="center",
            fontsize=ROW_LABEL_FONT_SIZE,
            fontweight="bold",
            rotation=90,
        )

    series_legend_handles = [
        Line2D(
            [0],
            [0],
            color=FOCAL_COLOR,
            marker="X",
            markerfacecolor=FOCAL_COLOR,
            markeredgecolor=FOCAL_COLOR,
            linewidth=3.25,
            markersize=8,
            label="Female / Black (mean across models)",
        ),
        Line2D(
            [0],
            [0],
            color=REFERENCE_COLOR,
            marker="o",
            markerfacecolor=REFERENCE_COLOR,
            markeredgecolor=REFERENCE_COLOR,
            linewidth=3.25,
            markersize=8,
            label="Male / White (mean across models)",
        ),
        Line2D(
            [0],
            [0],
            color="0.62",
            linewidth=2,
            label=_format_model_count(panel_model_counts),
        ),
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linestyle=(0, (4, 2.4)),
            linewidth=1.25,
            label=(
                "Uniform-random rate "
                f"({100.0 / pool_size:.0f}%)"
            ),
        ),
    ]

    fig.legend(
        handles=series_legend_handles,
        loc="upper center",
        bbox_to_anchor=(plots_center_x, series_legend_top_y),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.2,
        handletextpad=0.7,
        columnspacing=1.55,
        labelspacing=0.65,
        borderaxespad=0.0,
        title="Line",
        title_fontsize=LEGEND_FONT_SIZE,
    )

    # Explain the shaded and unshaded x-axis regions outside the panels.
    # The same encoding is used for the gender and race rows.
    context_legend_handles = [
        Patch(
            facecolor=MINORITY_SHADE_COLOR,
            edgecolor="0.62",
            linewidth=0.8,
            label="Female/Black is contextual minority",
        ),
        Patch(
            facecolor="white",
            edgecolor="0.62",
            linewidth=0.8,
            label="Male/White is contextual minority",
        ),
    ]

    fig.legend(
        handles=context_legend_handles,
        loc="upper center",
        bbox_to_anchor=(plots_center_x, context_legend_top_y),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.25,
        handleheight=0.9,
        handletextpad=0.55,
        columnspacing=1.45,
        labelspacing=0.55,
        borderaxespad=0.0,
        title="Region",
        title_fontsize=LEGEND_FONT_SIZE,
    )

    pdf_path = output_dir / f"{OUTPUT_BASENAME}.pdf"
    png_path = output_dir / f"{OUTPUT_BASENAME}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        png_path,
        dpi=png_dpi,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    return pdf_path, png_path


# ============================================================
# Command-line interface
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the reviewer-style 2 x 3 redesign of Figure 4."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("outputs"),
        help=(
            "Root directory containing hiring/, loan/, and edu/. "
            "Default: outputs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/contextual"),
        help=(
            "Directory in which the PDF and PNG will be saved. "
            "Default: outputs/contextual"
        ),
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=5,
        help="Number of candidates per comparison pool. Default: 5",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=1_000_000,
        help=(
            "Maximum number of usable JSONL trials read from each file. "
            "Default: 1000000"
        ),
    )
    parser.add_argument(
        "--hiring-pool-count",
        type=int,
        default=DEFAULT_APPLICATION_TO_POOL_COUNT["hiring"],
        help="Filename pool-count suffix for hiring. Default: 200",
    )
    parser.add_argument(
        "--loan-pool-count",
        type=int,
        default=DEFAULT_APPLICATION_TO_POOL_COUNT["loan"],
        help="Filename pool-count suffix for loan approval. Default: 500",
    )
    parser.add_argument(
        "--edu-pool-count",
        type=int,
        default=DEFAULT_APPLICATION_TO_POOL_COUNT["edu"],
        help=(
            "Filename pool-count suffix for scholarship allocation. "
            "Default: 500"
        ),
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="PNG resolution. Default: 600",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help=(
            "Fail immediately when any expected model-output file is "
            "missing. By default, missing models are skipped with a warning."
        ),
    )

    args = parser.parse_args()

    if args.pool_size < 2:
        parser.error("--pool-size must be at least 2.")
    if args.max_trials < 1:
        parser.error("--max-trials must be positive.")
    if args.png_dpi < 72:
        parser.error("--png-dpi must be at least 72.")

    return args


def main() -> None:
    args = parse_args()

    application_to_pool_count = {
        "hiring": args.hiring_pool_count,
        "loan": args.loan_pool_count,
        "edu": args.edu_pool_count,
    }

    pdf_path, png_path = draw_figure(
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_names=MODEL_NAMES,
        application_to_pool_count=application_to_pool_count,
        pool_size=args.pool_size,
        max_n_trials=args.max_trials,
        strict_missing=args.strict_missing,
        png_dpi=args.png_dpi,
    )

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
