#!/usr/bin/env python3
"""
Generate the reviewer-suggested redesign of Figure 4.

The output is a 2 x 3 summary figure:

    columns: Hiring | Loan approval | Scholarship allocation
    rows:    Gender | Race

Each panel contains:
    - thin group-colored lines for the individual LLMs;
    - thick lines with markers for the unweighted cross-model mean;
    - a dashed uniform-random candidate-level selection-rate baseline;
    - a shaded region indicating that the focal group is the contextual minority;
    - symmetric contextual-minority/contextual-majority annotations in the
      first panel of each row.

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
from matplotlib.ticker import MaxNLocator, MultipleLocator


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

DEFAULT_APPLICATION_TO_POOL_COUNT = {
    "hiring": 200,
    "loan": 500,
    "edu": 500,
}

APPLICATION_TEXT = {
    "hiring": {
        "title": "Hiring",
        "subtitle": (
            "Female/Black favored;\n"
            "gap widest when they are the minority"
        ),
    },
    "loan": {
        "title": "Loan approval",
        "subtitle": (
            "Male/White favored;\n"
            "gap widest when they are the minority"
        ),
    },
    "edu": {
        "title": "Scholarship allocation",
        "subtitle": (
            "Female/Black favored, as in hiring;\n"
            "pattern weaker and more model-dependent"
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

# Extra empty space beneath the curves in the first panel of each row so
# that the contextual-minority/contextual-majority annotations do not overlap
# the plotted lines.
FIRST_COLUMN_PANEL_LOWER_MARGIN_FRACTION = 0.22

FOCAL_COLOR = "#d81b60"
REFERENCE_COLOR = "#3949ab"
INDIVIDUAL_ALPHA = 0.27
MINORITY_SHADE_COLOR = "0.93"
BASELINE_COLOR = "0.05"

FIGURE_SIZE = (16.5, 10.5)
OUTPUT_BASENAME = "Figure4_contextual_bias_reviewer_style_summary"


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


def _add_inline_group_label(
    ax: plt.Axes,
    application: str,
    group_role: str,
    label: str,
    curve: Mapping[float, float],
    color: str,
) -> None:
    """
    Place an inline group label near the corresponding mean curve.

    Vertical offsets are proportional to the panel's own y-range, so labels
    remain well positioned when every panel uses different y-axis limits.
    """
    xs, ys = _curve_xy(curve)
    if not xs:
        return

    ymin, ymax = ax.get_ylim()
    y_span = max(ymax - ymin, 1e-12)

    if group_role == "focal":
        index = 0
        x_offset = 2.2
        y_offset_fraction = {
            "hiring": 0.075,
            "loan": -0.085,
            "edu": 0.075,
        }[application]
        horizontal_alignment = "left"
    elif group_role == "reference":
        index = -1
        x_offset = -2.2
        y_offset_fraction = {
            "hiring": -0.075,
            "loan": 0.075,
            "edu": -0.085,
        }[application]
        horizontal_alignment = "right"
    else:
        raise ValueError(f"Unknown group role: {group_role!r}")

    label_y = ys[index] + y_offset_fraction * y_span

    # Keep the annotation inside its own panel.
    inner_margin = 0.045 * y_span
    label_y = min(
        max(label_y, ymin + inner_margin),
        ymax - inner_margin,
    )

    ax.text(
        xs[index] + x_offset,
        label_y,
        label,
        color=color,
        fontsize=12.0,
        fontweight="bold",
        ha=horizontal_alignment,
        va="center",
        clip_on=False,
        zorder=6,
    )


def plot_summary_panel(
    ax: plt.Axes,
    model_to_curves: Mapping[
        str, Mapping[str, Mapping[float, float]]
    ],
    application: str,
    attribute_type: str,
    pool_size: int,
    show_column_header: bool,
    show_x_tick_labels: bool,
    add_context_note: bool,
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
                linewidth=1.05,
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
                linewidth=1.05,
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
        marker="o",
        markersize=5.8,
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
        marker="s",
        markersize=5.4,
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

    # The first panel of each row contains two explanatory annotations
    # beneath the curves. Expand only those lower limits to reserve clean
    # text space.
    if add_context_note:
        panel_span = max(panel_ymax - panel_ymin, 1e-12)
        panel_ymin = max(
            0.0,
            panel_ymin
            - FIRST_COLUMN_PANEL_LOWER_MARGIN_FRACTION * panel_span,
        )

    # Force integer y-axis limits and tick intervals for the lower-left
    # (Race × Hiring) and lower-right (Race × Scholarship allocation)
    # panels. The other four panels retain their existing automatic locator.
    use_integer_y_ticks = (
        attribute_type == "Race"
        and application in {"hiring", "edu"}
    )

    if use_integer_y_ticks:
        integer_ymin = math.floor(panel_ymin)
        integer_ymax = math.ceil(panel_ymax)

        # Choose an integer interval that gives approximately the requested
        # number of major ticks, then align both limits to that interval.
        integer_tick_step = max(
            1,
            math.ceil(
                (integer_ymax - integer_ymin)
                / max(Y_AXIS_TARGET_TICKS - 1, 1)
            ),
        )

        panel_ymin = (
            math.floor(integer_ymin / integer_tick_step)
            * integer_tick_step
        )
        panel_ymax = (
            math.ceil(integer_ymax / integer_tick_step)
            * integer_tick_step
        )

        ax.set_ylim(panel_ymin, panel_ymax)
        ax.yaxis.set_major_locator(
            MultipleLocator(integer_tick_step)
        )
    else:
        ax.set_ylim(panel_ymin, panel_ymax)
        ax.yaxis.set_major_locator(
            MaxNLocator(
                nbins=Y_AXIS_TARGET_TICKS,
                steps=[1, 2, 2.5, 5, 10],
                min_n_ticks=4,
            )
        )

    # Add labels only after the panel-specific y-range is known.
    _add_inline_group_label(
        ax=ax,
        application=application,
        group_role="focal",
        label=focal_group,
        curve=focal_mean,
        color=FOCAL_COLOR,
    )
    _add_inline_group_label(
        ax=ax,
        application=application,
        group_role="reference",
        label=reference_group,
        curve=reference_mean,
        color=REFERENCE_COLOR,
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        labelsize=10.5,
        width=0.85,
        length=5.0,
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
            fontsize=14.0,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            0.5,
            1.16,
            APPLICATION_TEXT[application]["subtitle"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10.6,
            fontstyle="italic",
            color="0.35",
            linespacing=1.08,
            clip_on=False,
        )

    if add_context_note:
        ymin, ymax = ax.get_ylim()
        note_y = ymin + 0.045 * (ymax - ymin)

        if attribute_type == "Gender":
            minority_text = "Female is\ncontextual minority"
            majority_text = "Male is\ncontextual minority"
        elif attribute_type == "Race":
            minority_text = "Black is\ncontextual minority"
            majority_text = "White is\ncontextual minority"
        else:
            raise ValueError(
                f"Unexpected attribute_type for contextual annotation: "
                f"{attribute_type!r}"
            )

        ax.text(
            25.0,
            note_y,
            minority_text,
            ha="center",
            va="bottom",
            fontsize=9.8,
            fontstyle="italic",
            color="0.38",
            linespacing=1.05,
            zorder=5,
        )

        ax.text(
            75.0,
            note_y,
            majority_text,
            ha="center",
            va="bottom",
            fontsize=9.8,
            fontstyle="italic",
            color="0.38",
            linespacing=1.05,
            zorder=5,
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

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=FIGURE_SIZE,
        squeeze=False,
    )

    fig.subplots_adjust(
        left=0.135,
        right=0.988,
        bottom=0.175,
        top=0.835,
        wspace=0.15,
        hspace=0.38,
    )

    for row_index, attribute_type in enumerate(ATTRIBUTE_TYPES):
        for column_index, application in enumerate(APPLICATIONS):
            plot_summary_panel(
                ax=axes[row_index, column_index],
                model_to_curves=panel_data[
                    (attribute_type, application)
                ],
                application=application,
                attribute_type=attribute_type,
                pool_size=pool_size,
                show_column_header=(row_index == 0),
                show_x_tick_labels=True,
                add_context_note=(column_index == 0),
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
    first_row_xlabel_y = second_row_top + 0.48 * (first_row_bottom - second_row_top)

    # Second-row shared xlabel: place it below the second row and above the legend.
    second_row_xlabel_y = second_row_bottom - 0.058

    fig.text(
        plots_center_x,
        first_row_xlabel_y,
        "Proportion of female in pool (%)",
        fontsize=11.5,
        ha="center",
        va="center",
    )

    fig.text(
        plots_center_x,
        second_row_xlabel_y,
        "Proportion of Black in pool (%)",
        fontsize=11.5,
        ha="center",
        va="center",
    )

    fig.supylabel(
        "Candidate-level selection rate (%)",
        fontsize=11.5,
        x=0.018,
        y=plots_center_y,
        va="center",
    )

    # Row-specific attribute labels are placed between the shared y label
    # and the plots and are centered vertically on each row.
    row_label_x = 0.078

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
            fontsize=12.0,
            fontweight="bold",
            rotation=90,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=FOCAL_COLOR,
            marker="o",
            markerfacecolor=FOCAL_COLOR,
            markeredgecolor=FOCAL_COLOR,
            linewidth=3.0,
            markersize=6.5,
            label="Female / Black (cross-model mean)",
        ),
        Line2D(
            [0],
            [0],
            color=REFERENCE_COLOR,
            marker="s",
            markerfacecolor=REFERENCE_COLOR,
            markeredgecolor=REFERENCE_COLOR,
            linewidth=3.0,
            markersize=6.1,
            label="Male / White (cross-model mean)",
        ),
        Line2D(
            [0],
            [0],
            color="0.62",
            linewidth=1.3,
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
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(plots_center_x, 0.012),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=2.2,
        handletextpad=0.7,
        columnspacing=1.55,
        borderaxespad=0.0,
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
