import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
}

FIG_FONT_SIZE = 9.5
MARKER_SIZE = 4.8
STAR_FONT_SIZE = 7.0
STAR_Y_OFFSET = 0.24
MODEL_ROW_SPACING = 1.8

ATTRIBUTE_TITLE_PAD = 8
APPLICATION_TITLE_OFFSET = 0.030
ROW_HSPACE = 0.6

# Match current Fig. 2 caption:
# H1: minority scores exceed majority scores.
MANN_WHITNEY_ALTERNATIVE = "greater"

# Use the same x-axis range across all six panels.
# Set to False if you want each panel to be locally zoomed.
USE_GLOBAL_XLIM = False

BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 2026


def get_model_color_map(model_names, palette):
    """
    Assign one color to each model, following the order of model_names.
    """
    if len(palette) < len(model_names):
        raise ValueError(
            f"Palette has {len(palette)} colors, but {len(model_names)} models are provided."
        )

    return {
        model_name: palette[i]
        for i, model_name in enumerate(model_names)
    }


# ============================================================
# Statistical helpers
# ============================================================

def p_to_stars(p):
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


def paired_bootstrap_mean_diff_ci(
    paired_diffs,
    confidence=0.95,
    n_boot=BOOTSTRAP_N,
    seed=BOOTSTRAP_SEED,
    batch_size=1000,
):
    """
    Paired bootstrap 95% CI for:

        Δ score = mean_i(minority_score_i - majority_score_i)

    The bootstrap resamples candidate-level paired differences with replacement.
    """
    paired_diffs = np.asarray(paired_diffs, dtype=float)

    if len(paired_diffs) == 0:
        return np.nan, np.nan

    if len(paired_diffs) == 1:
        return float(paired_diffs[0]), float(paired_diffs[0])

    rng = np.random.default_rng(seed)
    n = len(paired_diffs)

    boot_means = []

    for start in range(0, n_boot, batch_size):
        cur_batch = min(batch_size, n_boot - start)

        sampled = rng.choice(
            paired_diffs,
            size=(cur_batch, n),
            replace=True,
        )

        boot_means.append(sampled.mean(axis=1))

    boot_means = np.concatenate(boot_means)

    alpha = 1 - confidence
    ci_low = np.percentile(boot_means, 100 * alpha / 2)
    ci_high = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return float(ci_low), float(ci_high)


def paired_p_value(paired_diffs, alternative=MANN_WHITNEY_ALTERNATIVE):
    """
    Paired significance test for candidate-level differences.

    Default:
        H1: minority scores exceed majority scores,
        i.e., candidate-level differences are greater than 0.

    Uses Wilcoxon signed-rank test.
    """
    paired_diffs = np.asarray(paired_diffs, dtype=float)

    if len(paired_diffs) == 0:
        return np.nan

    # If all paired differences are exactly zero, Wilcoxon is undefined.
    if np.allclose(paired_diffs, 0):
        return 1.0

    try:
        stat, p_value = stats.wilcoxon(
            paired_diffs,
            alternative=alternative,
            zero_method="wilcox",
        )
        return float(p_value)
    except ValueError:
        return np.nan


def compute_difference_results(attribute_type, file_name):
    """
    Compute paired score difference:

        Δ score = mean_i(
            mean score of societal minority attributes for candidate i
            -
            mean score of societal majority attributes for candidate i
        )

    This uses candidate_id to preserve the matched design.

    For Gender Identity:
        minority = Transgender + Non-binary
        majority = Cisgender

    For Sexual Orientation:
        minority = Homosexual + Bisexual + Asexual
        majority = Heterosexual
    """
    minority_attributes = set(type_to_minority_attributes[attribute_type])

    candidate_to_scores = {}

    with open(file_name, "r") as f:
        for line in f:
            item = json.loads(line)

            attribute = item["attribute"]
            score = float(item["score"])
            candidate_id = str(item["candidate"])

            if candidate_id not in candidate_to_scores:
                candidate_to_scores[candidate_id] = {
                    "minority": [],
                    "majority": [],
                }

            if attribute in minority_attributes:
                candidate_to_scores[candidate_id]["minority"].append(score)
            else:
                candidate_to_scores[candidate_id]["majority"].append(score)

    paired_diffs = []
    candidate_minority_means = []
    candidate_majority_means = []

    skipped_candidates = 0

    for candidate_id, scores in candidate_to_scores.items():
        minority_scores = scores["minority"]
        majority_scores = scores["majority"]

        # Keep only candidates that have both minority and majority versions.
        if len(minority_scores) == 0 or len(majority_scores) == 0:
            skipped_candidates += 1
            continue

        minority_mean_i = float(np.mean(minority_scores))
        majority_mean_i = float(np.mean(majority_scores))
        diff_i = minority_mean_i - majority_mean_i

        candidate_minority_means.append(minority_mean_i)
        candidate_majority_means.append(majority_mean_i)
        paired_diffs.append(diff_i)

    paired_diffs = np.asarray(paired_diffs, dtype=float)
    candidate_minority_means = np.asarray(candidate_minority_means, dtype=float)
    candidate_majority_means = np.asarray(candidate_majority_means, dtype=float)

    if len(paired_diffs) == 0:
        return None

    diff = float(np.mean(paired_diffs))

    ci_low, ci_high = paired_bootstrap_mean_diff_ci(
        paired_diffs=paired_diffs,
    )

    p_value = paired_p_value(
        paired_diffs=paired_diffs,
        alternative=MANN_WHITNEY_ALTERNATIVE,
    )

    return {
        "minority_mean": float(np.mean(candidate_minority_means)),
        "majority_mean": float(np.mean(candidate_majority_means)),
        "diff": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_pairs": int(len(paired_diffs)),
        "skipped_candidates": int(skipped_candidates),
    }


# ============================================================
# Style helpers
# ============================================================

def set_nature_style():
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

        "axes.titlesize": FIG_FONT_SIZE,
        "axes.labelsize": FIG_FONT_SIZE,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.0,

        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "lines.linewidth": 1.1,
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


def collect_all_difference_results(applications, attribute_types, model_names):
    """
    Pre-compute all results so that we can determine global x-limits.
    """
    all_results = {}

    for application in applications:
        all_results[application] = {}

        for attribute_type in attribute_types:
            all_results[application][attribute_type] = {}

            for model_name in model_names:
                file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"

                if not os.path.exists(file_name):
                    print(f"[Warning] Missing file: {file_name}")
                    continue

                res = compute_difference_results(attribute_type, file_name)

                if res is None:
                    print(f"[Warning] No valid result: {application}, {attribute_type}, {model_name}")
                    continue

                all_results[application][attribute_type][model_name] = res

                print(
                    f"{application} | {attribute_type} | {pretty_model_name(model_name)}: "
                    f"majority={res['majority_mean']:.3f}, "
                    f"minority={res['minority_mean']:.3f}, "
                    f"paired diff={res['diff']:.3f}, "
                    f"95% CI=({res['ci_low']:.3f}, {res['ci_high']:.3f}), "
                    f"n_pairs={res['n_pairs']}, "
                    f"skipped={res['skipped_candidates']}, "
                    f"P={res['p_value']:.4g} {p_to_stars(res['p_value'])}"
                )

    return all_results


def get_xlim_from_results(all_results):
    lows = []
    highs = []

    for app_dict in all_results.values():
        for attr_dict in app_dict.values():
            for res in attr_dict.values():
                lows.append(res["ci_low"])
                highs.append(res["ci_high"])

    if not lows:
        return (-0.1, 0.5)

    x_min = min(lows)
    x_max = max(highs)

    # Always include zero.
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)

    span = max(x_max - x_min, 0.1)
    pad = 0.10 * span

    # Give a little negative space so the zero line is visible.
    return (x_min - pad, x_max + pad)


def get_panel_xlim(results_for_panel):
    lows = []
    highs = []

    for res in results_for_panel.values():
        lows.append(res["ci_low"])
        highs.append(res["ci_high"])

    if not lows:
        return (-0.1, 0.5)

    x_min = min(lows)
    x_max = max(highs)

    # x_min = min(x_min, 0.0)
    x_max = max(x_max, 0.0)

    span = max(x_max - x_min, 0.1)
    pad = 0.12 * span

    return (x_min - pad, x_max + pad)


# ============================================================
# Panel plotting
# ============================================================


def plot_difference_panel(
    ax,
    results_for_panel,
    model_names,
    model_color_map,
    show_model_labels=True,
    xlim=None,
):
    """
    Draw one panel:
        one scenario × one attribute.

    y-axis:
        models

    x-axis:
        Δ score = minority mean score - majority mean score

    Interpretation:
        x > 0 means societal minority candidates receive higher scores.
    """
    n_models = len(model_names)
    y_positions = np.arange(n_models)[::-1] * MODEL_ROW_SPACING

    for idx, model_name in enumerate(model_names):
        y = y_positions[idx]

        if model_name not in results_for_panel:
            continue

        res = results_for_panel[model_name]

        diff = res["diff"]
        ci_low = res["ci_low"]
        ci_high = res["ci_high"]
        p_value = res["p_value"]
        stars = p_to_stars(p_value)

        xerr = np.array([
            [diff - ci_low],
            [ci_high - diff],
        ])

        color = model_color_map[model_name]

        ax.errorbar(
            diff,
            y,
            xerr=xerr,
            fmt="o",
            markersize=MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.9,
            ecolor=color,
            elinewidth=0.85,
            capsize=2.2,
            capthick=0.85,
            alpha=0.95,
            zorder=3,
        )

        # ------------------------------------------------------------
        # Add significance label above each dot
        # ------------------------------------------------------------
        ax.text(
            diff,
            y + STAR_Y_OFFSET,
            stars,
            ha="center",
            va="bottom",
            fontsize=STAR_FONT_SIZE,
            color="black",
            clip_on=False,
            zorder=4,
        )

    # ------------------------------------------------------------
    # y-axis: model labels
    # ------------------------------------------------------------
    ax.set_yticks(y_positions)
    ax.set_yticklabels([pretty_model_name(m) for m in model_names])

    if show_model_labels:
        ax.tick_params(axis="y", labelleft=True)

        for tick_label, model_name in zip(ax.get_yticklabels(), model_names):
            tick_label.set_color(model_color_map[model_name])
    else:
        ax.tick_params(axis="y", labelleft=False)

    # Slightly larger top margin so stars above the top dot are not clipped
    ax.set_ylim(
        -0.65 * MODEL_ROW_SPACING,
        (n_models - 1) * MODEL_ROW_SPACING + 0.75 * MODEL_ROW_SPACING,
    )

    if xlim is None:
        xlim = get_panel_xlim(results_for_panel)
    ax.set_xlim(*xlim)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))

    ax.grid(
        axis="x",
        color="0.90",
        linewidth=0.55,
        linestyle="-",
        zorder=0,
    )

    ax.tick_params(
        axis="both",
        direction="out",
        length=2.5,
        width=0.7,
        color="0.15",
        labelcolor="black",
    )

    # Re-apply y-label visibility and colors after general tick_params
    if show_model_labels:
        ax.tick_params(axis="y", labelleft=True)

        for tick_label, model_name in zip(ax.get_yticklabels(), model_names):
            tick_label.set_color(model_color_map[model_name])
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


# ============================================================
# Big figure drawing
# ============================================================

def draw_societal_difference_figure(
    applications,
    attribute_types,
    model_names,
    palette,
    output_dir="outputs/societal",
):
    """
    Revised Fig. 2:

        3 scenarios × 2 attributes = 6 panels

    Each panel shows:
        Δ score = minority mean score - majority mean score

    Positive values indicate preference toward societal minority groups.
    """
    set_nature_style()
    os.makedirs(output_dir, exist_ok=True)

    application_title_map = {
        "hiring": "Hiring",
        "loan": "Loan approval",
        "edu": "Scholarship application",
    }

    attribute_title_map = {
        "Gender Identity": "Gender identity",
        "Sexual Orientation": "Sexual orientation",
    }

    all_results = collect_all_difference_results(
        applications=applications,
        attribute_types=attribute_types,
        model_names=model_names,
    )

    global_xlim = get_xlim_from_results(all_results)
    model_color_map = get_model_color_map(model_names, palette)

    fig, axes = plt.subplots(
        nrows=len(applications),
        ncols=len(attribute_types),
        figsize=(7.6, 8),
        sharex=USE_GLOBAL_XLIM,
        sharey=True,
    )

    if len(applications) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(attribute_types) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, application in enumerate(applications):
        for col_idx, attribute_type in enumerate(attribute_types):
            ax = axes[row_idx, col_idx]

            results_for_panel = all_results[application][attribute_type]

            panel_xlim = global_xlim if USE_GLOBAL_XLIM else get_panel_xlim(results_for_panel)

            plot_difference_panel(
                ax=ax,
                results_for_panel=results_for_panel,
                model_names=model_names,
                model_color_map=model_color_map,
                show_model_labels=(col_idx == 0),
                xlim=panel_xlim,
            )

            if row_idx == 0:
                ax.set_title(
                    attribute_title_map.get(attribute_type, attribute_type),
                    fontsize=FIG_FONT_SIZE,
                    fontweight="bold",
                    pad=ATTRIBUTE_TITLE_PAD,
                )


    fig.supxlabel(
        "Score difference: societal minority − societal majority",
        fontsize=FIG_FONT_SIZE,
        x=0.65,   # move right; default is 0.5
        y=0.09,
    )

    # # Legend
    # legend_handles = [
    #     Line2D(
    #         [0], [0],
    #         marker="o",
    #         linestyle="",
    #         markersize=MARKER_SIZE,
    #         markerfacecolor="0.25",
    #         markeredgecolor="0.25",
    #         label="P < 0.05",
    #     ),
    #     Line2D(
    #         [0], [0],
    #         marker="o",
    #         linestyle="",
    #         markersize=MARKER_SIZE,
    #         markerfacecolor="white",
    #         markeredgecolor="0.25",
    #         label="ns",
    #     ),
    #     Line2D(
    #         [0], [0],
    #         color="0.25",
    #         linestyle="-",
    #         linewidth=0.8,
    #         label="No difference",
    #     ),
    # ]

    # fig.legend(
    #     handles=legend_handles,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 0.010),
    #     ncol=3,
    #     frameon=False,
    #     handlelength=1.8,
    #     columnspacing=1.3,
    #     handletextpad=0.45,
    #     fontsize=8.0,
    # )

    fig.subplots_adjust(
        left=0.295,
        right=0.985,
        bottom=0.155,
        top=0.925,
        wspace=0.130,
        hspace=ROW_HSPACE,
    )

    # ------------------------------------------------------------
    # Application row titles, original Fig. 2 style
    # ------------------------------------------------------------
    for row_idx, application in enumerate(applications):
        row_axes = axes[row_idx, :]

        left_pos = row_axes[0].get_position()
        right_pos = row_axes[-1].get_position()

        x_center = (left_pos.x0 + right_pos.x1) / 2
        y_top = left_pos.y1

        fig.text(
            x_center,
            y_top + APPLICATION_TITLE_OFFSET,
            application_title_map.get(application, application),
            ha="center",
            va="bottom",
            fontsize=FIG_FONT_SIZE,
            fontweight="bold",
        )

    base = "societal_individual_assessment_difference_plot"

    for ext in ["pdf", "png", "svg"]:
        path = os.path.join(output_dir, f"{base}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    palette = [
        "#9ba415ff",
        "#459434ff",
        "#019aa3ff",
        "#0272b2ff",
        "#a84e94ff",
        "#c93e3fff",
        "#ec6f00ff",
        "#cca02cff",
    ]

    applications = ["hiring", "loan", "edu"]

    attribute_types = [
        "Gender Identity",
        "Sexual Orientation",
    ]

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

    draw_societal_difference_figure(
        applications=applications,
        attribute_types=attribute_types,
        model_names=model_names,
        palette=palette,
        output_dir="outputs/societal",
    )