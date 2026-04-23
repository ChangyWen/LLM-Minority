import json
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
    "Disability Status": ["Colorblindness", "Hearing Impairment", "Mobility Impairment"],
    "Chronic Health Condition Status": ["HIV Positive", "Chronic Hepatitis", "Type 1 Diabetes", "Asthma"],
    "Religious Affiliation": ["Jewish", "Jain", "Taoist"],
    "Political Affiliation": ["Green Party", "Libertarian"],
    "Race": ["Black"],
}


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
        return ""


def t_ci(scores, confidence=0.95):
    mean = np.mean(scores)
    sem = stats.sem(scores)
    df = len(scores) - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    lower = mean - t_crit * sem
    upper = mean + t_crit * sem
    return (mean, lower, upper)


def compute_results(attribute_type, file_name):
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

    minority_scores = np.array(minority_scores)
    majority_scores = np.array(majority_scores)

    # Guard against empty groups
    if len(minority_scores) == 0 or len(majority_scores) == 0:
        return None

    minority_mean, minority_ci_low, minority_ci_high = t_ci(minority_scores)
    majority_mean, majority_ci_low, majority_ci_high = t_ci(majority_scores)
    stat, p_value = stats.mannwhitneyu(
        minority_scores, majority_scores, alternative="two-sided"
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


def lighten_color(color, amount=0.45):
    """
    Lighten a color by blending it with white.
    color: tuple (r,g,b) in 0-1 range
    amount: higher = lighter
    """
    return tuple((1 - amount) * c + amount for c in color)


class HandlerVerticalErrorbar(HandlerBase):
    """Custom legend handler: vertical error bar with caps + marker."""
    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height, fontsize, trans
    ):
        # CENTER of legend entry
        x = xdescent + width / 2.0

        # EXTEND vertical line by a scale factor
        scale = 1.9   # increase for longer error bar in legend
        mid = ydescent + height / 2.0
        half = (height / 2.0) * scale
        y0 = mid - half
        y1 = mid + half

        # Extract styles
        linestyle = orig_handle.get_linestyle()
        color = orig_handle.get_color()
        lw = orig_handle.get_linewidth()
        marker = orig_handle.get_marker()
        markersize = orig_handle.get_markersize()

        # ------- Vertical line -------
        vline = Line2D(
            [x, x], [y0, y1],
            linestyle=linestyle,
            color=color,
            linewidth=lw,
        )

        # ------- Caps (horizontal ticks) -------
        cap_width = width * 0.25   # adjust cap width here
        x0 = x - cap_width / 2
        x1 = x + cap_width / 2

        top_cap = Line2D(
            [x0, x1], [y1, y1],     # horizontal top cap
            color=color,
            linewidth=lw,
            linestyle="-",
        )

        bot_cap = Line2D(
            [x0, x1], [y0, y0],     # horizontal bottom cap
            color=color,
            linewidth=lw,
            linestyle="-",
        )

        # ------- Marker at center -------
        marker_artist = Line2D(
            [x], [mid],
            marker=marker,
            markersize=markersize,
            markerfacecolor=orig_handle.get_markerfacecolor(),
            markeredgecolor=orig_handle.get_markeredgecolor(),
            linestyle="none",
        )

        # Apply transform
        for artist in (vline, top_cap, bot_cap, marker_artist):
            artist.set_transform(trans)

        return [vline, top_cap, bot_cap, marker_artist]


def plot_model_panel_societal(
    ax_main,
    attribute_types_present,
    all_results,
    model_name,
    palette_dict,
    show_legend=False,
    is_top_row=False,
):
    """
    Draw one model's societal-bias panel on ax_main.
    - attribute_types_present: list of attribute types that have data for this model
    - all_results: dict[attr_type] -> result (minority/majority + p_value)
    - palette_dict: dict[attr_type] -> color
    - show_legend: whether to draw the legend in this panel
    - is_top_row: if True, x-labels show only stars (no attribute text)
    """

    # Horizontal positions per attribute
    n_attr = len(attribute_types_present)
    x_base = np.arange(n_attr)  # 0,1,2,...
    delta = 0.12

    x_min = x_base - delta
    x_maj = x_base + delta

    # Markers
    minority_marker = "o"
    majority_marker = "s"

    xlabels = []

    for i, attr in enumerate(attribute_types_present):
        res = all_results[attr]
        res_min = res["minority"]
        res_maj = res["majority"]
        p_value = res["p_value"]

        stars = p_to_stars(p_value)

        # --- X tick label logic ---
        if is_top_row:
            # First row: keep only stars (no attribute text)
            label_text = stars
        else:
            # Lower row(s): attribute name + stars (if any)
            label_text = f"{attr}\n{stars}" if stars else attr

        xlabels.append(label_text)

        base_color = palette_dict[attr]
        minority_color = base_color
        majority_color = base_color

        # ----- Minority -----
        m_mean = res_min["mean"]
        m_low = res_min["ci_low"]
        m_high = res_min["ci_high"]

        line_minority, cap_minority, bar_minority = ax_main.errorbar(
            x_min[i],
            m_mean,
            yerr=[[m_mean - m_low], [m_high - m_mean]],
            marker=minority_marker,
            markersize=5.5,
            capsize=5,
            capthick=1.2,
            linewidth=1.5,
            linestyle="--",               # dashed error bar line for minority
            dash_capstyle='round',        # nicer dashed appearance
            color=minority_color,
            markeredgecolor=minority_color,
            markeredgewidth=0.7,
        )

        for bar in bar_minority:
            bar.set_linestyle("--")

        # ----- Majority -----
        g_mean = res_maj["mean"]
        g_low = res_maj["ci_low"]
        g_high = res_maj["ci_high"]

        line_majority, cap_majority, bar_majority = ax_main.errorbar(
            x_maj[i],
            g_mean,
            yerr=[[g_mean - g_low], [g_high - g_mean]],
            marker=majority_marker,
            markersize=5.5,
            capsize=5,
            capthick=1.2,
            linewidth=1.5,
            linestyle="solid",           # solid error bar line for majority
            color=majority_color,
            markeredgecolor=majority_color,
            markeredgewidth=0.7,
        )

        # ---- Numeric labels ----
        ax_main.text(
            x_min[i] - 0.03, m_mean,
            f"{m_mean:.2f}",
            ha="right", va="center",
            fontsize=8, fontweight="bold",
            color=minority_color,
        )
        ax_main.text(
            x_maj[i] + 0.03, g_mean,
            f"{g_mean:.2f}",
            ha="left", va="center",
            fontsize=8, fontweight="bold",
            color=majority_color,
        )

    # -----------------------------
    # Axis Formatting (per panel)
    # -----------------------------
    ax_main.set_xticks(x_base)
    ax_main.set_xlim(x_base[0] - 0.6, x_base[-1] + 0.6)
    ax_main.set_xticklabels(xlabels, rotation=0, ha="center")

    # Color xticklabels by attribute (even if only stars)
    for i, tick in enumerate(ax_main.get_xticklabels()):
        attr = attribute_types_present[i]
        tick.set_color(palette_dict[attr])
        tick.set_fontweight("bold")

    # Title per model
    model_clean = model_name.replace("msra-", "")
    ax_main.set_title(model_clean, fontweight="bold")

    ax_main.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_main.set_axisbelow(True)

    # Legend (only if requested, e.g., top-left panel)
    if show_legend:
        minority_handle = Line2D(
            [0], [0],
            marker=minority_marker,
            linestyle="--",
            color="black",
            markersize=2,
            markerfacecolor="black",
            markeredgecolor="black",
            linewidth=1,
            label="Minority",
        )

        majority_handle = Line2D(
            [0], [0],
            marker=majority_marker,
            linestyle="solid",
            color="black",
            markersize=2,
            markerfacecolor="black",
            markeredgecolor="black",
            linewidth=1,
            label="Majority",
        )

        legend_fontsize = 18   # larger font
        legend_markersize = 3.5  # bigger markers

        ax_main.legend(
            handles=[minority_handle, majority_handle],
            loc="best",
            frameon=True,
            framealpha=0.55,
            borderpad=0.4,
            handler_map={
                minority_handle: HandlerVerticalErrorbar(),
                majority_handle: HandlerVerticalErrorbar(),
            },
            fontsize=legend_fontsize,
            handlelength=2.2,
            markerscale=legend_markersize,
        )

    # Remove top + right spines
    for spine in ["top", "right"]:
        ax_main.spines[spine].set_visible(False)


def draw_results_grid_societal(attribute_types, model_names):
    """
    Create one big figure (2x4 grid) for societal-bias results across models.
    Each panel is one model.
    """

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    # Palette consistent across models: one color per attribute type
    palette = sns.color_palette("husl", len(attribute_types))
    palette_dict = {attr: palette[i] for i, attr in enumerate(attribute_types)}

    fig, axes = plt.subplots(
        2, 4,
        dpi=1024,
        figsize=(18, 8),
        sharex=False,
        sharey=False,  # <-- each subfigure uses its own y-axis range
    )

    fig.suptitle("Loan - Societal Minority vs Majority Scores", fontweight="bold", y=0.97)

    fig.subplots_adjust(
        left=0.06,
        right=0.96,
        bottom=0.08,
        top=0.90,
        wspace=0.28,
        hspace=0.30,
    )

    # --- REMOVE all sub-figure ylabels ---
    for ax_row in axes:
        for ax in ax_row:
            ax.set_ylabel(None)

    # Plot each model
    for idx, model_name in enumerate(model_names):
        row = idx // 4
        col = idx % 4
        ax_main = axes[row, col]

        all_results = {}
        for attribute_type in attribute_types:
            file_name = f"outputs/loan/societal/{attribute_type}/{model_name}.jsonl"
            if os.path.exists(file_name):
                res = compute_results(attribute_type, file_name)
                if res is not None:
                    all_results[attribute_type] = res

        if not all_results:
            print(f"[Warning] No results found for model {model_name}, skipping this panel.")
            ax_main.set_visible(False)
            continue

        attribute_types_present = [a for a in attribute_types if a in all_results]

        plot_model_panel_societal(
            ax_main=ax_main,
            attribute_types_present=attribute_types_present,
            all_results=all_results,
            model_name=model_name,
            palette_dict=palette_dict,
            show_legend=(idx == 0),        # legend only on top-left panel
            is_top_row=(row == 0),         # first row: x-labels = stars only
        )

    # Global labels
    fig.supxlabel("Attribute Type", fontsize=12, fontweight="bold")
    fig.supylabel("Mean Score (±95% CI)", fontsize=12, fontweight="bold")

    # Save one big figure
    save_file = "outputs/loan/societal_grid.png"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    print(f"Saving figure to: {save_file}")
    fig.savefig(save_file, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":

    attribute_types = [
        "Gender Identity",
        "Sexual Orientation",
        # "Religious Affiliation",
        # add more if needed
    ]

    # Fixed order of models in the 2x4 grid
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

    draw_results_grid_societal(attribute_types, model_names)
