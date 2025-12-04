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

    minority_mean, minority_ci_low, minority_ci_high = t_ci(minority_scores)
    majority_mean, majority_ci_low, majority_ci_high = t_ci(majority_scores)
    stat, p_value = stats.mannwhitneyu(minority_scores, majority_scores, alternative="two-sided")

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


def draw_results(all_results, attribute_types, model_name):

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    xlabels = []
    x_base = np.array([-0.3 + i * 0.3 for i in range(len(attribute_types))])
    delta = 0.05

    x_min = x_base - delta
    x_maj = x_base + delta

    # One base color per attribute
    palette = sns.color_palette("husl", len(attribute_types))

    # Markers
    minority_marker = "o"
    majority_marker = "s"

    fig, ax = plt.subplots(dpi=1024, figsize=(6, 4))

    # ---- Drawing ----
    for i, attr in enumerate(attribute_types):
        res_min = all_results[attr]["minority"]
        res_maj = all_results[attr]["majority"]
        p_value = all_results[attr]["p_value"]
        stars = p_to_stars(p_value)
        xlabels.append(f"{attr}\n{stars}" if stars else attr)

        base_color = palette[i]
        minority_color = base_color
        majority_color = base_color

        # ----- Minority -----
        m_mean = res_min["mean"]
        m_low = res_min["ci_low"]
        m_high = res_min["ci_high"]

        line_minority, cap_minority, bar_minority = ax.errorbar(
            x_min[i], m_mean,
            yerr=[[m_mean - m_low], [m_high - m_mean]],
            marker=minority_marker,
            markersize=6,
            capsize=6,
            capthick=1.2,
            linewidth=2,
            linestyle="--",               # dashed error bar line for minority
            dash_capstyle='round',        # nicer dashed appearance
            color=minority_color,
            markeredgecolor=minority_color,
            markeredgewidth=0.7,
        )

        # Ensure the vertical bar segments are dashed as well
        for bar in bar_minority:
            bar.set_linestyle("--")

        # ----- Majority -----
        g_mean = res_maj["mean"]
        g_low = res_maj["ci_low"]
        g_high = res_maj["ci_high"]

        line_majority, cap_majority, bar_majority = ax.errorbar(
            x_maj[i], g_mean,
            yerr=[[g_mean - g_low], [g_high - g_mean]],
            marker=majority_marker,
            markersize=6,
            capsize=6,
            capthick=1.2,
            linewidth=2,
            linestyle="solid",           # solid error bar line for majority
            color=majority_color,
            markeredgecolor=majority_color,
            markeredgewidth=0.7,
        )

        # ---- Numeric labels (optional) ----
        ax.text(
            x_min[i] - 0.015, m_mean,
            f"{m_mean:.2f}",
            ha="right", va="center",
            fontsize=8, fontweight="bold",
            color=minority_color,
        )
        ax.text(
            x_maj[i] + 0.015, g_mean,
            f"{g_mean:.2f}",
            ha="left", va="center",
            fontsize=8, fontweight="bold",
            color=majority_color,
        )

    # -----------------------------
    # Axis Formatting
    # -----------------------------
    ax.set_xticks(x_base)
    ax.set_xlim(x_base[0] - 0.15, x_base[-1] + 0.15)

    xticks = ax.get_xticklabels()
    for i, tick in enumerate(xticks):
        tick.set_color(palette[i])         # color per attribute
        tick.set_fontweight("bold")

    ax.set_xticklabels(xlabels, rotation=0, ha="center")
    ax.set_ylabel("Mean Score (±95% CI)", fontsize=11, fontweight="bold")

    model_clean = model_name.replace("msra-", "")
    ax.set_title(f"{model_clean}", pad=15, weight="bold")

    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)

    # -----------------------------
    # SINGLE LEGEND WITH VERTICAL ERROR BAR
    # -----------------------------
    # Create dummy handles with desired style; handler draws them vertically
    minority_handle = Line2D(
        [0], [0],
        marker=minority_marker,
        linestyle="--",         # dashed error bar style
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
        linestyle="solid",      # solid error bar style
        color="black",
        markersize=2,
        markerfacecolor="black",
        markeredgecolor="black",
        linewidth=1,
        label="Majority",
    )

    ax.legend(
        handles=[minority_handle, majority_handle],
        loc="best",
        frameon=True,
        handler_map={
            minority_handle: HandlerVerticalErrorbar(),
            majority_handle: HandlerVerticalErrorbar(),
        },
    )

    # Remove top + right spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    save_file = f"outputs/edu/societal_{model_clean}.png"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    attribute_types = [
        "Gender Identity",
        "Sexual Orientation",
    ]

    model_names = [
        "msra-gpt-4o",
        "Qwen3-235B-A22B-Instruct-2507",
        "Qwen3-Next-80B-A3B-Instruct",
        "Llama-3.3-70B-Instruct",
        "gpt-oss-120b",
        "gemma-3-27b-it",
        "NVIDIA-Nemotron-Nano-12B-v2",
        "GLM-4.5-Air",
    ]

    for model_name in model_names:
        all_results = {}
        for attribute_type in attribute_types:
            file_name = f"outputs/edu/societal/{attribute_type}/{model_name}.jsonl"
            if os.path.exists(file_name):
                all_results[attribute_type] = compute_results(attribute_type, file_name)

        if not all_results:
            print(f"No results found for model {model_name}, skip plotting.")
            continue

        # Only plot attribute types that have data
        attribute_types_present = [a for a in attribute_types if a in all_results]
        draw_results(all_results, attribute_types_present, model_name)
