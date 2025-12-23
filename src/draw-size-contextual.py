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
        return ""


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


def iut_pvalue_by_ratio(counts_size5, counts_size10, ratios=("20%","40%","60%","80%")):
    per_ratio_p = {}

    for r in ratios:
        if r not in counts_size5 or r not in counts_size10:
            continue  # or raise if you expect all ratios

        d5,  se5  = norm_delta_and_se(counts_size5[r],  context_size=5)
        d10, se10 = norm_delta_and_se(counts_size10[r], context_size=10)

        p = one_sided_p(d10, se10, d5, se5)  # tests d10 > d5
        per_ratio_p[r] = p

    if len(per_ratio_p) == 0:
        return {}, float("nan")

    return per_ratio_p


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


def draw_results_by_application(application_to_model_to_delta, application_to_model_to_pvalue, attribute_type, model_names, context_sizes):
    """
    One figure per application.
    8 subplots (2x4) in the order of `model_names`.
    Each subplot: x = contextual ratio (20%, 40%, 60%, 80%),
    two lines for context sizes (e.g., 5 and 10),
    y = delta (with 95% CI).
    """

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
    })
    sns.set_theme(style="whitegrid")

    applications = sorted(application_to_model_to_delta.keys())
    context_sizes = list(sorted(context_sizes))

    # X-axis contextual ratios (fixed as requested)
    ratio_strs = ["20%", "40%", "60%", "80%"]
    ratio_x = np.array([20, 40, 60, 80], dtype=float)

    # Modern colors for context sizes (2 colors)
    # (You can swap to "colorblind" if you prefer)
    cs_palette = sns.color_palette("tab10", n_colors=max(2, len(context_sizes)))
    context_size_to_color = {cs: cs_palette[i] for i, cs in enumerate(context_sizes)}

    for application in applications:
        fig, axes = plt.subplots(
            2, 4,
            figsize=(16.5, 7.2),
            sharex=False,
            sharey=False
        )
        axes = axes.flatten()

        for i, model_name in enumerate(model_names):
            ax = axes[i]

            # Remove upper and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Draw one line per context size
            for cs in context_sizes:
                if (
                    application not in application_to_model_to_delta
                    or model_name not in application_to_model_to_delta[application]
                    or cs not in application_to_model_to_delta[application][model_name]
                ):
                    continue

                # This is the dict returned by compute_results: keys are "20%","40%","60%","80%"
                delta_by_ratio = application_to_model_to_delta[application][model_name][cs]

                xs, ys, yerr_lo, yerr_hi = [], [], [], []
                for rx, rstr in zip(ratio_x, ratio_strs):
                    if rstr not in delta_by_ratio:
                        continue
                    d = delta_by_ratio[rstr]
                    y = float(d["delta"])
                    lo = float(d["ci_low"])
                    hi = float(d["ci_high"])

                    xs.append(rx)
                    ys.append(y)
                    yerr_lo.append(y - lo)
                    yerr_hi.append(hi - y)

                if len(xs) == 0:
                    continue

                yerr = np.vstack([yerr_lo, yerr_hi])  # (2, N) asymmetric

                ax.errorbar(
                    xs, ys,
                    yerr=yerr,
                    fmt="-o",
                    markersize=6.5,
                    capsize=4.0,
                    elinewidth=1.5,
                    linewidth=1.6,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    color=context_size_to_color[cs],
                    label=f"ctx={cs}",
                    zorder=3,
                )

            model_name = model_name.replace("msra-", "")
            ax.set_title(model_name, fontweight="bold", pad=8, fontsize=16)

            # X-axis formatting: 20/40/60/80
            # ----- add stars on xticks based on per-ratio p-values -----
            per_ratio_p = {}
            if (
                application_to_model_to_pvalue is not None
                and application in application_to_model_to_pvalue
                and model_name in application_to_model_to_pvalue[application]
                and isinstance(application_to_model_to_pvalue[application][model_name], dict)
            ):
                per_ratio_p = application_to_model_to_pvalue[application][model_name]

            tick_labels = []
            for r in ratio_strs:
                p = per_ratio_p.get(r, float("nan"))
                if i <= 3:
                    tick_labels.append(f"{p_to_stars(p)}")
                else:
                    tick_labels.append(f"{p_to_stars(p)}\n{r}")

            ax.set_xticks(ratio_x)
            ax.set_xticklabels(tick_labels)
            ax.set_xlim(10, 90)

            # Keep your y-axis style/meaning unchanged
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))

            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)

            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)

            # Per-subplot legend (small + unobtrusive)
            ax.legend(frameon=False, fontsize=10, loc="best", handlelength=2.0)

        # If fewer than 8 models, hide unused axes
        for j in range(len(model_names), len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"{application.capitalize()} - {attribute_type}",
            fontsize=16,
            fontweight="bold",
            y=0.95
        )

        fig.supylabel(
            "Norm. Abs. Diff. of Selection Rate\n(Δ / random-rate)",
            fontweight="bold",
            fontsize=16,
            x=0.03,
            ha="center"
        )

        fig.supxlabel(
            "Contextual Ratio",
            fontweight="bold",
            fontsize=16,
            y=0.04
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = f"outputs/size/{application}_{attribute_type}-v2.png"
        plt.savefig(out_path, dpi=512, bbox_inches="tight")
        print(f"Saved subplot grid to {out_path}")
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

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(dict))
        application_to_model_to_pvalue = defaultdict(lambda: defaultdict(dict))
        for application in applications:
            for model_name in model_names:
                counts_size5 = None
                counts_size10 = None
                for context_size in context_sizes:
                    if "no_thinking" in model_name:
                        file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name[:-12]}_{context_size}_500_no_thinking.jsonl"
                        if not os.path.exists(file_name):
                            file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name[:-12]}_{context_size}_200_no_thinking.jsonl"
                    else:
                        file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_{context_size}_500.jsonl"
                        if not os.path.exists(file_name):
                            file_name = f"outputs/{application}/contextual/{attribute_type}/{model_name}_{context_size}_200.jsonl"
                    if not os.path.exists(file_name):
                        continue
                        # raise FileNotFoundError(f"File not found: {application} {attribute_type} {model_name}")

                    delta = compute_results(file_name, context_size)
                    raw_data = dict([(k, v["raw_data"]) for k, v in delta.items()])
                    if context_size == 5:
                        counts_size5 = raw_data
                    else:
                        counts_size10 = raw_data
                    application_to_model_to_delta[application][model_name][context_size] = delta
                per_ratio_p = iut_pvalue_by_ratio(counts_size5, counts_size10)
                application_to_model_to_pvalue[application][model_name] = per_ratio_p
                print(f"attribute_type: {attribute_type}, application: {application}, model_name: {model_name}, per_ratio_p: {per_ratio_p}")

        draw_results_by_application(
            application_to_model_to_delta=application_to_model_to_delta,
            application_to_model_to_pvalue=application_to_model_to_pvalue,
            attribute_type=attribute_type,
            model_names=model_names,
            context_sizes=context_sizes,
        )