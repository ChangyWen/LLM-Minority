import json
import sys
from collections import defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, norm
import statsmodels.api as sm
import numpy as np


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


def chi2_test_same_attr_effect(attr_counts):
    """
    attr_counts: dict[int -> (hit_count, total_count)]
                 e.g. {0: (h0, n0), 1: (h1, n1), ...}
    Returns: chi2, p_value, dof, levels
    """
    table = []
    levels = sorted(attr_counts.keys())
    for c in levels:
        hit, total = attr_counts[c]
        miss = total - hit
        table.append([hit, miss])

    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, dof, levels


def cochran_armitage_trend(attr_counts):
    """
    Cochran–Armitage trend test for ordered proportions.

    returns:
        z                # test statistic
        p_two_sided
        p_one_inc        # one-sided p for INCREASING trend
        p_one_dec        # one-sided p for DECREASING trend
    """
    levels = sorted(attr_counts.keys())
    if not levels:
        return float("nan"), float("nan"), float("nan"), float("nan")

    scores = [float(c) for c in levels]
    hits = [attr_counts[c][0] for c in levels]
    totals = [attr_counts[c][1] for c in levels]

    N = sum(totals)
    X = sum(hits)
    if N == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    p_hat = X / N
    T = sum(w * (x - p_hat * n) for w, x, n in zip(scores, hits, totals))
    sum_nw = sum(n * w for n, w in zip(totals, scores))
    sum_nw2 = sum(n * (w ** 2) for n, w in zip(totals, scores))

    var_T = p_hat * (1 - p_hat) * (sum_nw2 - (sum_nw ** 2) / N)
    if var_T <= 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    z = T / math.sqrt(var_T)

    p_two = 2 * (1 - norm.cdf(abs(z)))
    p_inc = 1 - norm.cdf(z)   # one-sided increasing
    p_dec = norm.cdf(z)       # one-sided decreasing

    return z, p_two, p_inc, p_dec


def trend_test_delta_counts(attr_counts_A, attr_counts_B):
    """
    Tests whether delta(c) = pA(c) - pB(c) changes systematically with c
    using a logistic regression with interaction term group*c.

    Returns: z, p_two_sided, p_one_inc, p_one_dec
    """
    print("A counts:", attr_counts_A)
    print("B counts:", attr_counts_B)
    for c in sorted(set(attr_counts_A) & set(attr_counts_B)):
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]
        print(c, " A:", hA, "/", nA, "=", hA/nA, "  B:", hB, "/", nB, "=", hB/nB)

    levels = sorted(set(attr_counts_A) & set(attr_counts_B))
    rows = []

    for c in levels:
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]
        # group = 1 for A, 0 for B
        rows.append([1, c, hA, nA])  # A
        rows.append([0, c, hB, nB])  # B

    rows = np.array(rows, dtype=float)
    group = rows[:, 0]
    cvals = rows[:, 1]
    hits = rows[:, 2]
    totals = rows[:, 3]

    # endog must be proportion in [0,1] when using freq_weights with Binomial
    y = hits / totals
    w = totals

    # design matrix: intercept + group + c + group*c
    X = np.column_stack([np.ones_like(group), group, cvals, group * cvals])

    model = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w)
    result = model.fit()

    beta3 = result.params[3]   # interaction coefficient
    se3 = result.bse[3]
    z = beta3 / se3

    # two-sided and one-sided p-values for the interaction
    p_two = 2 * (1 - norm.cdf(abs(z)))
    p_inc = 1 - norm.cdf(z)   # delta increases with c
    p_dec = norm.cdf(z)       # delta decreases with c

    print(result.summary())   # very helpful to inspect once
    print("beta3, se3, z:", beta3, se3, z)

    return z, p_two, p_inc, p_dec


def two_proportion_z_test(x1, n1, x2, n2):
    """
    Two-sided z-test for equality of two proportions.
    H0: p1 == p2
    H1: p1 != p2

    Returns: z, p_two_sided
    """
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    # standard error under H0
    se = math.sqrt(p_pool * (1 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        # no variability: proportions are identical or degenerate
        return float("nan"), 1.0

    z = (p1 - p2) / se
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return z, p


def compute_results(file_name, attribute_type, max_n_trials=100000):

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
                attr_value_to_results[attr_value]["same_attr_count_to_hit_count"][same_attr_count] += (1 if inner_idx == suggested_candidate_id else 0)

    print(f"Attribute type: {attribute_type}")
    results = {}
    significance = {}
    attr_counts_A = None
    attr_counts_B = None
    for attr_value, attr_value_results in attr_value_to_results.items():
        # sort the attr_value_results by same_attr_count
        print(f"attr_value: {attr_value}")
        results[attr_value] = {}
        significance[attr_value] = {}

        # store raw counts for global and trend tests
        attr_counts = {}

        # sort attr_value_results["same_attr_count_to_count"]
        same_attr_count_to_count = dict(sorted(attr_value_results["same_attr_count_to_count"].items(), key=lambda x: x[0]))
        for same_attr_count, count in same_attr_count_to_count.items():
            hit_count = attr_value_results["same_attr_count_to_hit_count"][same_attr_count]
            hit_rate = hit_count / count
            ci_low, ci_high = wilson_ci(hit_count, count)
            print(f"same_attr_count: {same_attr_count}, total: {count}, hit_rate: {hit_rate:.6f} [{ci_low:.6f}, {ci_high:.6f}]")
            results[attr_value][same_attr_count] = {
                "hit_rate": hit_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

            attr_counts[same_attr_count] = (hit_count, count)

        # ---------- Global test (χ²) ----------
        chi2, p_global, dof, levels = chi2_test_same_attr_effect(attr_counts)
        print(f"[Global test] p-value={p_global:.6g}")
        significance[attr_value]["global_test_p_value"] = p_global

        # ---------- Cochran–Armitage trend test ----------
        z, p_two, p_inc, p_dec = cochran_armitage_trend(attr_counts)
        print(f"[Cochran-Armitage trend test] z={z:.6g}, p-value={p_two:.6g}, p-inc={p_inc:.6g}, p-dec={p_dec:.6g}")
        significance[attr_value]["p_value_two_sided"] = p_two
        significance[attr_value]["p_value_one_inc"] = p_inc
        significance[attr_value]["p_value_one_dec"] = p_dec

        # attr_counts_A/B for delta trend test
        if attr_value == "Black" or attr_value == "Female":
            attr_counts_A = attr_counts
        else:
            attr_counts_B = attr_counts

    z, p_two, p_inc, p_dec = trend_test_delta_counts(attr_counts_A, attr_counts_B)
    print(f"[Delta Trend test] z={z:.6g}, p-value={p_two:.6g}, p-inc={p_inc:.6g}, p-dec={p_dec:.6g}")
    significance["delta"] = {
        "p_value_two_sided": p_two,
        "p_value_one_inc": p_inc,
        "p_value_one_dec": p_dec,
    }

    results["delta"] = {}
    for c in sorted(set(attr_counts_A) & set(attr_counts_B)):
        hA, nA = attr_counts_A[c]
        hB, nB = attr_counts_B[c]
        pA = hA / nA
        pB = hB / nB
        delta = pA - pB
        ciA_low, ciA_high = wilson_ci(hA, nA)
        ciB_low, ciB_high = wilson_ci(hB, nB)
        ci_low = ciA_low - ciB_high
        ci_high = ciA_high - ciB_low
        results["delta"][c] = {
            "delta": delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return results, significance, n_trials


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


def draw_results(model_name, attribute_type, resume_count, all_results, significance):

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.edgecolor": "gray",
        "axes.linewidth": 0.8,
    })

    non_delta_values = set(all_results.keys()) - {"delta"}
    attribute_values = sorted(list(non_delta_values))
    palette = sns.color_palette("husl", len(attribute_values))

    delta_color = "blue"      # fixed color for Δ

    # -----------------------------------------------------------
    # Create figure and MAIN AXIS (no bracket panel)
    # -----------------------------------------------------------
    fig, ax_main = plt.subplots(dpi=1024)

    model_name_clean = model_name.replace("msra-", "")
    fig.suptitle(
        f"Hiring - {attribute_type} ({model_name_clean})",
        fontweight="bold",
        y=0.93
    )

    # -----------------------------------------------------------
    # MAIN PLOT (per-attribute selection rates)
    # -----------------------------------------------------------

    baseline_value = 0
    xticks = []
    all_barlines = []
    attr_to_color = {}

    for i, attribute_value in enumerate(attribute_values):
        res = all_results[attribute_value]
        attr_to_color[attribute_value] = palette[i]

        xs = sorted(res.keys())
        xticks = xs
        baseline_value = 1 / len(xs)
        ys = [res[x]["hit_rate"] for x in xs]

        lower_err = [res[x]["hit_rate"] - res[x]["ci_low"] for x in xs]
        upper_err = [res[x]["ci_high"] - res[x]["hit_rate"] for x in xs]
        yerr = [lower_err, upper_err]

        line, caplines, barlines = ax_main.errorbar(
            xs, ys, yerr=yerr,
            marker="o",
            markersize=6,
            linewidth=1.5,
            linestyle="-",
            color=palette[i],
            capsize=6,
            capthick=1.5,
        )
        all_barlines.extend(barlines)

        # Legend label with directional stars (trend test)
        p_one_inc = significance.get(attribute_value, {}).get("p_value_one_inc", float("nan"))
        p_one_dec = significance.get(attribute_value, {}).get("p_value_one_dec", float("nan"))

        if attribute_value in ("Male", "White"):
            stars = p_to_stars(p_one_inc)
            stars = f"↑{stars}" if stars else ""
        elif attribute_value in ("Female", "Black"):
            stars = p_to_stars(p_one_dec)
            stars = f"↓{stars}" if stars else ""
        else:
            raise ValueError(f"Unknown attribute value: {attribute_value}")

        label = f"{attribute_value} {stars}" if stars else attribute_value
        line.set_label(label)

    # style group CI barlines
    for bar in all_barlines:
        bar.set_linestyle("-")
        bar.set_linewidth(1.2)

    # baseline with legend entry
    ax_main.axhline(
        y=baseline_value,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"Random ({baseline_value:.1f})"
    )

    ax_main.set_xticks(xticks, labels=[f"{(c + 1)/(len(xticks)) * 100:.0f}%" for c in xticks])
    ax_main.set_xlim(-0.1, len(xticks) - 1 + 0.1)

    ax_main.set_xlabel("Same-attribute Ratio", fontsize=11, fontweight="bold")
    ax_main.set_ylabel("Selection Rate", fontsize=11, fontweight="bold")

    ax_main.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_main.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax_main.spines[spine].set_visible(False)

    # -----------------------------------------------------------
    # DELTA LINE on twin y-axis (with marker + CI)
    # -----------------------------------------------------------
    ax_delta = None
    if "delta" in all_results:
        ax_delta = ax_main.twinx()

        delta_res = all_results["delta"]
        xs_delta = sorted(delta_res.keys())
        ys_delta = [delta_res[c]["delta"] for c in xs_delta]

        # Wilson-based CIs already stored
        lower_err_delta = [ys_delta[i] - delta_res[c]["ci_low"] for i, c in enumerate(xs_delta)]
        upper_err_delta = [delta_res[c]["ci_high"] - ys_delta[i] for i, c in enumerate(xs_delta)]
        yerr_delta = [lower_err_delta, upper_err_delta]

        # stars from trend_test_p_value_one_dec (delta trend)
        delta_stars = p_to_stars(significance.get("delta", {}).get("p_value_one_dec", float("nan")))
        delta_stars = f"↓{delta_stars}" if delta_stars else ""
        delta_label_pre = "(F. - M.)" if attribute_type == "Gender" else "(B. - W.)"
        delta_label = r"$\Delta $" + delta_label_pre
        if delta_stars:
            delta_label += f" {delta_stars}"

        # errorbar with marker + CI
        line_delta, cap_delta, bar_delta = ax_delta.errorbar(
            xs_delta,
            ys_delta,
            yerr=yerr_delta,
            marker="s",
            markersize=5,
            linestyle="--",
            linewidth=1.5,
            color=delta_color,
            capsize=5,
            capthick=1.3,
        )
        for bar in bar_delta:
            bar.set_linestyle("--")
        # ensure legend shows only the delta marker line
        line_delta.set_label(delta_label)
        for bar in bar_delta:
            bar.set_label(None)
        for cap in cap_delta:
            cap.set_label(None)

        # axis styling
        ax_delta.set_ylabel(
            r"$\Delta$ in Selection Rate " + delta_label_pre,
            fontsize=11,
            fontweight="bold",
            color=delta_color
        )
        ax_delta.tick_params(axis="y", labelcolor=delta_color)

        # remove top spine (keep right to show twin axis)
        ax_delta.spines["top"].set_visible(False)

    # -----------------------------------------------------------
    # LEGEND (combine main + delta), top-right
    # -----------------------------------------------------------
    handles_main, labels_main = ax_main.get_legend_handles_labels()
    if ax_delta is not None:
        handles_delta, labels_delta = ax_delta.get_legend_handles_labels()
        handles = handles_main + handles_delta
        labels = labels_main + labels_delta
    else:
        handles, labels = handles_main, labels_main

    ax_main.legend(
        handles,
        labels,
        fontsize=12,
        title_fontsize=13,
        markerscale=1,
        loc="upper right",
        frameon=True,
        framealpha=0.5,
        borderpad=0.4,
    )

    save_file = f"outputs/hiring/contextual_{model_name_clean}_{attribute_type}_{resume_count}.png"
    fig.savefig(save_file, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pool_count = 200
    max_n_trials = 1000000

    for attribute_type in ["Gender", "Race"]:
        for resume_count in [5]:
            for model_name in [
                # "msra-gpt-4o",
                # "Qwen3-Next-80B-A3B-Instruct",
                # "Llama-3.3-70B-Instruct",
                # "gpt-oss-120b",
                # "GLM-4.5-Air",
                # "gemma-3-27b-it",
                "Seed-OSS-36B-Instruct",
                "Qwen3-235B-A22B-Instruct-2507",
                "NVIDIA-Nemotron-Nano-12B-v2",
            ]:
                file_name = f"outputs/hiring/contextual/{attribute_type}/{model_name}_{resume_count}_{pool_count}.jsonl"
                if os.path.exists(file_name):
                    print(f"------------------------------------\n\n{file_name}")
                    results, significance, n_trials = compute_results(file_name, attribute_type, max_n_trials)
                    draw_results(model_name, attribute_type, resume_count, results, significance)
