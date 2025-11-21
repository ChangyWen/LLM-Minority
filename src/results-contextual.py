import json
import sys
from collections import defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency, norm
import matplotlib.gridspec as gridspec


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
    Returns: chi2, p_value, dof
    """
    table = []
    levels = sorted(attr_counts.keys())
    for c in levels:
        hit, total = attr_counts[c]
        miss = total - hit
        table.append([hit, miss])

    chi2, p, dof, expected = chi2_contingency(table)
    return chi2, p, dof, levels


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


def compute_results(file_name, attribute_type):

    attr_value_to_results = defaultdict(lambda: {
        "same_attr_count_to_count": defaultdict(int),
        "same_attr_count_to_hit_count": defaultdict(int),
    })
    n_trials = 0

    with open(file_name, "r") as f:
        for line in f:
            n_trials += 1

            item = json.loads(line)
            attributes = item["attributes"]
            suggested_candidate_id = item["suggested_candidate_id"]

            for inner_idx, attr_value in enumerate(attributes):
                same_attr_count = attributes.count(attr_value) - 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_hit_count"][same_attr_count] += (1 if inner_idx == suggested_candidate_id else 0)

                attr_value_to_results[attr_value]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results[attr_value]["same_attr_count_to_hit_count"][same_attr_count] += (1 if inner_idx == suggested_candidate_id else 0)

    print(f"Attribute type: {attribute_type}")
    results = {}
    significance = {}
    for attr_value, attr_value_results in attr_value_to_results.items():
        # sort the attr_value_results by same_attr_count
        print(f"attr_value: {attr_value}")
        results[attr_value] = {}
        significance[attr_value] = {}

        # store raw counts for global and pairwise tests
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

        # ---------- Pairwise tests (two-proportion z-tests) ----------
        levels = sorted(levels)
        print("[Pairwise tests: two-proportion z-test]")
        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                c1 = levels[i]
                c2 = levels[j]
                h1, n1 = attr_counts[c1]
                h2, n2 = attr_counts[c2]
                z, p_pair = two_proportion_z_test(h1, n1, h2, n2)
                significance[attr_value][str((c1, c2))] = p_pair

                print(f"[Pairwise test] {c1} vs {c2}: p-value={p_pair:.6g}")

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

    non_all_values = set(all_results.keys()) - {"all_attr_values"}
    attribute_values = sorted(list(non_all_values)) + ["all_attr_values"]
    palette = sns.color_palette("husl", len(attribute_values))

    fig = plt.figure(dpi=1024)

    # -----------------------------------------------
    # Create a main title ABOVE the brackets panel
    # -----------------------------------------------
    model_name_clean = model_name.replace("msra-", "")
    fig.suptitle(
        f"{attribute_type} ({model_name_clean})",
        fontweight="bold",
        y=0.93        # push title higher
    )

    # -----------------------------------------------
    # Now the figure has THREE vertical regions:
    # suptitle (automatic)
    # brackets panel   (gs row 0)
    # main plot        (gs row 1)
    # -----------------------------------------------
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[0.32, 1.0],   # give brackets a bit more room
        hspace=0.05                  # decrease spacing
    )

    ax_brackets = fig.add_subplot(gs[0])
    ax_main = fig.add_subplot(gs[1])

    # -----------------------------------------------------------
    # MAIN PLOT: Exactly your original code (slightly adapted)
    # -----------------------------------------------------------

    baseline_value = 0
    xticks = []
    all_barlines = []
    line_handles = []
    legend_labels = []
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
            linestyle="--" if attribute_value != "all_attr_values" else "-",
            color=palette[i],
            capsize=6,
            capthick=1.5,
        )

        if attribute_value != "all_attr_values":
            all_barlines.extend(barlines)

        # Legend label with global significance
        base_label = attribute_value if attribute_value != "all_attr_values" else "All"
        p_global = significance.get(attribute_value, {}).get("global_test_p_value", float("nan"))
        stars = p_to_stars(p_global)
        label = f"{base_label} {stars}" if stars else base_label
        line_handles.append(line)
        legend_labels.append(label)

    for bar in all_barlines:
        bar.set_linestyle("--")
        bar.set_linewidth(1.2)

    # Draw the baseline WITH a legend label
    ax_main.axhline(
        y=baseline_value,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"Random ({baseline_value:.2f})"
    )

    ax_main.set_xticks(xticks)
    ax_main.set_xlim(-0.1, len(xticks) - 1 + 0.1)

    ax_main.set_xlabel("Number of same-attribute candidates", fontsize=11, fontweight="bold")
    ax_main.set_ylabel("Selection rate", fontsize=11, fontweight="bold")

    ax_main.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax_main.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax_main.spines[spine].set_visible(False)

    handles, labels = ax_main.get_legend_handles_labels()

    ax_main.legend(
        line_handles + handles,
        legend_labels + labels,
        fontsize=12,
        title_fontsize=13,
        markerscale=1,
    )

    # -----------------------------------------------------------
    # BRACKET PANEL (Top): ONLY brackets + stars here
    # -----------------------------------------------------------

    # Turn off all axis decorations
    ax_brackets.set_xlim(ax_main.get_xlim())
    ax_brackets.set_ylim(0, 1)
    ax_brackets.axis("off")

    # -----------------------------------------------------------
    # Separate brackets BY ATTRIBUTE GROUP (legend order)
    # -----------------------------------------------------------
    attr_to_pairs = {attr_value: [] for attr_value in attribute_values}

    for attr_value in attribute_values:
        res = all_results[attr_value]
        sig_dict = significance.get(attr_value, {})
        pairs = []

        for key, p_val in sig_dict.items():
            if key == "global_test_p_value":
                continue
            stars = p_to_stars(p_val)
            if not stars:
                continue

            s = key.strip()[1:-1]
            c1, c2 = map(int, s.split(","))
            if c1 > c2:
                c1, c2 = c2, c1
            if c1 in res and c2 in res:
                pairs.append((c1, c2, stars))

        # Within each attribute group, sort so largest span |c1-c2| on top
        pairs.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
        attr_to_pairs[attr_value] = pairs

    # Count total rows required
    total_rows = sum(len(v) for v in attr_to_pairs.values())

    if total_rows == 0:
        fig.tight_layout()
        fig.savefig(f"outputs/contextual_{model_name_clean}_{attribute_type}_{resume_count}.png", bbox_inches="tight")
        plt.close(fig)
        return

    # -----------------------------------------------------------
    # Compute row geometry
    # -----------------------------------------------------------
    row_step = 1.0 / (total_rows + 1)
    bracket_height = 0.35 * row_step  # thickness of bracket
    current_row = 0

    # -----------------------------------------------------------
    # Draw brackets GROUP-BY-GROUP in legend order
    # -----------------------------------------------------------
    for attr_value in attribute_values:
        pairs = attr_to_pairs[attr_value]
        color = attr_to_color[attr_value]

        for (c1, c2, stars) in pairs:
            # Compute row placement
            y_bottom = 1 - (current_row + 1) * row_step
            y_top = y_bottom + bracket_height

            # Draw bracket
            ax_brackets.plot(
                [c1, c1, c2, c2],
                [y_bottom, y_top, y_top, y_bottom],
                color=color,
                linewidth=1.2,
            )

            # Draw stars INSIDE the bracket (vertically centered)
            ax_brackets.text(
                (c1 + c2) / 2,
                y_bottom + bracket_height / 2,
                stars,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
            )

            current_row += 1

    fig.tight_layout()
    save_file = f"outputs/contextual_{model_name_clean}_{attribute_type}_{resume_count}.png"
    fig.savefig(save_file, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pool_count = 200

    for attribute_type in ["Gender"]:
        for resume_count in [5]:
            for model_name in ["msra-gpt-4o", "msra-gpt-4.1-nano", "Qwen3-Next-80B-A3B-Instruct"]:
                file_name = f"outputs/contextual/{attribute_type}/{model_name}_{resume_count}_{pool_count}.jsonl"
                if os.path.exists(file_name):
                    print(f"------------------------------------\n\n{file_name}")
                    results, significance, n_trials = compute_results(file_name, attribute_type)
                    draw_results(model_name, attribute_type, resume_count, results, significance)
