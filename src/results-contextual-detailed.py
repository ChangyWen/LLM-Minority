import json
import sys
from collections import defaultdict
import math
import seaborn as sns
import matplotlib.pyplot as plt


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


def compute_results(attribute_type, resume_count):
    total_count = 0

    # check the total count of the file
    with open(f"outputs/contextual/{attribute_type}/consultant_samples_{resume_count}.jsonl", "r") as f:
        total_count = sum(1 for _ in f)
    part_size = total_count // 6

    attr_value_to_results = defaultdict(lambda: {
        "same_attr_count_to_count": defaultdict(int),
        "same_attr_count_to_hit_count": defaultdict(int),
    })

    for anchor_index in range(6):
        start_index = anchor_index * part_size
        end_index = start_index + part_size - 1
        # print(f"part {anchor_index}: {end_index - start_index + 1}; start_index: {start_index}, end_index: {end_index}")

        cur_index = 0
        with open(f"outputs/contextual/{attribute_type}/consultant_samples_{resume_count}.jsonl", "r") as f:
            for line in f:
                if cur_index < start_index or cur_index > end_index:
                    cur_index += 1
                    continue
                cur_index += 1
                item = json.loads(line)
                attributes = item["attributes"]
                hit_candidate_id = item["hit_candidate_id"]
                candidate_order = item["candidate_order"]

                if anchor_index not in candidate_order:
                    continue

                anchor_index_attr_value = attributes[candidate_order.index(anchor_index)]

                # record the results for the anchor attribute
                same_attr_count = attributes.count(anchor_index_attr_value) - 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results["all_attr_values"]["same_attr_count_to_hit_count"][same_attr_count] += (1 if anchor_index == hit_candidate_id else 0)

                attr_value_to_results[anchor_index_attr_value]["same_attr_count_to_count"][same_attr_count] += 1
                attr_value_to_results[anchor_index_attr_value]["same_attr_count_to_hit_count"][same_attr_count] += (1 if anchor_index == hit_candidate_id else 0)

    print(f"Attribute type: {attribute_type}")
    results = {}
    for attr_value, attr_value_results in attr_value_to_results.items():
        # sort the attr_value_results by same_attr_count
        print(f"attr_value: {attr_value}")
        results[attr_value] = {}
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

    return results


if __name__ == "__main__":

    for attribute_type in ["Race", "Gender", "Religious Affiliation", "Gender Identity", "Sexual Orientation"]:
        for resume_count in [6]:
            results = compute_results(attribute_type, resume_count)
