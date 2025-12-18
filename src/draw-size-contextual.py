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
        else:
            delta = pB - pA
            ciA_low, ciA_high = wilson_ci(hA, nA)
            ciB_low, ciB_high = wilson_ci(hB, nB)
            ci_low = ciB_low - ciA_high
            ci_high = ciB_high - ciA_low
        results["delta"][c] = {
            "delta": delta / random_selection_rate,
            "ci_low": ci_low / random_selection_rate,
            "ci_high": ci_high / random_selection_rate,
        }

    return results["delta"][1]


def draw_results_by_application(application_to_model_to_delta, attribute_type, model_names, context_sizes):
    pass


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

    attribute_types = ["Gender"]

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(dict))
        for application in applications:
            for model_name in model_names:
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
                        raise FileNotFoundError(f"File not found: {application} {attribute_type} {model_name}")

                    delta = compute_results(file_name, context_size)
                    application_to_model_to_delta[application][model_name] = delta

        draw_results_by_application(
            application_to_model_to_delta=application_to_model_to_delta,
            attribute_type=attribute_type,
            model_names=model_names,
            context_sizes=context_sizes,
        )