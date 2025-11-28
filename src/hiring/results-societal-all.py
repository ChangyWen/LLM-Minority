import json
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import binomtest
from scipy import stats


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
    "Disability Status": ["Colorblindness", "Hearing Impairment", "Mobility Impairment"],
    "Chronic Health Condition Status": ["HIV Positive", "Chronic Hepatitis", "Type 1 Diabetes", "Asthma"],
}


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
    }


if __name__ == "__main__":

    attribute_types = ["Gender Identity", "Sexual Orientation", "Disability Status", "Chronic Health Condition Status"]

    for model_name in ["msra-gpt-4o", "Qwen3-Next-80B-A3B-Instruct", "Llama-3.3-70B-Instruct", "gemma-3-27b-it"]:
        all_results = {}
        for attribute_type in attribute_types:
            file_name = f"outputs/hiring/societal/{attribute_type}/{model_name}.jsonl"
            if os.path.exists(file_name):
                results = compute_results(attribute_type, file_name)
                if model_name not in all_results:
                    all_results[model_name] = {}
                all_results[model_name][attribute_type] = results

