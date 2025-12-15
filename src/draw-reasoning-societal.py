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
from scipy import stats


type_to_minority_attributes = {
    "Gender Identity": ["Transgender", "Non-binary"],
    "Sexual Orientation": ["Homosexual", "Bisexual", "Asexual"],
    "Disability Status": ["Colorblindness", "Hearing Impairment", "Mobility Impairment"],
    "Chronic Health Condition Status": ["HIV Positive", "Chronic Hepatitis", "Type 1 Diabetes", "Asthma"],
    "Religious Affiliation": ["Jewish", "Jain", "Taoist"],
    "Political Affiliation": ["Green Party", "Libertarian"],
    "Race": ["Black"],
}


def t_ci(scores, confidence=0.95):
    mean = np.mean(scores)
    sem = stats.sem(scores)
    df = len(scores) - 1
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    lower = mean - t_crit * sem
    upper = mean + t_crit * sem
    return (mean, lower, upper)


def bootstrap_relative_diff_ci(minority_scores, majority_scores,
                               n_bootstrap=5000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)

    deltas = []

    n_min = len(minority_scores)
    n_maj = len(majority_scores)

    for _ in range(n_bootstrap):
        min_sample = rng.choice(minority_scores, size=n_min, replace=True)
        maj_sample = rng.choice(majority_scores, size=n_maj, replace=True)

        min_mean = np.mean(min_sample)
        maj_mean = np.mean(maj_sample)

        # avoid pathological division
        if maj_mean == 0:
            continue

        delta = (min_mean - maj_mean) / maj_mean
        deltas.append(delta)

    deltas = np.array(deltas)

    lower = np.percentile(deltas, 100 * alpha / 2)
    upper = np.percentile(deltas, 100 * (1 - alpha / 2))

    return np.mean(deltas), lower, upper


def compute_results(file_name, attribute_type):
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

    delta_mean, delta_ci_low, delta_ci_high = bootstrap_relative_diff_ci(
        minority_scores, majority_scores
    )

    return {
        "delta": delta_mean,
        "ci_low": delta_ci_low,
        "ci_high": delta_ci_high,
    }


if __name__ == "__main__":
    applications = ["edu", "hiring", "loan"]

    model_names = [
        "GLM-4.5-Air",
        "GLM-4.5-Air_no_thinking",
        "NVIDIA-Nemotron-Nano-12B-v2",
        "NVIDIA-Nemotron-Nano-12B-v2_no_thinking",
    ]

    attribute_types = ["Gender Identity", "Sexual Orientation"]

    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(float))
        for application in applications:
            for model_name in model_names:
                file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"
                if not os.path.exists(file_name):
                    file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}.jsonl"
                if not os.path.exists(file_name):
                    assert False, f"File not found: {application} {attribute_type} {model_name}"
                delta = compute_results(file_name, attribute_type)
                application_to_model_to_delta[application][model_name] = delta

        draw_results_by_application(
            application_to_model_to_delta=application_to_model_to_delta,
            attribute_type=attribute_type,
        )