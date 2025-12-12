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
from collections import defaultdict


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

    minority_mean, _, _ = t_ci(minority_scores)
    majority_mean, _, _ = t_ci(majority_scores)
    delta = (minority_mean - majority_mean) / majority_mean

    return delta


def draw_results(application_to_model_to_delta, attribute_type):
    #### TODO: draw a radar chart for the application_to_model_to_delta
    pass


if __name__ == "__main__":
    applications = ["edu", "hiring", "loan"]

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

    attribute_types = ["Gender Identity", "Sexual Orientation"]


    for attribute_type in attribute_types:
        application_to_model_to_delta = defaultdict(lambda: defaultdict(float))
        for application in applications:
            for model_name in model_names:
                file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}_5_200.jsonl"
                if not os.path.exists(file_name):
                    file_name = f"outputs/{application}/societal/{attribute_type}/{model_name}_5_500.jsonl"
                if not os.path.exists(file_name):
                    assert False, f"File not found: {application} {attribute_type} {model_name}"
                delta = compute_results(file_name, attribute_type)
                application_to_model_to_delta[application][model_name] = delta

        draw_results(application_to_model_to_delta, attribute_type)
