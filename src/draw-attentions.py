import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from collections import defaultdict
from scipy.stats import t  # for 95% CI


def remove_thinking_draft(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
        if len(text) > 0:
            return text
    if "</seed:think>" in text:
        text = text.split("</seed:think>")[-1].strip()
        if len(text) > 0:
            return text
    return text


def extract_from_tags(text, tag):
    if text is None:
        return None
    pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def get_results(uuid_to_data):
    male_count_to_attns = defaultdict(list)
    female_count_to_attns = defaultdict(list)

    for uuid, data in uuid_to_data.items():
        sentence_attention_file = data["sentence_attention_file"]
        sentences_file = data["sentences_file"]
        attributes = data["attributes"]

        with open(sentence_attention_file, "rb") as f:
            sentence_attention = torch.load(f)
        with open(sentences_file, "rb") as f:
            sentences = torch.load(f)

        output_sentence_id = None
        male_sentence_ids = []
        female_sentence_ids = []

        # find output sentence + gender sentences (search from the end, like your original)
        for sentence_id, sentence in enumerate(sentences[::-1]):
            sentence = sentence.strip()
            if "<suggested-candidate>" in sentence and "</suggested-candidate>" in sentence and output_sentence_id is None:
                try:
                    int(extract_from_tags(sentence, "suggested-candidate").strip())
                    output_sentence_id = len(sentences) - sentence_id - 1
                except:
                    pass
            if "Gender: Male" in sentence:
                male_sentence_ids.append(len(sentences) - sentence_id - 1)
            if "Gender: Female" in sentence:
                female_sentence_ids.append(len(sentences) - sentence_id - 1)

        if output_sentence_id is None:
            continue
        if len(male_sentence_ids) == 0 and len(female_sentence_ids) == 0:
            continue

        male_count = attributes.count("Male")
        female_count = attributes.count("Female")

        # collect attention weights: output_sentence attends to each gender sentence
        for male_sentence_id in male_sentence_ids:
            attn = sentence_attention[output_sentence_id, male_sentence_id]
            male_count_to_attns[male_count].append(float(attn))

        for female_sentence_id in female_sentence_ids:
            attn = sentence_attention[output_sentence_id, female_sentence_id]
            female_count_to_attns[female_count].append(float(attn))

    return male_count_to_attns, female_count_to_attns


def mean_ci95(values):
    """
    Return (mean, half_width) for 95% CI using Student-t.
    If n < 2, CI half_width = 0.
    """
    vals = np.asarray(values, dtype=float)
    n = vals.size
    if n == 0:
        return np.nan, np.nan
    m = float(np.mean(vals))
    if n < 2:
        return m, 0.0
    s = float(np.std(vals, ddof=1))
    se = s / np.sqrt(n)
    h = float(t.ppf(0.975, df=n - 1) * se)
    return m, h


if __name__ == "__main__":
    application = "loan"
    attribute_type = "Gender"
    total_count = 5
    pool_count = 500
    model_name = "google/gemma-3-27b-it"
    sub_model_name = model_name.split("/")[-1]

    save_dir = f"outputs/{application}/contextual/{attribute_type}/attentions/{sub_model_name}"

    # load data
    attentions_files = glob.glob(f"{save_dir}/sentence_attention_*.pt")
    sentences_files = glob.glob(f"{save_dir}/sentences_*.pt")

    uuid_to_data = defaultdict(lambda: {
        "sentence_attention_file": None,
        "sentences_file": None,
        "attributes": None,
    })

    for attentions_file in attentions_files:
        uuid = attentions_file.split("/")[-1].split("_")[-1][:-3]
        uuid_to_data[uuid]["sentence_attention_file"] = attentions_file

    for sentences_file in sentences_files:
        uuid = sentences_file.split("/")[-1].split("_")[-1][:-3]
        uuid_to_data[uuid]["sentences_file"] = sentences_file

    with open(f"{save_dir}/{total_count}_{pool_count}.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            uuid = item["uuid"]
            uuid_to_data[uuid]["attributes"] = item["attributes"]

    # filter incomplete
    uuid_to_del = set()
    for uuid, data in uuid_to_data.items():
        if data["sentence_attention_file"] is None or data["sentences_file"] is None or data["attributes"] is None:
            uuid_to_del.add(uuid)
    for uuid in uuid_to_del:
        del uuid_to_data[uuid]
    print("Loaded uuids:", len(uuid_to_data))

    # get results
    male_count_to_attns, female_count_to_attns = get_results(uuid_to_data)

    # x-axis fixed to attribute counts 1..5 (as you requested)
    x = list(range(1, total_count + 1))

    male_means, male_errs = [], []
    female_means, female_errs = [], []

    for c in x:
        m_mean, m_ci = mean_ci95(male_count_to_attns.get(c, []))
        f_mean, f_ci = mean_ci95(female_count_to_attns.get(c, []))
        male_means.append(m_mean)
        male_errs.append(m_ci)
        female_means.append(f_mean)
        female_errs.append(f_ci)

    # ---- plot one figure (line chart + error bars) ----
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Male line + error bars
    ax.errorbar(
        x, male_means, yerr=male_errs,
        marker="o", linestyle="-", linewidth=1.8, markersize=6,
        capsize=4, elinewidth=1.2,
        label="Male"
    )

    # Female line + error bars
    ax.errorbar(
        x, female_means, yerr=female_errs,
        marker="s", linestyle="-", linewidth=1.8, markersize=6,
        capsize=4, elinewidth=1.2,
        label="Female"
    )

    ax.set_xlabel("Same Attribute count")
    ax.set_ylabel("Attention weight")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    ax.set_title(f"{application} | {attribute_type} | {sub_model_name} (mean ± 95% CI)")

    plt.tight_layout()
    outpath = os.path.join(save_dir, f"attn_by_count_{total_count}_{pool_count}.png")
    plt.savefig(outpath, dpi=300)
    print("Saved:", outpath)
    plt.show()
