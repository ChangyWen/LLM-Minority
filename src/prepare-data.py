# from datasets import load_dataset

# ds = load_dataset("opensporks/resumes")

# for item in ds["train"]:
#     idx = str(item["ID"])
#     resume = item["Resume_str"]
#     print(f"{idx}\n{resume}\n\n")
#     input()

import pandas as pd
import json

# df = pd.read_csv("dataset/Resume.csv", encoding="utf-8")

# save_file = "dataset/resumes.jsonl"
# for index, row in df.iterrows():
#     idx = str(row["ID"]).strip()
#     job = row["Category"].strip()
#     resume = row["Resume_str"].strip()

#     if len(job) <= 0: continue
#     if len(resume) <= 500: continue

#     with open(save_file, "a") as f:
#         f.write(json.dumps({
#             "idx": idx,
#             "job": job,
#             "resume": resume
#         }) + "\n")


with open("dataset/consultant_samples_paraphrased.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        resumes = item["paraphrased_resumes"]
        for resume in resumes:
            print(resume)
            input()

# with open("dataset/consultant_samples_paraphrased.json", "r") as f:
#     data = json.load(f)
#     with open("dataset/consultant_samples_paraphrased.jsonl", "w") as f:
#         f.write(json.dumps(data) + "\n")