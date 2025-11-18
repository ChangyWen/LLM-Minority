import json
import random


save_file = "dataset/resumes_shuffled.jsonl"
all_items = []
with open("dataset/resumes.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        all_items.append(item)

random.shuffle(all_items)

with open(save_file, "w") as f:
    for item in all_items:
        f.write(json.dumps(item) + "\n")