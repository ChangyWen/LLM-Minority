import json
import sys
from collections import defaultdict


if __name__ == "__main__":
    # {"genders": ["Female", "Male", "Male", "Female", "Female"], "candidate_order": [1, 2, 3, 4, 0], "suggested_candidate_id": 2, "hit_candidate_id": 3}

    anchor_index_to_results = {}

    total_count = 0
    attribute_type = sys.argv[1]

    # check the total count of the file
    with open(f"outputs/contextual/{attribute_type}/consultant_samples.jsonl", "r") as f:
        total_count = sum(1 for _ in f)
    print(f"total_count: {total_count}")
    # equally divide the total count into 5 parts
    part_size = total_count // 5
    print(f"part_size: {part_size}")
    all_index = list(range(total_count))

    for anchor_index in range(5):
        start_index = anchor_index * part_size
        end_index = start_index + part_size - 1
        print(f"part {anchor_index}: {end_index - start_index + 1}; start_index: {start_index}, end_index: {end_index}")

        if anchor_index not in anchor_index_to_results:
            anchor_index_to_results[anchor_index] = {
                "same_gender_count_to_count": defaultdict(int),
                "same_gender_count_to_hit_count": defaultdict(int),
            }

        cur_index = 0
        with open(f"outputs/contextual/{attribute_type}/consultant_samples.jsonl", "r") as f:
            for line in f:
                if cur_index < start_index or cur_index > end_index:
                    cur_index += 1
                    continue
                cur_index += 1
                item = json.loads(line)
                if "genders" in item:
                    attributes = item["genders"]
                else:
                    attributes = item["attributes"]

                hit_candidate_id = item["hit_candidate_id"]

                same_gender_count = attributes.count(attributes[anchor_index]) - 1
                anchor_index_to_results[anchor_index]["same_gender_count_to_count"][same_gender_count] += 1
                anchor_index_to_results[anchor_index]["same_gender_count_to_hit_count"][same_gender_count] += 1 if anchor_index == hit_candidate_id else 0

    same_gender_count_to_count = defaultdict(int)
    same_gender_count_to_hit_count = defaultdict(int)

    for anchor_index, results in anchor_index_to_results.items():
        for same_gender_count, count in results["same_gender_count_to_count"].items():
            same_gender_count_to_count[same_gender_count] += count
        for same_gender_count, hit_count in results["same_gender_count_to_hit_count"].items():
            same_gender_count_to_hit_count[same_gender_count] += hit_count

    same_gender_count_to_count = dict(sorted(same_gender_count_to_count.items(), key=lambda x: x[0]))
    for same_gender_count, count in same_gender_count_to_count.items():
        hit_count = same_gender_count_to_hit_count[same_gender_count]
        print(f"same_gender_count: {same_gender_count}, count: {count}, hit_rate: {(hit_count / count):.6f}")
