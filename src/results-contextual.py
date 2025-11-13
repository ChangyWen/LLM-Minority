import json
import sys


if __name__ == "__main__":
    # {"genders": ["Female", "Male", "Male", "Female", "Female"], "candidate_order": [1, 2, 3, 4, 0], "suggested_candidate_id": 2, "hit_candidate_id": 3}

    same_gender_count_to_count = {}
    same_gender_count_to_hit_count = {}

    total_count = 0
    with open("outputs/contextual/gender/consultant_samples.jsonl", "r") as f:
        for line in f:
            total_count += 1
            item = json.loads(line)
            genders = item["genders"]
            candidate_order = item["candidate_order"]
            suggested_candidate_id = item["suggested_candidate_id"]
            hit_candidate_id = item["hit_candidate_id"]

            for candidate_id in [0, 1, 2, 3, 4]:
                same_gender_count = genders.count(genders[candidate_id]) - 1
                if same_gender_count not in same_gender_count_to_count:
                    same_gender_count_to_count[same_gender_count] = 0
                same_gender_count_to_count[same_gender_count] += 1
                if candidate_id == hit_candidate_id:
                    if same_gender_count not in same_gender_count_to_hit_count:
                        same_gender_count_to_hit_count[same_gender_count] = 0
                    same_gender_count_to_hit_count[same_gender_count] += 1

    same_gender_count_to_count = dict(sorted(same_gender_count_to_count.items(), key=lambda x: x[0]))
    for same_gender_count, count in same_gender_count_to_count.items():
        hit_count = 0
        if same_gender_count in same_gender_count_to_hit_count:
            hit_count = same_gender_count_to_hit_count[same_gender_count]
        print(f"same_gender_count: {same_gender_count}, count: {count}, hit_rate: {(hit_count / count):.2f}")
