import json


gender = "female"
all_gender_order_ids_list = [
    [0],
    [12,13,14,15],
    [6,7,8,9,10,11],
    [2,3,4,5],
    [1],
]

for gender_order_ids in all_gender_order_ids_list:
    hit_count = 0
    total_count = 0
    for gender_order_id in gender_order_ids:
        with open(f"outputs/{gender}/consultant_samples_genders_{gender_order_id}.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                total_count += 1
                if item["hit"]:
                    hit_count += 1

    print(f"gender_order_ids: {gender_order_ids}; total_count: {total_count}")
    print(f"hit_rate: {hit_count / total_count}")
