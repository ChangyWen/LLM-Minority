import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents import chat
import re
import random


def extract_from_tags(text, tag):
    if text is None:
        return None
    pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return None


def complete(prompt, model_name="msra-gpt-5", reasoning_effort_or_thinking_budget="high"):
    response = chat(
        max_retry=1,
        prompt=prompt,
        model_name=model_name,
        enable_search=False,
        enable_thinking=True,
        reasoning_effort_or_thinking_budget=reasoning_effort_or_thinking_budget,
        temperature=1.0,
        top_p=1.0,
    )
    if response is None:
        return None
    return response["value"]


prompt_template = """
Below is an anonymous resume (enclosed within the <resume> </resume> tags).
<resume>
{resume}
</resume>

**Your Task:**
1. Read the resume carefully.
2. Check if the resume satisfies the following criteria:
- It does not contain the candidate's name (first name, last name, or any identifiable personal name).
- It does not contain the candidate's gender (e.g., "male", "female", or pronouns)
- It does not contain the candidate's race (e.g., racial or ethnic identifiers such as "Black", "White", "Asian", "Hispanic", etc.).
- It does not explicitly mention the age or the birth date of the candidate. However, references to work timelines and experience periods (e.g., "2019–2022", "10+ years of experience") are acceptable.
3. If the resume satisfies all the three criteria above, output "<anonymous>True</anonymous>" at the end of your response. Otherwise, output "<anonymous>False</anonymous>" with brief justification.
""".strip()


if __name__ == "__main__":
    total_count = int(sys.argv[1])
    index = int(sys.argv[2])
    random.seed(total_count)

    save_file = f"dataset/resumes_shuffled_processed_anonymous.jsonl"
    idx_done = []
    if not os.path.exists(save_file):
        with open(save_file, "w") as f:
            pass
    else:
        with open(save_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                idx = item["idx"]
                idx_done.append(idx)
    all_idx = set()
    with open("dataset/resumes_shuffled_processed.jsonl", "r") as f:
        for line in f:
            item = json.loads(line.strip())
            idx = item["idx"]
            all_idx.add(idx)
    remaining_idx = all_idx - set(idx_done)
    remaining_idx = sorted(list(remaining_idx))
    remaining_idx = [remaining_idx[i::total_count] for i in range(total_count)]
    remaining_idx = remaining_idx[index]
    print(f"chunk {index} size: {len(remaining_idx)}")
    print(f"total size: {len(remaining_idx)}")

    with open("dataset/resumes_shuffled_processed.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            idx = item["idx"]
            if idx not in remaining_idx:
                continue
            try:
                resume = item["resume"].strip()
                prompt = prompt_template.format(resume=resume)
                response = complete(prompt)
                anonymous = extract_from_tags(response, "anonymous")
                if anonymous is None:
                    continue
                if anonymous.lower() == "true":
                    with open(save_file, "a") as f:
                        f.write(json.dumps(item) + "\n")
            except Exception as e:
                continue