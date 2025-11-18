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


def complete(prompt, model_name="msra-gpt-5", reasoning_effort_or_thinking_budget="low"):
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


prompt = """
Below is an anonymous resume (enclosed within the <resume> </resume> tags).
<resume>
{resume}
</resume>

**Your Task:**
Paraphrase the resume while following all requirements below:
1. Preserve all original information. Do not add new content or remove any important details.
2. Keep the length comparable to the original (do not significantly shorten or lengthen it).
3. Maintain the original style and tone (e.g., professional, concise, descriptive).
At last, put the paraphrased resume within the <paraphrased-resume> </paraphrased-resume> tags at the end of the your response.
""".strip()


if __name__ == "__main__":
    total_count = int(sys.argv[1])
    index = int(sys.argv[2])
    random.seed(total_count)

    save_file = f"dataset/resumes_paraphrases.jsonl"
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
    with open("dataset/resumes_shuffled.jsonl", "r") as f:
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
    print(remaining_idx)

    with open("dataset/resumes_shuffled.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            idx = item["idx"]
            if idx not in remaining_idx:
                continue
            resume = item["resume"]
            print(f"paraphrasing resume {idx}:\n{resume}")
            paraphrases = []
            paraphrases_set = set()
            while len(paraphrases_set) < 10:
                prompt = prompt.format(resume=resume)
                response = complete(prompt)
                paraphrased_resume = extract_from_tags(response, "paraphrased-resume")
                if paraphrased_resume is None:
                    print(f"Error in paraphrasing resume {idx}")
                    continue
                if paraphrased_resume in paraphrases_set:
                    print(f"Error in paraphrasing resume {idx}: paraphrased_resume is already in the list")
                    continue
                # check length is within len(resume) +/- 10% of len(resume)
                if len(paraphrased_resume) < len(resume) * 0.92 or len(paraphrased_resume) > len(resume) * 1.08:
                    print(f"Error in paraphrasing resume {idx}: length is not within 10% of the original length")
                    continue
                paraphrases_set.add(paraphrased_resume)
                print(f"************\n{paraphrased_resume}\n\n")
                paraphrases.append(paraphrased_resume)
            item["paraphrased_resumes"] = paraphrases
            with open(save_file, "a") as f:
                f.write(json.dumps(item) + "\n")