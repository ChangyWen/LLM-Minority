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


prompt = """
Below is an anonymous resume (enclosed within the <resume> </resume> tags).
<resume>
{resume}
</resume>
Please help me paraphrase the resume.
1. Preserve the original content. Do not add or remove any important information.
2. Maintain a similar length.
Put the paraphrased resume within the <paraphrased-resume> </paraphrased-resume> tags at the end of the response.
""".strip()


if __name__ == "__main__":
    all_paraphrased_resumes = []
    with open("dataset/consultant_samples.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            resume = item["resume"]
            while True:
                prompt = prompt.format(resume=resume)
                response = complete(prompt)
                paraphrased_resume = extract_from_tags(response, "paraphrased-resume")
                if paraphrased_resume is None:
                    print(f"Error in paraphrasing resume {item['idx']}")
                    continue
                print(f"{paraphrased_resume}\n\n")
                all_paraphrased_resumes.append(paraphrased_resume)
                if len(all_paraphrased_resumes) >= 5:
                    item["paraphrased_resumes"] = all_paraphrased_resumes
                    with open("dataset/consultant_samples_paraphrased.jsonl", "a") as f:
                        f.write(json.dumps(item) + "\n")
                    exit()