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


prompt_template = """
Below is an anonymous resume (enclosed within the <resume> </resume> tags).
<resume>
{resume}
</resume>

**Your Task:**
1. Read the resume carefully.
2. Identify the job title(s) listed at the very beginning of the resume. These may include one or multiple professional titles (e.g., "HR DIRECTOR", "FINANCE OFFICE ASSOCIATE", "LICENSED UNITED HEALTHCARE INSURANCE SALES AGENT"), typically presented in all capital letters.
3. Extract these job titles verbatim, without modifying capitalization, punctuation, or spacing.
4. Extract only those job titles that are listed at the very beginning of the resume. Ignore any job titles appearing later in the resume.
5. If one or more job titles are found, output them inside <job-titles> and </job-titles> tags at the end of your response.
6. If no such job titles appear at the beginning of the resume, respond with: No job titles found.
""".strip()


if __name__ == "__main__":
    total_count = int(sys.argv[1])
    index = int(sys.argv[2])
    random.seed(total_count)

    save_file = f"dataset/resumes_shuffled_processed.jsonl"
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

    with open("dataset/resumes_shuffled.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            idx = item["idx"]
            if idx not in remaining_idx:
                continue
            resume = item["resume"].strip()
            original_resume = resume
            prompt = prompt_template.format(resume=resume)
            response = complete(prompt)
            job_title = extract_from_tags(response, "job-titles")
            job_title_idx = resume.find(job_title)
            if job_title_idx != 0:
                print(f"Error in finding job title in resume {idx}")
                print("********************")
                print(resume)
                print("--------------------------------")
                print(response)
                print("********************")
                continue
            resume = resume.replace(job_title, "", 1)
            resume = resume.strip()
            if (job_title not in original_resume) or (resume not in original_resume):
                print(f"Error in finding job title in resume {idx}")
                print("********************")
                print(original_resume)
                print("--------------------------------")
                print(job_title)
                print("--------------------------------")
                print(resume)
                print("********************")
                continue
            item["resume"] = resume
            item["job_title"] = job_title
            with open(save_file, "a") as f:
                f.write(json.dumps(item) + "\n")