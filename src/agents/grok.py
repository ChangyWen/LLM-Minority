import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


class Grok:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ['XAI_API_KEY'], base_url="https://api.x.ai/v1")
        print("grok initialized")

    def chat(self, prompt, model_name, enable_search, enable_thinking, reasoning_effort, temperature=1.0, top_p=1.0):
        try:
            assert model_name in ["grok-3", "grok-3-mini", "grok-4"]
            if model_name == "grok-3-latest": assert enable_thinking == False
            if model_name == "grok-3-mini": assert enable_thinking == True
            # prepare the messages
            messages = [ {"role": "user", "content": prompt} ]
            # generate the response
            if not enable_search:
                if enable_thinking:
                    completion = self.client.chat.completions.create(
                        model=model_name,
                        reasoning_effort=reasoning_effort,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return {
                        "value": completion.choices[0].message.content,
                        "thinking_draft": completion.choices[0].message.reasoning_content
                    }
                else:
                    completion = self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return { "value": completion.choices[0].message.content }
            else:
                max_search_results = 20 # sufficient for most cases (default settings of grok)
                url = "https://api.x.ai/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"
                }
                payload = {
                    "messages": messages,
                    "search_parameters": {
                        "mode": "on", # force search to be enabled
                        "max_search_results": max_search_results,
                        "return_citations": True
                    },
                    "model": model_name,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                if enable_thinking:
                    payload["reasoning_effort"] = reasoning_effort
                res = requests.post(url, headers=headers, json=payload).json()
                response = res["choices"][0]
                results = { "value": response["message"]["content"] }
                if enable_thinking:
                    results["thinking_draft"] = response["message"]["reasoning_content"]
                if enable_search:
                    results["sources"] = {
                        "num_sources_used": res["usage"]["num_sources_used"],
                        "citations": res["citations"]
                    }
                return results
        except Exception as e:
            print(f"Error in grok: {e}")
            return None


# if __name__ == "__main__":
#     grok = Grok()
#     print(json.dumps(grok.chat("How many people live in the US?", "grok-3-latest", False, False, "low"), indent=4))
#     print(json.dumps(grok.chat("How many people live in the US?", "grok-3-latest", True, False, "low"), indent=4))
#     print(json.dumps(grok.chat("How many people live in the US?", "grok-3-mini", False, True, "low"), indent=4))
#     print(json.dumps(grok.chat("How many people live in the US?", "grok-3-mini", True, True, "low"), indent=4))