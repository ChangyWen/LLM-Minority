import os
import anthropic
import random
import requests
import time
import json
import sys
from openai import AzureOpenAI, OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.projects import AIProjectClient
from azure.ai.agents.models import MessageRole
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, HttpOptions, ThinkingConfig
from dotenv import load_dotenv
load_dotenv()


class Qwen:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        print("qwen initialized")

    def chat(self, prompt, model_name, enable_search, enable_thinking, thinking_budget, temperature=1.0, top_p=1.0):
        try:
            # generate the response
            messages = [ {"role": "user", "content": prompt} ]
            stream = self.client.chat.completions.create(
                model=model_name,
                extra_body={
                    "enable_thinking": enable_thinking,
                    "thinking_budget": thinking_budget if enable_thinking else 0,
                },
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            )
            response_text = ""
            thinking_draft = ""
            for chunk in stream:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content is not None:
                    response_text += delta.content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    thinking_draft += delta.reasoning_content
            results = { "value": response_text }
            if enable_thinking:
                results["thinking_draft"] = thinking_draft
            return results
        except Exception as e:
            print(f"Error in qwen: {e}")
            return None


# if __name__ == "__main__":
#     qwen = Qwen()
#     print(qwen.chat("How many people live in the US?", "qwen3-235b-a22b", False, True, 2000))
#     print(qwen.chat("How many people live in the US?", "qwen3-235b-a22b", False, False, 2000))