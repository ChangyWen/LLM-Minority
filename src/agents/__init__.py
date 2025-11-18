import time
from .gemini import Gemini
from .gpt import GPT
from .grok import Grok
from .qwen import Qwen
from .msra import MSRA
import threading
import functools


gemini = Gemini()
gpt = GPT()
grok = Grok()
qwen = Qwen()
msra = MSRA()


def timeout_decorator(func):
    def wrapper(*args, **kwargs):

        result = [None]
        exception = [None]

        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        timeout_in_seconds = 60
        thread.join(timeout=timeout_in_seconds)

        if thread.is_alive():
            print(f"[ERROR] Function {func.__name__} timed out after {timeout_in_seconds} seconds")
            return None

        if exception[0]:
            raise exception[0]

        return result[0]

    return functools.wraps(func)(wrapper)


@timeout_decorator
def chat(
    max_retry,
    prompt,
    model_name,
    enable_search,
    enable_thinking,
    reasoning_effort_or_thinking_budget,
    temperature=1.0,
    top_p=1.0,
):
    if prompt is None:
        return None

    response = None
    for _ in range(max_retry):

        chat_func = None
        if model_name.lower().startswith("gemini"):
            chat_func = gemini.chat
        elif model_name.lower().startswith("msra"):
            chat_func = msra.chat
        elif model_name.lower().startswith("grok"):
            chat_func = grok.chat
        elif model_name.lower().startswith("qwen"):
            chat_func = qwen.chat
        elif model_name.lower().startswith("gpt") or model_name.lower().startswith("o4"):
            chat_func = gpt.chat
        else:
            raise ValueError(f"Model name {model_name} not supported")

        response = chat_func(prompt, model_name, enable_search, enable_thinking, reasoning_effort_or_thinking_budget, temperature, top_p)

        if response is None:
            time.sleep(5)
            print(f"retry {_ + 1} / {max_retry}")
            continue
        if "value" in response:
            if response["value"] is None:
                time.sleep(5)
                print(f"retry {_ + 1} / {max_retry}")
                continue
        return response
    return response
