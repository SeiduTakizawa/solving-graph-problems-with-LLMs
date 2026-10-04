"""Model layer: every LLM call goes through LiteLLM, so switching model is changing one string."""
import time

import litellm

DEFAULT_MODEL = "ollama_chat/qwen3:8b"


def call_model(messages: list[dict], tools: list[dict], model: str = DEFAULT_MODEL) -> dict:
    """Send the conversation and the available tools to the model, return its reply as a plain dict.

    Token counts and latency go under "extra"; the loop removes it before the reply
    joins the history, so the model never sees it.
    """
    start = time.time()
    response = litellm.completion(model=model, messages=messages, tools=tools)
    reply = response.choices[0].message.model_dump()
    reply["extra"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "latency_s": round(time.time() - start, 3),
    }
    return reply
