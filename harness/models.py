"""Model layer: every LLM call goes through LiteLLM, so switching model is changing one string."""
import time

import litellm

DEFAULT_MODEL = "ollama_chat/qwen3:8b"

# Errors that say "try again later", not "your request is wrong": a crashed or overloaded server (e.g. Ollama's
# "CUDA error: the launch timed out"), a dropped connection, a timeout, a rate limit. Anything else (a wrong model
# name, a bad request, a missing API key) would fail the same way again, so it is raised at once.
TRANSIENT_ERRORS = (litellm.InternalServerError, litellm.ServiceUnavailableError, litellm.APIConnectionError,
                    litellm.Timeout, litellm.RateLimitError)
RETRY_WAITS_S = (2, 8, 30)  # seconds to wait before each retry; after the last one the error is raised

# Ollama runs every model with a 4,096-token context unless asked otherwise, whatever the model supports
# (qwen3.5:9b: 262k). Past it, the prompt is cut and the reply stops short, silently: qwen3.5's "empty replies" in
# the combine runs all came at ~3,850 prompt tokens. 16k fits qwen3.5:9b and qwen3:8b on the 12 GB GPU.
OLLAMA_NUM_CTX = 16_384
NEAR_LIMIT = 0.9  # a prompt above this share of the context gets flagged in the trace


def context_window(model: str) -> int | None:
    """The context we ask for: set for Ollama models, left to the provider for API models."""
    return OLLAMA_NUM_CTX if model.startswith(("ollama/", "ollama_chat/")) else None


def call_model(messages: list[dict], tools: list[dict], model: str = DEFAULT_MODEL) -> dict:
    """Send the conversation and the available tools to the model, return its reply as a plain dict.

    Token counts, latency and retries go under "extra"; the loop removes it before the reply joins the
    history, so the model never sees it. latency_s includes the waits between retries: that time was spent.
    """
    start = time.time()
    errors = []
    num_ctx = context_window(model)
    options = {"num_ctx": num_ctx} if num_ctx else {}
    for wait in (*RETRY_WAITS_S, None):
        try:
            response = litellm.completion(model=model, messages=messages, tools=tools, **options)
            break
        except TRANSIENT_ERRORS as e:
            if wait is None:
                raise
            errors.append(f"{type(e).__name__}: {e}"[:300])
            time.sleep(wait)
    reply = response.choices[0].message.model_dump()
    reply["extra"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "latency_s": round(time.time() - start, 3),
    }
    if num_ctx and response.usage.prompt_tokens > NEAR_LIMIT * num_ctx:  # the reply may have been cut short
        reply["extra"]["near_context_limit"] = num_ctx
    if errors:  # only present when something went wrong, so normal traces stay unchanged
        reply["extra"]["retries"] = len(errors)
        reply["extra"]["retry_errors"] = errors
    return reply
