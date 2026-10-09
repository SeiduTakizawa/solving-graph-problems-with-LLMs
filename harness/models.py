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
NEAR_LIMIT = 0.9  # a prompt (or prompt + reply) above this share of the context gets flagged in the trace
# Output cap per model call (Ollama's num_predict). Normal replies are short (2026-10-09, 914 qwen3.5 calls: median
# 290 tokens, p99 5.5k); the 5 over 8k were all thinking spirals (11.6k-14.9k tokens, up to 162 s) that ran into the
# 16k window and came back empty. With the cap a spiral stops at 8k, inside the window, and gets its own nudge.
OLLAMA_NUM_PREDICT = 8_192


def context_window(model: str) -> int | None:
    """The context we ask for: set for Ollama models, left to the provider for API models."""
    return OLLAMA_NUM_CTX if model.startswith(("ollama/", "ollama_chat/")) else None


def call_model(messages: list[dict], tools: list[dict], model: str = DEFAULT_MODEL, think: bool | None = None) -> dict:
    """Send the conversation and the available tools to the model, return its reply as a plain dict.

    Token counts, latency and retries go under "extra"; the loop removes it before the reply joins the
    history, so the model never sees it. latency_s includes the waits between retries: that time was spent.
    think: turn a thinking model's reasoning on or off (Ollama only); None leaves the model's default.
    """
    start = time.time()
    errors = []
    num_ctx = context_window(model)
    options = {"num_ctx": num_ctx, "max_tokens": OLLAMA_NUM_PREDICT} if num_ctx else {}
    if think is not None and num_ctx:
        options["think"] = think
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
    if num_ctx and response.usage.prompt_tokens + response.usage.completion_tokens > NEAR_LIMIT * num_ctx:
        reply["extra"]["near_context_limit"] = num_ctx  # the reply may have been cut short by the window
    if response.choices[0].finish_reason == "length":  # stopped by the output cap (or the window), not by the model
        reply["extra"]["output_cut"] = True
    if errors:  # only present when something went wrong, so normal traces stay unchanged
        reply["extra"]["retries"] = len(errors)
        reply["extra"]["retry_errors"] = errors
    return reply
