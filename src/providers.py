"""Provider wrappers — the ONLY file that touches real APIs.

Keys come from environment variables, never from code:

    export GOOGLE_API_KEY=...     (Windows: $env:GOOGLE_API_KEY="...")
    export GROQ_API_KEY=...

Install SDKs:  pip install google-genai groq

Each call function:
  - returns the model's response text on success
  - raises RetryableRateLimit on a per-minute 429 (RPM/TPM) -> harness backs off
  - raises DailyLimitExhausted on a per-day 429 (RPD/TPD) -> harness stops,
    resume after the daily reset

Both providers return HTTP 429 for rate limits; the distinguishing detail is
in the error message/retry-after. Heuristic: if the message mentions a daily
limit, or retry-after is very long (> ~10 min), treat it as daily.

The three experimental models (recorded here per the analysis plan):
  - Google:      gemini-3.1-flash-lite   (500 RPD free tier — the usable one)
  - Meta:        llama-3.1-8b-instant    (Groq, 1K RPD)
  - OpenAI-OSS:  openai/gpt-oss-20b       (Groq, 1K RPD)
Three different families, comparable small size.
"""

from __future__ import annotations

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from harness import Model, RetryableRateLimit, DailyLimitExhausted

TEMPERATURE = 0.7  # nonzero so repeated samples are independent (see plan §3)

# Exact model IDs — confirm against the live console and edit here if they drift.
GEMINI_MODEL = "gemini-3.1-flash-lite"
LLAMA_MODEL = "llama-3.1-8b-instant"
GPTOSS_MODEL = "openai/gpt-oss-20b"

_LAST_USAGE = None

def get_last_usage():
    """Return {'prompt_tokens': int, 'completion_tokens': int} for the most
    recent call, or None. Consumed by pipeline.TokenMeter."""
    global _LAST_USAGE
    u, _LAST_USAGE = _LAST_USAGE, None
    return u

def _looks_daily(err: Exception) -> bool:
    """True only for per-DAY limits. Ambiguous words like 'quota' or
    'exceeded' also appear in per-MINUTE errors, so we match only phrases
    that unambiguously mean daily."""
    msg = str(err).lower()
    daily_markers = ("per day", "requests per day", "rpd", "tpd",
                     "per-day", "daily limit", "daily quota")
    return any(m in msg for m in daily_markers)

# --- Google AI Studio (Gemini) ----------------------------------------------

from google import genai
from google.genai import errors as genai_errors

_gclient = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY")) if os.environ.get("GOOGLE_API_KEY") else None


def call_gemini(prompt: str) -> str:
    if _gclient is None:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set. Please set it to run the Gemini model.")
    try:
        resp = _gclient.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": TEMPERATURE, "max_output_tokens": 8192},
        )
        global _LAST_USAGE
        um = getattr(resp, "usage_metadata", None)
        if um:
            _LAST_USAGE = {"prompt_tokens": um.prompt_token_count,
                           "completion_tokens": um.candidates_token_count}
        return resp.text or ""
    except genai_errors.APIError as e:
        code = getattr(e, "code", None)
        if code == 429 or "429" in str(e):
            raise (DailyLimitExhausted if _looks_daily(e) else RetryableRateLimit)() from e
        if code in (500, 502, 503, 504) or "503" in str(e) or "UNAVAILABLE" in str(e):
            raise RetryableRateLimit() from e   # transient server error: back off and retry
        raise


# --- Groq (OpenAI-compatible; serves both open models) ----------------------

from groq import Groq, APIStatusError
from groq import RateLimitError as GroqRateLimitError

_groq = Groq(api_key=os.environ.get("GROQ_API_KEY")) if os.environ.get("GROQ_API_KEY") else None


def _call_groq(model_id: str, prompt: str, max_tokens: int = 4096) -> str:
    try:
        resp = _groq.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
        )
        global _LAST_USAGE
        if getattr(resp, "usage", None):
            _LAST_USAGE = {"prompt_tokens": resp.usage.prompt_tokens,
                           "completion_tokens": resp.usage.completion_tokens}
        return resp.choices[0].message.content or ""
    except GroqRateLimitError as e:
        raise (DailyLimitExhausted if _looks_daily(e) else RetryableRateLimit)() from e
    except APIStatusError as e:
        if e.status_code == 413:
            raise RetryableRateLimit() from e
        raise
def call_llama(prompt: str) -> str:
    return _call_groq(LLAMA_MODEL, prompt, max_tokens=2048)

def call_gptoss(prompt: str) -> str:
    return _call_groq(GPTOSS_MODEL, prompt, max_tokens=6144)
# --- The three experimental models ------------------------------------------

def experimental_models() -> list[Model]:
    return [
        Model(GEMINI_MODEL, call_gemini),
        Model(LLAMA_MODEL, call_llama),
        Model(GPTOSS_MODEL, call_gptoss),
    ]
