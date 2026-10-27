"""Provider wrappers — the ONLY file that touches real APIs.

Keys come from environment variables, never from code:

    export GOOGLE_API_KEY=...     (Windows: $env:GOOGLE_API_KEY="...")
    export GROQ_API_KEY=...
    export DEEPINFRA_API_KEY=...  (code study C3 only)

Install SDKs:  pip install -r requirements.txt

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
    Groq shut this model down on 2026-08-16. The code study's C3 serves the
    same Llama 3.1 8B Instruct weights from DeepInfra instead
    (LLAMA_CODE_MODEL, code_experimental_models); the math study's
    definitions below are left as they were.
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

def find_daily_marker(err: Exception) -> Optional[str]:
    """Return the matched daily marker string if err is a per-day limit, else None."""
    msg = str(err).lower()
    daily_markers = ("per day", "requests per day", "rpd", "tpd",
                     "per-day", "daily limit", "daily quota",
                     "perday", "per_day", "requests_per_day", "daily")
    for m in daily_markers:
        if m in msg:
            return m
    return None

def _looks_daily(err: Exception) -> bool:
    """True only for per-DAY limits. Ambiguous words like 'quota' or
    'exceeded' also appear in per-MINUTE errors, so we match only phrases
    that unambiguously mean daily."""
    return find_daily_marker(err) is not None

# --- Google AI Studio (Gemini) ----------------------------------------------

try:
    from google import genai
    from google.genai import errors as genai_errors
    _gclient = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY")) if os.environ.get("GOOGLE_API_KEY") else None
    _GENAI_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as _e:
    genai = None
    genai_errors = None
    _gclient = None
    _GENAI_IMPORT_ERROR = _e


def call_gemini(prompt: str) -> str:
    if _GENAI_IMPORT_ERROR is not None:
        raise RuntimeError(f"google-genai SDK is not installed ({_GENAI_IMPORT_ERROR}); "
                           f"run: pip install -r requirements.txt")
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
            marker = find_daily_marker(e)
            if marker:
                raise DailyLimitExhausted(f"Daily limit marker matched: '{marker}'. Raw error: {e}",
                                         raw_error=str(e), matched_marker=marker) from e
            raise RetryableRateLimit(str(e)) from e
        if code in (500, 502, 503, 504) or "503" in str(e) or "UNAVAILABLE" in str(e):
            raise RetryableRateLimit(str(e)) from e   # transient server error: back off and retry
        raise


# --- Groq (OpenAI-compatible; serves both open models) ----------------------

try:
    from groq import Groq, APIStatusError
    from groq import RateLimitError as GroqRateLimitError
    _groq = Groq(api_key=os.environ.get("GROQ_API_KEY")) if os.environ.get("GROQ_API_KEY") else None
    _GROQ_IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as _e:
    Groq = None
    APIStatusError = Exception
    GroqRateLimitError = Exception
    _groq = None
    _GROQ_IMPORT_ERROR = _e


def _call_groq(model_id: str, prompt: str, max_tokens: int = 4096) -> str:
    if _GROQ_IMPORT_ERROR is not None:
        raise RuntimeError(f"groq SDK is not installed ({_GROQ_IMPORT_ERROR}); "
                           f"run: pip install -r requirements.txt")
    if _groq is None:
        raise RuntimeError("GROQ_API_KEY environment variable is not set. Please set it to run the Groq models.")
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
        marker = find_daily_marker(e)
        if marker:
            raise DailyLimitExhausted(f"Daily limit marker matched: '{marker}'. Raw error: {e}",
                                     raw_error=str(e), matched_marker=marker) from e
        raise RetryableRateLimit(str(e)) from e
    except APIStatusError as e:
        if e.status_code == 413:
            raise RetryableRateLimit(str(e)) from e
        raise
def call_llama(prompt: str) -> str:
    return _call_groq(LLAMA_MODEL, prompt, max_tokens=2048)

def call_gptoss(prompt: str) -> str:
    return _call_groq(GPTOSS_MODEL, prompt, max_tokens=6144)
# --- DeepInfra (OpenAI-compatible; code study C3 only) -----------------------

LLAMA_CODE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"


def call_llama_deepinfra(prompt: str) -> str:
    import httpx
    key = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        raise RuntimeError("DEEPINFRA_API_KEY environment variable is not set. Please set it to run the code-study C3 reviewer.")
    try:
        resp = httpx.post(f"{DEEPINFRA_BASE_URL}/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": LLAMA_CODE_MODEL,
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": TEMPERATURE, "max_tokens": 2048},
                          timeout=120.0)
    except httpx.TransportError as e:
        raise RetryableRateLimit(str(e)) from e   # network blip: back off and retry
    if resp.status_code == 429:
        err = RuntimeError(resp.text)
        marker = find_daily_marker(err)
        if marker:
            raise DailyLimitExhausted(f"Daily limit marker matched: '{marker}'. Raw error: {resp.text}",
                                      raw_error=resp.text, matched_marker=marker)
        raise RetryableRateLimit(resp.text)
    if resp.status_code in (500, 502, 503, 504):
        raise RetryableRateLimit(resp.text)
    resp.raise_for_status()
    data = resp.json()
    global _LAST_USAGE
    usage = data.get("usage") or {}
    if usage:
        _LAST_USAGE = {"prompt_tokens": usage.get("prompt_tokens", 0),
                       "completion_tokens": usage.get("completion_tokens", 0)}
    return data["choices"][0]["message"].get("content") or ""


# --- The three experimental models ------------------------------------------

def experimental_models() -> list[Model]:
    return [
        Model(GEMINI_MODEL, call_gemini),
        Model(LLAMA_MODEL, call_llama),
        Model(GPTOSS_MODEL, call_gptoss),
    ]


def code_experimental_models() -> list[Model]:
    """Models for the code study: as experimental_models(), but C3's Llama 3.1
    8B Instruct is served by DeepInfra because Groq retired llama-3.1-8b-instant."""
    return [
        Model(GEMINI_MODEL, call_gemini),
        Model(LLAMA_CODE_MODEL, call_llama_deepinfra),
        Model(GPTOSS_MODEL, call_gptoss),
    ]


def preflight_code_providers() -> list[str]:
    """Zero-cost startup check for the code study: SDKs importable, keys set,
    and each reviewer model listed by its host. Returns a list of problems
    (empty when ready). Listing models makes no completion calls."""
    import httpx
    problems = []
    if _GENAI_IMPORT_ERROR is not None:
        problems.append(f"google-genai SDK not installed: {_GENAI_IMPORT_ERROR}")
    if _GROQ_IMPORT_ERROR is not None:
        problems.append(f"groq SDK not installed: {_GROQ_IMPORT_ERROR}")
    for var in ("GOOGLE_API_KEY", "GROQ_API_KEY", "DEEPINFRA_API_KEY"):
        if not os.environ.get(var):
            problems.append(f"{var} is not set")
    hosts = [("Groq", "https://api.groq.com/openai/v1/models", "GROQ_API_KEY", GPTOSS_MODEL),
             ("DeepInfra", f"{DEEPINFRA_BASE_URL}/models", "DEEPINFRA_API_KEY", LLAMA_CODE_MODEL)]
    for host, url, var, model_id in hosts:
        if not os.environ.get(var):
            continue
        try:
            r = httpx.get(url, headers={"Authorization": f"Bearer {os.environ[var]}"}, timeout=30.0)
            r.raise_for_status()
            ids = {m.get("id") for m in r.json().get("data", [])}
            if model_id not in ids:
                problems.append(f"{host} does not list model {model_id}")
        except Exception as e:
            problems.append(f"{host} model list unavailable: {e}")
    return problems
