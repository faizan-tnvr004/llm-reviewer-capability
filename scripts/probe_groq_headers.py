import os
import json
import urllib.request

if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

key = os.environ.get("GROQ_API_KEY")
req = urllib.request.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    data=json.dumps({
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1
    }).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "MultiagentErrorDiversity/1.0",
    },
    method="POST"
)
with urllib.request.urlopen(req) as resp:
    print("GROQ RATELIMIT HEADERS:")
    for h, v in resp.headers.items():
        if "ratelimit" in h.lower():
            print(f"  {h}: {v}")
