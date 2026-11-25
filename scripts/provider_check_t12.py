"""T12: Provider check for openai/gpt-oss-20b on long code statements.

Per spec:
- Send 3 C2 review prompts built from the 3 longest candidate statements.
- Record prompt and completion tokens and whether each response contained a VERDICT line.
- If any response is empty, report it with the token counts.
"""

from __future__ import annotations

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tasks.livecodebench_tasks import load_lcb_tasks
from prompts_code import CODE_REVIEW_PROMPT
from providers import call_gptoss, get_last_usage

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "run_code_data", "raw")


def main():
    print("Loading candidate problems...")
    tasks = load_lcb_tasks(n=None, difficulties=("medium", "hard"), raw_dir=RAW_DIR)
    print(f"Total candidate problems: {len(tasks)}")

    # Sort by length of statement (longest first)
    tasks_by_len = sorted(tasks, key=lambda t: len(t.meta.get("statement", "")), reverse=True)
    top3 = tasks_by_len[:3]

    print("\nTop 3 longest candidate problems:")
    for i, t in enumerate(top3, 1):
        stmt_len = len(t.meta.get("statement", ""))
        print(f"  {i}. {t.task_id} ({t.meta.get('platform')}, {t.meta.get('difficulty')}): {stmt_len} chars")

    # Sample solution for review
    sample_code = (
        "import sys\n\n"
        "def solve():\n"
        "    input_data = sys.stdin.read().split()\n"
        "    if not input_data:\n"
        "        return\n"
        "    n = int(input_data[0])\n"
        "    ans = sum(int(x) for x in input_data[1:n+1])\n"
        "    print(ans)\n\n"
        "if __name__ == '__main__':\n"
        "    solve()\n"
    )

    results = []

    for idx, t in enumerate(top3, 1):
        statement = t.meta.get("statement", "")
        prompt = CODE_REVIEW_PROMPT.format(problem=statement, solution=sample_code)

        print(f"\n--- Testing problem {idx}/3: {t.task_id} (Prompt chars: {len(prompt)}) ---")
        t0 = time.time()
        try:
            resp = call_gptoss(prompt)
            elapsed = time.time() - t0
            usage = get_last_usage() or {}
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)

            has_verdict = "VERDICT:" in resp
            is_empty = len(resp.strip()) == 0

            print(f"Time: {elapsed:.1f}s | Prompt tokens: {p_tok} | Completion tokens: {c_tok}")
            print(f"Has VERDICT line: {has_verdict} | Is empty: {is_empty}")
            print(f"Response preview:\n{resp[:300]}...")

            results.append({
                "task_id": t.task_id,
                "statement_chars": len(statement),
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "has_verdict": has_verdict,
                "is_empty": is_empty,
                "response": resp,
            })
        except Exception as e:
            print(f"Error calling gpt-oss for {t.task_id}: {e}")
            results.append({
                "task_id": t.task_id,
                "statement_chars": len(statement),
                "error": str(e),
            })

        if idx < 3:
            print("Waiting 10s between calls for Groq rate limits...")
            time.sleep(10)

    print("\n" + "=" * 70)
    print("T12 PROVIDER CHECK SUMMARY")
    print("=" * 70)
    for r in results:
        if "error" in r:
            print(f"Task: {r['task_id']} | ERROR: {r['error']}")
        else:
            v_status = "PASS (has VERDICT)" if r["has_verdict"] else "FAIL (no VERDICT)"
            empty_status = "EMPTY" if r["is_empty"] else "NON-EMPTY"
            print(f"Task: {r['task_id']} ({r['statement_chars']} chars) | Prompt Tok: {r['prompt_tokens']} | Compl Tok: {r['completion_tokens']} | {v_status} | {empty_status}")


if __name__ == "__main__":
    main()
