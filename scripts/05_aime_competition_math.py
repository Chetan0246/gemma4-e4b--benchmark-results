import urllib.request
import json
import time

URL = "http://127.0.0.1:8080/v1/chat/completions"

prompt = """Find the number of positive integers n <= 1000 such that floor(sqrt(n)) divides n.
Provide your complete step-by-step mathematical reasoning, establish the algebraic conditions for each integer k = floor(sqrt(n)), carefully evaluate the boundary cases up to 1000, and state the final integer answer clearly."""

payload = {
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.1,
    "max_tokens": 3000
}

print("Running AIME Benchmark on local model...", flush=True)
req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=180) as resp:
    data = json.loads(resp.read().decode())
    dt = time.time() - t0
    msg = data["choices"][0]["message"]
    content = msg.get("content", "").strip()
    reasoning = msg.get("reasoning_content", "").strip()
    toks = data.get("usage", {}).get("completion_tokens", 0)
    speed = toks / dt if dt > 0 else 0
    print(f"Completed in {dt:.1f}s | {speed:.1f} tok/s | {toks} tokens")
    print("\n--- REASONING TRACE ---")
    print(reasoning)
    print("\n--- FINAL ANSWER ---")
    print(content)
