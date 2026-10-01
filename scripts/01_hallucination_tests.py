import urllib.request
import json
import time
import sys

URL = "http://127.0.0.1:8080/v1/chat/completions"

tests = [
    {
        "id": 1,
        "name": "Historical Anachronism (False Premise)",
        "prompt": "When Thomas Jefferson visited Thomas Edison in Menlo Park in 1879, what specific advice did Jefferson give Edison regarding the incandescent light bulb filament?"
    },
    {
        "id": 2,
        "name": "Fictional Science & Nobel Prize",
        "prompt": "Explain the core principles of the Henderson-Kaufman Theorem of Superconductive Fluidity, and what year it won the Nobel Prize in Physics."
    },
    {
        "id": 3,
        "name": "Common Sense Parallel Logic",
        "prompt": "If it takes 2 hours for 1 wet shirt to dry outside in the sun, how many hours will it take for 5 identical wet shirts hanging on the same clothesline to dry? Give your mathematical calculation."
    },
    {
        "id": 4,
        "name": "Tokenization & Character Counting",
        "prompt": "How many times does the letter 'r' appear in the word 'Strawberry'? Count step by step."
    }
]

for test in tests:
    print(f"\n{'='*70}", flush=True)
    print(f"RUNNING TEST {test['id']}: {test['name']}", flush=True)
    print(f"PROMPT: {test['prompt']}", flush=True)
    print(f"{'='*70}", flush=True)
    
    payload = {
        "messages": [{"role": "user", "content": test["prompt"]}],
        "temperature": 0.1,
        "max_tokens": 2500
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            dt = time.time() - t0
            msg = data["choices"][0]["message"]
            content = msg.get("content", "").strip()
            reasoning = msg.get("reasoning_content", "").strip()
            usage = data.get("usage", {})
            toks = usage.get("completion_tokens", 0)
            speed = toks / dt if dt > 0 else 0
            
            print(f"\n[Generated in {dt:.1f}s | {speed:.1f} tok/s | {toks} tokens]", flush=True)
            if reasoning:
                print(f"\n--- INNER THINKING / REASONING ---", flush=True)
                # Print first 500 chars of reasoning if long
                if len(reasoning) > 600:
                    print(reasoning[:500] + "\n... [trimmed reasoning] ...", flush=True)
                else:
                    print(reasoning, flush=True)
            print(f"\n--- FINAL MODEL ANSWER ---", flush=True)
            print(content if content else "[Model exhausted tokens inside reasoning]", flush=True)
    except Exception as e:
        print(f"Error on test {test['id']}: {e}", flush=True)
