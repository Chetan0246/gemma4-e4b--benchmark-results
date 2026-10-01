import urllib.request
import json
import time

URL = "http://127.0.0.1:8080/v1/chat/completions"

tests = [
    {
        "id": 1,
        "category": "Distributed Systems Architecture (70B-Level)",
        "prompt": "Design a globally distributed, low-latency API rate limiter supporting 100,000 requests/sec with sliding window counter semantics. Detail the trade-offs between consistency and availability (CAP theorem) during cross-datacenter WAN partitions, explain how to eliminate cache stampedes / thundering herds, and provide the exact atomic Redis Lua script used to increment and evaluate the sliding window counter."
    },
    {
        "id": 2,
        "category": "Rigorous Mathematical Proof (Fermat's Method)",
        "prompt": "Prove that the square root of 2 is irrational using Fermat's Method of Infinite Descent (Descente Infinie) rather than the standard even/odd parity contradiction proof. Clearly state the geometric or algebraic descent step, demonstrate why a strictly decreasing sequence of positive integers leads to a contradiction, and explain precisely why this construction fails for the square root of 4."
    },
    {
        "id": 3,
        "category": "Cross-Domain Metaphysical & OS Architecture Synthesis",
        "prompt": "Compare the 'Ship of Theseus' identity paradox to Linux Kernel Virtual Memory management with Copy-on-Write (CoW) pages during fork(). Analyze whether a memory page modified by a child process represents an Endurantist or Perdurantist object of identity. Formulate a precise technical and philosophical argument."
    }
]

for test in tests:
    print(f"\n{'='*75}", flush=True)
    print(f"FRONTIER BENCHMARK {test['id']}: {test['category']}", flush=True)
    print(f"{'='*75}", flush=True)
    
    payload = {
        "messages": [{"role": "user", "content": test["prompt"]}],
        "temperature": 0.2,
        "max_tokens": 3000
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode())
            dt = time.time() - t0
            msg = data["choices"][0]["message"]
            content = msg.get("content", "").strip()
            reasoning = msg.get("reasoning_content", "").strip()
            usage = data.get("usage", {})
            toks = usage.get("completion_tokens", 0)
            speed = toks / dt if dt > 0 else 0
            
            print(f"\n[Completed in {dt:.1f}s | {speed:.1f} tok/s | {toks} tokens]", flush=True)
            if reasoning:
                print(f"\n--- REASONING SUMMARY (first 500 chars) ---", flush=True)
                print(reasoning[:500] + ("..." if len(reasoning) > 500 else ""), flush=True)
            print(f"\n--- MODEL SYNTHESIS & OUTPUT ---", flush=True)
            print(content if content else "[Generated entire output in reasoning tokens]", flush=True)
    except Exception as e:
        print(f"Error on test {test['id']}: {e}", flush=True)
