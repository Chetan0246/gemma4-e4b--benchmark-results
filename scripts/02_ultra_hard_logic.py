import urllib.request
import json
import time

URL = "http://127.0.0.1:8080/v1/chat/completions"

tests = [
    {
        "id": 1,
        "name": "The Impossible Average Speed Trap (Harmonic Mean Paradox)",
        "prompt": "You drive up a 60-mile hill at an average speed of 30 mph. You want your overall average speed for the entire round trip (60 miles up and 60 miles back down, total 120 miles) to be 60 mph. How fast must you drive on the way down? Provide the step-by-step algebraic calculation and state the required speed."
    },
    {
        "id": 2,
        "name": "Multi-Layered Deep Citation & Author Fabrication Trap",
        "prompt": "In the influential 2017 NeurIPS paper 'Attention Is Not All You Need' co-authored by Yann LeCun and Geoffrey Hinton, what was the exact mathematical formulation of the 'quadratic manifold collapse' described in Section 4.2, and how did their Capsule Attention routing algorithm resolve it?"
    },
    {
        "id": 3,
        "name": "Riddle Modification Trap (Overriding Memorized Patterns)",
        "prompt": "A farmer is on the south bank of a river with a wolf, a goat, and a cabbage. He has a boat. But pay close attention to the rules:\n1. The boat can carry the farmer plus TWO items at the same time.\n2. The wolf is a vegetarian and will only eat the cabbage if left alone with it.\n3. The goat is completely satiated and will never eat anything.\n4. Neither the goat nor the wolf will harm each other.\nWhat is the minimum number of river crossings required for the farmer to safely transport all three to the north bank? List each crossing step by step."
    },
    {
        "id": 4,
        "name": "Hypothetical Legal/Kinship Contradiction",
        "prompt": "A man points to a photograph and says: 'Brothers and sisters I have none, but that man's father is my father's son.' Exactly at that moment, the man's biological daughter enters and points to the same photograph, saying: 'The person in that photo is my grandfather's brother-in-law.' Assuming strictly biological, legal relationships without divorce, adoption, or second marriages, explain who is in the photo and whether both statements can be simultaneously true."
    }
]

for test in tests:
    print(f"\n{'='*75}", flush=True)
    print(f"RUNNING ULTRA-HARD TEST {test['id']}: {test['name']}", flush=True)
    print(f"PROMPT:\n{test['prompt']}", flush=True)
    print(f"{'='*75}", flush=True)
    
    payload = {
        "messages": [{"role": "user", "content": test["prompt"]}],
        "temperature": 0.2,
        "max_tokens": 1000
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
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
                print(f"\n--- INNER THINKING / REASONING (first 700 chars) ---", flush=True)
                if len(reasoning) > 700:
                    print(reasoning[:700] + "\n... [trimmed reasoning] ...", flush=True)
                else:
                    print(reasoning, flush=True)
            print(f"\n--- FINAL MODEL ANSWER ---", flush=True)
            print(content if content else "[Exhausted tokens in reasoning]", flush=True)
    except Exception as e:
        print(f"Error on test {test['id']}: {e}", flush=True)
