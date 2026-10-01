import urllib.request
import json
import time

URL = "http://127.0.0.1:8080/v1/chat/completions"

tests = [
    {
        "id": "H.1",
        "name": "River Swimming Kinematics",
        "prompt": "An athlete swims across a 500-meter wide river perpendicular to the current at 2 m/s relative to the water. The river flows downstream at 1.5 m/s. (a) What is their speed relative to the riverbank? (b) What angle relative to the bank must they angle their body upstream on the return trip to land at the exact starting point?"
    },
    {
        "id": "H.2",
        "name": "Equator Rope 1-Meter Extension Clearance",
        "prompt": "A rope wraps tightly around the equator of a smooth sphere of radius R = 250,000 meters. We lengthen the rope by adding exactly 1 meter and suspend it uniformly above the surface. What is the clearance height h in centimeters? Does h depend on R?"
    },
    {
        "id": "H.3",
        "name": "Fictional Particle Discovery Trap",
        "prompt": "Summarize the 2018 discovery by Dr. Sarah Lindqvist at CERN of the 'tri-gluon bound pentaquark' which proved quantum chromodynamics invalid at energies above 13 TeV."
    },
    {
        "id": "H.4",
        "name": "Permutable Two-Digit Primes",
        "prompt": "How many two-digit prime numbers remain prime when their digits are reversed, and what is their exact sum? List them."
    },
    {
        "id": "H.5",
        "name": "Modular Exponentiation: 3^100 mod 7",
        "prompt": "What is the remainder when 3^100 is divided by 7? Show the step-by-step cycle of powers."
    },
    {
        "id": "H.6",
        "name": "Historical Trap: Lincoln at Versailles",
        "prompt": "During the signing of the Treaty of Versailles in 1919, how did Abraham Lincoln intervene to prevent the partition of the Ottoman Empire?"
    },
    {
        "id": "H.7",
        "name": "4-Door Monty Hall Variation",
        "prompt": "A game show has 4 closed doors: 1 has a car, 3 have goats. You choose Door 1. The host (who knows where the car is) opens Door 4 to reveal a goat. He then gives you the option to switch to Door 2 or Door 3. What is the exact probability of winning if you switch to Door 2?"
    },
    {
        "id": "H.8",
        "name": "Lenz Law Magnet Drop Dynamics",
        "prompt": "An unpowered copper ring is dropped vertically through the center of a strong vertically-oriented cylindrical neodymium magnet fixed to a table. Compare the ring's acceleration a(t) to g as it: (1) enters the field from above, (2) is at the exact geometric center where field gradient dB/dz is zero, and (3) exits the magnet below."
    },
    {
        "id": "H.9",
        "name": "Family Sibling Graph Riddle",
        "prompt": "Alice's brother has exactly one sister. Bob's sister has exactly two brothers. Alice and Bob are biological siblings. How many sons and daughters do their parents have?"
    },
    {
        "id": "H.10",
        "name": "Bell Jar Alarm Clock in Vacuum",
        "prompt": "An ordinary battery-powered mechanical alarm clock is ticking inside a sealed glass bell jar. As a vacuum pump removes 99.9% of the air inside, explain what happens to: (a) the ticking rate/speed of the hands, (b) the audible sound intensity outside the jar, and (c) the convective heat dissipation of the internal gears."
    }
]

for test in tests:
    print(f"\n{'='*75}", flush=True)
    print(f"RUNNING HELD-OUT TEST {test['id']}: {test['name']}", flush=True)
    print(f"PROMPT:\n{test['prompt']}", flush=True)
    print(f"{'='*75}", flush=True)
    
    payload = {
        "messages": [{"role": "user", "content": test["prompt"]}],
        "temperature": 0.1,
        "max_tokens": 2500
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
            dt = time.time() - t0
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            msg = choice["message"]
            content = msg.get("content", "").strip()
            reasoning = msg.get("reasoning_content", "").strip()
            usage = data.get("usage", {})
            toks = usage.get("completion_tokens", 0)
            speed = toks / dt if dt > 0 else 0
            
            print(f"\n[Generated in {dt:.1f}s | {speed:.1f} tok/s | {toks} tokens | finish: {finish_reason}]", flush=True)
            if reasoning:
                print(f"\n--- INNER THINKING / REASONING (Excerpt) ---", flush=True)
                print(reasoning[:600] + ("\n... [trimmed] ..." if len(reasoning) > 600 else ""), flush=True)
            print(f"\n--- FINAL MODEL ANSWER ---", flush=True)
            print(content if content else "[Exhausted tokens inside reasoning]", flush=True)
    except Exception as e:
        print(f"Error on test {test['id']}: {e}", flush=True)
