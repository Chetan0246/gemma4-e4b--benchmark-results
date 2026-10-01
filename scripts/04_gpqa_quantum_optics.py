import urllib.request
import json
import time

URL = "http://127.0.0.1:8080/v1/chat/completions"

prompt = """Consider a two-level atom with transition frequency omega_0 coupled to a single-mode resonant optical cavity (omega_c = omega_0) with coupling strength g, described by the resonant Jaynes-Cummings model in the rotating wave approximation:
H = hbar * omega_0 * (a^dagger a + 1/2 * sigma_z) + hbar * g * (a^dagger sigma_- + a sigma_+)

At time t = 0, the atom is prepared in the excited state |e> and the cavity is in a coherent state |alpha> with average photon number n_bar = |alpha|^2 >> 1 (real alpha).

1. Provide the exact analytical expression for the atomic inversion <sigma_z(t)> as an explicit sum over the Fock state photon number distribution P(n).
2. Derive the exact characteristic timescale for:
   (a) The initial quantum collapse of Rabi oscillations (t_collapse), in terms of g and n_bar.
   (b) The first quantum revival of Rabi oscillations (t_revival), in terms of g and n_bar.
3. Explain precisely why the quantum revivals are imperfect (i.e. why the envelope of oscillations does not return to 100% of its initial amplitude), identifying the specific mathematical property of the Jaynes-Cummings spectrum responsible for this dephasing."""

payload = {
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.2,
    "max_tokens": 1500
}

print("Dispatching GPQA Diamond-level Quantum Optics benchmark to gemma-4...", flush=True)
req = urllib.request.Request(URL, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
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
            print("\n=== REASONING (EXCERPT) ===", flush=True)
            print(reasoning[:800] + ("\n... [trimmed reasoning] ..." if len(reasoning) > 800 else ""), flush=True)
        print("\n=== FINAL MODEL SOLUTION ===", flush=True)
        print(content if content else "[Generated entirely in reasoning_content]", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
