import json
import time

import laya

agent = laya.load("convaiinnovations/laya")
q = laya.triage_questions()

print("=== QUESTION FORMAT ===")
print(json.dumps(q, indent=2, default=str))

state = {"message": "My payment failed twice"}
agent.predict(state, q)  # warm-up
t = time.time()
r = agent.predict(state, q)
print("\n=== SECONDS PER CALL (short) ===", round(time.time() - t, 3))
print("\n=== RAW RESULT ===")
print(json.dumps(r, indent=2, default=str))

# How is the 8192 limit enabled? Try both ways and report.
long_state = {"message": "word " * 3000}      # about 3000 tokens
for label, fn in [
    ("predict(max_len=8192)", lambda: agent.predict(long_state, q, max_len=8192)),
    ("plain predict", lambda: agent.predict(long_state, q)),
]:
    try:
        t = time.time(); fn()
        print(f"\n=== {label}: OK, {round(time.time()-t,2)}s for ~3000 tokens ===")
    except Exception as e:
        print(f"\n=== {label}: FAILED -> {e!r} ===")

try:
    agent8 = laya.load("convaiinnovations/laya", max_len=8192)
    t = time.time(); agent8.predict(long_state, q)
    print(f"\n=== load(max_len=8192): OK, {round(time.time()-t,2)}s ===")
except Exception as e:
    print("\n=== load(max_len=8192): FAILED ->", repr(e))