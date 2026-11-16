"""Call a running FreeWidth server with plain HTTP (any OpenAI-compatible client works too).

    python -m freewidth serve --config configs/engine_mock.yaml &
    python examples/client.py "What is 17 * 23?"
"""
import json
import os
import sys
import urllib.request

url = os.environ.get("FREEWIDTH_URL", "http://localhost:8000/v1/chat/completions")
question = " ".join(sys.argv[1:]) or "What is 17 * 23?"
body = {
    "model": "freewidth",
    "messages": [{"role": "user", "content": question}],
    # optional per-request policy overrides
    "freewidth": {"name": "freewidth", "tau": 0.9},
}
headers = {"Content-Type": "application/json"}
if os.environ.get("FREEWIDTH_API_KEY"):
    headers["Authorization"] = f"Bearer {os.environ['FREEWIDTH_API_KEY']}"
req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
with urllib.request.urlopen(req, timeout=900) as r:
    out = json.loads(r.read())
print(out["choices"][0]["message"]["content"])
fw = out["freewidth"]
print(f"\nanswer={fw['answer']} confidence={fw['confidence']} model={fw['model']} "
      f"waves={len(fw['waves'])} tokens={fw['tokens']} seconds={fw['seconds']} "
      f"escalated={fw['escalated']} verified={fw['verified']}")
