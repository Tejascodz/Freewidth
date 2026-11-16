"""Use the official OpenAI Python SDK against a FreeWidth server (pip install openai)."""
import os

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1",
                api_key=os.environ.get("FREEWIDTH_API_KEY", "not-needed-on-an-unauthenticated-local-server"))
resp = client.chat.completions.create(
    model="freewidth",
    messages=[{"role": "user", "content": "A train travels 60 km in 45 minutes. What is its speed in km/h?"}],
    extra_body={"freewidth": {"tau": 0.9, "max_waves": 2}},
)
print(resp.choices[0].message.content)
print(resp.model_extra.get("freewidth"))
