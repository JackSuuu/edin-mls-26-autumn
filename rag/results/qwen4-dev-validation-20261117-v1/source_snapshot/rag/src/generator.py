from __future__ import annotations

import os
import time
from dotenv import load_dotenv

SYSTEM_PROMPT = (
    "Use the supplied evidence to answer the user's question. "
    "Do not invent facts that are unsupported by the evidence."
)


class ELMGenerator:
    def __init__(self):
        from openai import OpenAI

        load_dotenv()

        api_key = os.environ.get("ELM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing ELM_API_KEY. Copy .env.example to .env and add your key."
            )

        base_url = os.environ.get(
            "ELM_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "https://elm.edina.ac.uk/api/v1"),
        )
        self.model = os.environ.get(
            "ELM_MODEL",
            "utter-project/EuroLLM-22B-Instruct-2512",
        )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=180,
        )

    def generate(self, prompt: str):
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "answer": response.choices[0].message.content.strip(),
            "latency_ms": elapsed_ms,
            "usage": response.usage.model_dump() if response.usage is not None else None,
            "response_model": response.model,
            "finish_reason": response.choices[0].finish_reason,
        }
