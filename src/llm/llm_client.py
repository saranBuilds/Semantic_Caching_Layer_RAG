"""Talks to local Ollama and reports token usage for cost/latency tracking."""
import time
import requests


class OllamaClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Returns a dict with the answer text plus usage/latency info,
        so the caller can log it (this is your Phase 4 data later).
        """
        start = time.time()
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        latency = time.time() - start

        return {
            "answer": data.get("response", "").strip(),
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "latency_seconds": round(latency, 3),
        }