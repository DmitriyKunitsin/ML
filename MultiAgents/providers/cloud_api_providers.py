import asyncio
import sys
import time
import httpx

from core.base_llm import BaseLLM


class CloudAPIProvider(BaseLLM):
    def __init__(
        self, model_name: str, api_key: str, url: str = "https://deepseek.com"
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.url = f"{url}/chat/completions"

    async def _generate(
        self,
        model_role: str,
        prompt: str,
        system_prompt: str = "",
        task_type: str = "chat",
    ) -> str | None:
        """реализация _generate"""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"]
