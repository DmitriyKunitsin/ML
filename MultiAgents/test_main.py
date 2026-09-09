import asyncio
from providers.ollama_providers import AsyncOllamaClient
from core.base_agent import BaseAgent
from prompts.boss_promt import ARDUINO_PROMPT


async def main():
    ollama = AsyncOllamaClient()

    boss = BaseAgent(
        role_name="boss",
        role_prompt="Ты — строгий, но конструктивный тимлид и архитектор встроенных систем с 10-летним стажем разработки под Arduino.",
        llm=ollama,
    )
    result = await boss.execute_task(prompt=ARDUINO_PROMPT, task_type="boss")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
