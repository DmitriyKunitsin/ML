import asyncio
import sys
import time
import httpx
from uuid import uuid4
from core.base_llm import BaseLLM

class CloudAPIProvider(BaseLLM):
    def __init__(self, model_name: str, api_key: str, url: str, timeout: float = 300.0):
        # 1. Корректно инициализируем родительский класс и передаем таймаут для таймера
        super().__init__(timeout=timeout)
        self.model_name = model_name
        self.api_key = api_key
        # Сохраняем чистый базовый хост инференса
        self.url = url.rstrip('/')

    async def _get_context_limit(self) -> int:
        """
        Асинхронная реализация абстрактного метода родителя.
        Сообщает лимит контекста для модели фабрики агентов (например, 64к).
        """
        return 65536
    async def _get_context_limit(self, *args, **kwargs) -> int:
        """
        Асинхронная реализация абстрактного метода родителя.
        Принимает любые переданные аргументы (например, task_type) 
        и возвращает лимит контекста модели (64к).
        """
        return 65536

    async def _generate(self, *args, **kwargs) -> str | None:
        """
        Универсальная реализация, которая принимает любые аргументы (*args, **kwargs),
        чтобы не падать при изменении их порядка в базовом классе.
        """
        # Безопасно извлекаем prompt и system_prompt, где бы они ни находились
        # (в kwargs или позиционно в args)
        prompt = kwargs.get("prompt")
        system_prompt = kwargs.get("system_prompt", "")
        
        # Если аргументы передали позиционно (без имен), маппим их по стандартному порядку:
        # Обычно это: prompt (если 1-й), или model_role, prompt, system_prompt
        if not prompt and len(args) > 0:
            if len(args) == 1:
                prompt = args[0]
            elif len(args) == 2:
                # возможно передали (model_role, prompt) или (prompt, system_prompt)
                prompt = args[1] if isinstance(args[0], str) and len(args[0]) < 30 else args[0]
                system_prompt = args[0] if prompt == args[1] else args[1]
            else:
                # Если передали 3+ аргумента (model_role, prompt, system_prompt)
                prompt = args[1]
                system_prompt = args[2] if len(args) > 2 else ""

        # На всякий случай дефолтная заглушка, если промпт совсем пустой
        if not prompt:
            prompt = "Передан пустой запрос."

        # Дальше идет ваш рабочий код запроса к Cloud.ru
        timeout_config = httpx.Timeout(self.timeout)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        full_text = prompt
        if system_prompt:
            full_text = f"{system_prompt}\n\nКонтекст текущей задачи:\n{prompt}"
            
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
        
        chat_url = f"{self.url}/v1/chat/completions"
        
        async with httpx.AsyncClient(timeout=timeout_config, headers=headers) as client:
            response = await client.post(chat_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data["result"]["message"]["parts"]["text"]


    # async def _generate(
    #     self,
    #     model_role: str,
    #     prompt: str,
    #     system_prompt: str = "",
    #     task_type: str = "chat",
    # ) -> str | None:
    #     """Реализация через прямые HTTP-запросы (JSON-RPC) под формат Cloud.ru A2A"""
        
    #     # Берем таймаут, настроенный в конструкторе
    #     timeout_config = httpx.Timeout(self.timeout)
        
    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json"
    #     }
        
    #     # Склеиваем роль агента из стейт-машины и его текущую задачу
    #     full_text = prompt
    #     if system_prompt:
    #         full_text = f"{system_prompt}\n\nКонтекст текущей задачи:\n{prompt}"
            
    #     # Формируем структуру JSON-RPC запроса фабрики Cloud.ru
    #     payload = {
    #         "id": str(uuid4()),
    #         "jsonrpc": "2.0",
    #         "method": "message/send",
    #         "params": {
    #             "message": {
    #                 "role": "user",
    #                 "parts": [
    #                     {"kind": "text", "text": full_text}
    #                 ],
    #                 "messageId": uuid4().hex
    #             }
    #         }
    #     }
        
    #     chat_url = f"{self.url}/chat"
        
    #     async with httpx.AsyncClient(timeout=timeout_config, headers=headers) as client:
    #         response = await client.post(chat_url, json=payload)
    #         response.raise_for_status()
            
    #         response_data = response.json()
            
    #         # Извлекаем текст ответа из структуры A2A
    #         return response_data["result"]["message"]["parts"]["text"]
