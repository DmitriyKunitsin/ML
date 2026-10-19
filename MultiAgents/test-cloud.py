import asyncio
import logging
import httpx

from typing import Any
from uuid import uuid4

from key.token import access_token

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)

async def main():
    timeout_config = httpx.Timeout(5 * 60.0)
    base_url = 'https://552fa2c0-f962-4028-be18-a68f73c74699-agent.ai-agent.inference.cloud.ru'
    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        httpx_client.headers['Authorization'] = f'Bearer <{access_token}>'
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
        )
        public_card =  await resolver.get_agent_card()
        client = A2AClient(
            httpx_client=httpx_client,
            agent_card=public_card,
        )
        send_message_payload: dict[str, Any] = {
            'message': {
                'role': 'user',
                'parts': [
                    {'kind': 'text', 'text': 'how much is 10 USD in RUB?'}
                ],
                'messageId': uuid4().hex,
            },
        }
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )
        response = await client.send_message(request)
        print("Ответ : \n")
        print(response.model_dump(mode='json', exclude_none=True))

if __name__ == "__main__":
    asyncio.run(main())
