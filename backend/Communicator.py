import os
import asyncio
from openai import AsyncOpenAI



class Communicator:
    _KEY = open("backend/API_KEY", "r").readline().rstrip()
    _client = AsyncOpenAI(api_key=_KEY)

    @staticmethod
    async def sendRequest(prompt: str, code: str) -> None:


        message = f"{prompt}\n\nHere is the code:\n```{code}\n```"

        stream = await Communicator._client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                 "content": message
                }],
            stream=True,
        )
        async for chunk in stream:
            print(chunk.choices[0].delta.content or "", end="")



message = "".join(open("frontend/main.py","r").readlines())


asyncio.run(Communicator.sendRequest("Can you explain what this code does?", message))
