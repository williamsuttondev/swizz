import os
import asyncio
from openai import AsyncOpenAI
from typing import Type

class Communicator:

    __KEY    = None 
    __client = None




    @classmethod
    def initialise_client(cls: Type['Communicator']):

        try:
            cls.__KEY = open("backend/API_KEY", "r").readline().rstrip()
            cls.__client = AsyncOpenAI(api_key=cls._KEY)
        except (openai.AuthenticationError, FileNotFoundError) as e:
            print(f"Error initializing OpenAI client: {e}")

    @staticmethod
    async def sendRequest(prompt: str, code: str) -> None:

        if Communicator.__KEY == None:
            print("No key") # return JSON response
            return
        elif Communicator.__client == None:
            print("No client") # return JSON response
            return

        message = f"{prompt}\n\nHere is the code:\n```{code}\n```"

        stream = await Communicator.__client .chat.completions.create(
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
