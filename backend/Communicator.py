import os
import asyncio
from openai import AsyncOpenAI
from openai import AuthenticationError  # Import the correct exception
from typing import Type

class Communicator:

    __KEY          = None 
    __client       = None
    __Error        = None
    __conversation = ""



    @classmethod
    def initialise_client(cls: Type['Communicator']):

        try:
            cls.__KEY = open("backend/API_KEY", "r").readline().rstrip()
            cls.__client = AsyncOpenAI(api_key=cls.__KEY)
        except (AuthenticationError, FileNotFoundError) as e:
            cls.__Error = e

    async def __chunkConcatenation(stream) -> str:
        chunks = [chunk.choices[0].delta.content or "" async for chunk in stream]
        return "".join(chunks)


    @staticmethod
    async def sendRequest(prompt: str, code: str) -> None:

        print("Contacting GPT model...")

        if Communicator.__KEY == None:
            response = {
                "prompt": prompt,
                "code": code,
                "model_response": None,
                "conversation" : Communicator.__conversation,
                "Error": "No API was given to the communicator class. Have you ran Communicator.initialise_client()? Is there a valid API key in backend/API_KEY?",
                "Error_RAW": Communicator.__Error
            }
            return response
        elif Communicator.__client == None:
            response = {
                "prompt": prompt,
                "code": code,
                "model_response": None,
                "conversation" : Communicator.__conversation,
                "Error": "A fatal error ocurred when initalising the client in the Communicator class!",
                "Error_RAW": Communicator.__Error
            }
            return response
        
        message = f"{Communicator.__conversation}\n{prompt}\n\nHere is the code:\n```{code}\n```" if len(code) > 0 else f"{Communicator.__conversation}\n{prompt}"

        stream = await Communicator.__client .chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                 "content": message
                }],
            stream=True,
        )

        model_response = await Communicator.__chunkConcatenation(stream)

        Communicator.__conversation += f"User: {message}\n"
        Communicator.__conversation += f"ChatGPT: {model_response}\n"
        
        response = {
            "prompt": prompt,
            "code": code,
            "model_response": model_response,
            "conversation" : Communicator.__conversation
        }

        return response

