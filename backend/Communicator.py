
"""
Usage of modules
================

asyncio             - Allows for asynchronous code to be written providing syntax for await and async respectively

AsyncOpenAI         - Allows for aysnchronous versions of functions provided by openAI

AuthenticationError - A way to identify the specific error that occurred due to bad auth, this is most liekly going to occur due to a bad API key

Type                - Allows type hinting for Class types in the static context rather than an instance of the class

"""


import asyncio
from openai import AsyncOpenAI
from openai import AuthenticationError  # Import the correct exception
from typing import Type

class Communicator:
    """
        @author     : Brandon Wright - Barnold8

        This class manages the communications between the user and the OpenAI chatGPT model. 
    """

    __KEY          = None  # The API key needed to access openAI's models
    __client       = None  # The client object that will be needed to handle communications between this program and openAI's model
    __Error        = None  # A global error variable to store the current error and pass it on in the JSON response if it is needed
    __conversation = ""    # The conversation thus far to ensure persistance in the conversation between user and AI model, stored as a string

    @classmethod
    def initialise_client(cls: Type['Communicator']):

        """
            @author     : Brandon Wright - Barnold8

            Initialises the API key and the openAI client respectively to allow for communications between the user and the openAI model

            :cls: Reference to the Communicator class, weird way of allowing python to modify static variables but here we are. 
        """

        try:
            cls.__KEY = open("backend/API_KEY", "r").readline().rstrip()
            cls.__client = AsyncOpenAI(api_key=cls.__KEY)
        except (AuthenticationError, FileNotFoundError) as e:
            cls.__Error = e

    @classmethod
    def clearConversation(cls):
        """
            @author     : Brandon Wright - Barnold8

            Clears the conversation by resetting Communicator.__conversation to an empty string

            :cls: Reference to the Communicator class, weird way of allowing python to modify static variables but here we are. 
        """
        cls.__conversation = ""
        

    async def __chunkConcatenation(stream) -> str:
        """
            @author    : Brandon Wright - Barnold8

            Takes incoming stream from the chatGPT model (it's response), and concatenates it all to a string for easier processing

            :param stream: The stream of information coming in from the chatGPT model, it will be the response it has given. 

            :return:   Returns a string of the model's response
            

         """
        chunks = [chunk.choices[0].delta.content or "" async for chunk in stream]
        return "".join(chunks)


    @staticmethod
    async def sendRequest(prompt: str, code: str) -> dict:
        """
            @author    : Brandon Wright - Barnold8

            Sends the request out to the openAI chatGPT model. It will keep a consistant track of the conversation between user and 
            model to ensure context and consistancy throughout the entire conversation. 

            :param prompt: This is what the user is saying to the model. An example could be "Can you explain to me what this code does?"

            :param code: This is an optional parameter, this is just a string of all the code meant to be sent to the model. If this is empty, it will be ignored. 

            :return:    Returns a dictionary object of relevant information regarding the conversation encounter that has just transpired. If an error occurs, the relevant error information will be included. 
        
         """

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


Communicator.initialise_client()