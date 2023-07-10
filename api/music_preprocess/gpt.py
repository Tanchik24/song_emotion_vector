import numpy as np
import openai
from api.response import ChatGptResponse


def get_description(arousal: float, valence: float) -> ChatGptResponse:
    """
    Generates a description of a song based on its arousal and valence values using OpenAI GPT-3 model.

    Args:
        arousal (float): Array of arousal values of the song
        valence (float): Array of valence values of the song

    Returns:
        str: A description of the song based on its arousal and valence values generated by the GPT-3 model.
    """
    openai.api_key = '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'

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": f"Задача: опиши мне эмоцию песни не больше 100 токенов, у которой arousal = {arousal}, valence = {valence}"}])
    response = ChatGptResponse(text=completion.choices[0].message.content)

    return response
