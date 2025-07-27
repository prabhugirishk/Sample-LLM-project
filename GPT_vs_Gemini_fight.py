import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI

# constants
env_path = r"C:\Users\user\Girish-projects\llm_engineering\.env"
load_dotenv(dotenv_path=env_path, override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GEMINI_API_KEY')


if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:8]}")
else:
    print("Google API Key not set") 

openai_GPT = OpenAI()
#or if you want to use ollama 
#openai_ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')


gpt_model = "gpt-4.1-mini"
gemini_model = "gemini-2.5-flash"
ollama_model = "llama3.2"

openai = OpenAI()

gemini_via_openai_client = OpenAI(
    api_key=google_api_key, 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


# Let's make a conversation between GPT-4.1-mini and Gemini
gpt_system = "You are a person who is well versed in Advaita philosophy propounded by Shankarcharya. \
You try to put forth the tenets of Advaita philosophy. You are very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way.\
Your responses should be less than 100 words"

gemini_system = "You are a person well versed in Dvaita philosophy propounded by Madhvacharya. \
But you negate Advaita and propound the tenets of Dvaita, also known as Tattvavada\
As per this Brahman is not the same as the Atman and Atman is a servant of Lord\
You are witty in your responses \
The world cannot be an illusion as per Dvaita. You refer to online resources such as 'Philosophy of Madhvacharya' by BNK Sharma\
You provide references and quotes from Bhagavadgita, or Upanishads as much as possible to support your theory.\
You are very calm and provie the point of view that gels well with Madhva's Dvaita or Tattvavada philosophy.\
Your responses should be less than 100 words."

def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, gemini in zip(gpt_messages, gemini_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": gemini})
    completion = openai.chat.completions.create(
        model=gpt_model,
        messages=messages
    )
    return completion.choices[0].message.content

def call_gemini():
    messages = []
    for gpt, gemini_message in zip(gpt_messages, gemini_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": gemini_message})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    message = gemini_via_openai_client.chat.completions.create(
        model=gemini_model,
        messages=messages
    )
    return message.choices[0].message.content

gpt_messages = ["Hi there"]
gemini_messages = ["Hi"]

print(f"GPT:\n{gpt_messages[0]}\n")
print(f"Gemini:\n{gemini_messages[0]}\n")

for i in range(3):
    gpt_next = call_gpt()
    print(f"GPT:\n{gpt_next}\n")
    gpt_messages.append(gpt_next)
    
    gemini_next = call_gemini()
    print(f"Gemini:\n{gemini_next}\n")
    gemini_messages.append(gemini_next)