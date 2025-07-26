# imports
# imports
# If these fail, please check you're running from an 'activated' environment with (llms) in the command prompt

import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI

# constants

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key)>10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")
    
#MODEL = 'gpt-4o-mini'
openai_GPT = OpenAI()
openai_ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA = 'llama3.2'

# set up environment
system_prompt = "You are provided with a question related to a programming language.\
The query could be either a snippet of code or a question regarding the syntax \
You are able to answer the question based on the relevant programming language.\
You also suggest usage of the piece of code and share some examples .\n"

def user_prompt_for(query):
    user_prompt = "You are looking at a query related to a snippet of code"
    user_prompt += "\nThe contents of this query is as follows; \
please provide what this code does and why does it do so. \
If possible provide other examples related to this query.\n\n"
    user_prompt += query
    return user_prompt



# Get gpt-4o-mini to answer, with streaming
def stream_query_GPT(query):
    stream = openai_GPT.chat.completions.create(
        model=MODEL_GPT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_for(query)}
          ],
        stream=True
    )
    
    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        response = response.replace("```","").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)

# Get Llama 3.2 to answer, with streaming
def stream_query_ollama(query):
    stream = openai_ollama.chat.completions.create(
        model=MODEL_LLAMA,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_for(query)}
          ],
        stream=True
    )
    
    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        response = response.replace("```","").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)

#Or use Ollama directly without openai API
# OLLAMA_API = "http://localhost:11434/api/chat"
# HEADERS = {"Content-Type": "application/json"}
# MODEL = "llama3.2"
# messages = [
#     {"role": "user", "content": "Describe some of the business applications of Generative AI"}
# ]
# payload = {
#         "model": MODEL,
#         "messages": messages,
#         "stream": False
#     }
# response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
# print(response.json()['message']['content'])
#response = ollama.chat(model=MODEL, messages=messages)
#print(response['message']['content'])


#Enter the question here Or accept it as user input
question = """
Please explain what this code does and why:
yield from {book.get("author") for book in books if book.get("author")}
"""


# Call either the Ollama function or GPT function
stream_query_GPT(question)
stream_query_ollama(question)
