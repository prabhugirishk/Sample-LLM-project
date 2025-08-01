''''Ice-cream-assitant-voice2voice
Chatbot with Gradio + audio assistant (takes audio input, outputs audio and text) + uses tools 
LLM decides to call a tool under a particular circumstance
'''


import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
import time
from mutagen.mp3 import MP3 # To get the length of the audio file

from pydub import AudioSegment
from pydub.playback import play

import random

# Initialization

load_dotenv(override=True)
    
MODEL = "gpt-4o-mini"
openai = OpenAI()
MODEL_ollama = "llama3.2"
openai_ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

system_message = "You are a helpful assistant for an ice cream vendor. "
system_message += "You specialize in tasty ice creams with multiple varieties. You can suggest different ice creams"
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

#function to convert speech to text
def speech_to_text():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)  # Using Google Web Speech API
        print(f"You said: {text}")
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API service; {e}")

    return(text)

#Function to convert text to speech
def text_to_speech(text):
    #text_to_say = "Hello Moda, this is a much more natural voice from Google."
    
    random_number = random.randint(1, 100)
   
    # Create the gTTS object
    gtts_object = gTTS(text=text, lang='en', slow=False)
    audio_file = "voice"+str(random_number)+".mp3"
    
    gtts_object.save(audio_file)
    sound = AudioSegment.from_mp3(audio_file)
    gtts_object.save(audio_file)
    
    #Get the duration of the audio file
    audio = MP3(audio_file)
    duration = audio.info.length

    print(f"Saying: {text}")
    # Play the audio file
    play(sound)
    
    time.sleep(duration + 1)
    
    # Clean up the file
    os.remove(audio_file)

# Let's start by making a useful function

ice_cream_prices = {"vanilla": "Rs.35", "chocolate": "Rs.45", "strawberry": "Rs.65", "pistachio": "Rs.50", "caramel": "Rs.40"}

def get_icecream_price(flavor):
    print(f"Tool get_icecream_price called for {flavor}")
    variety = flavor.lower()
    return ice_cream_prices.get(flavor, "Unknown")

# There's a particular dictionary structure that's required to describe our function:

price_function = {
    "name": "get_icecream_price",
    "description": "Get the price of a particular flavor of the ice cream. Call this whenever you need to know the price, for example when a customer asks 'How much is a cost of this flavor'",
    "parameters": {
        "type": "object",
        "properties": {
            "flavor": {
                "type": "string",
                "description": "The flavor that the customer wants to have",
            },
        },
        "required": ["flavor"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]

# We have to write that function handle_tool_call:

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    variety = arguments.get('flavor')
    price = get_icecream_price(variety)
    response = {
        "role": "tool",
        "content": json.dumps({"flavor": variety,"price": price}),
        "tool_call_id": tool_call.id
    }
    return response, variety

import threading
def chat(history):
    message = speech_to_text()
    message_from_speech = message
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    #image = None
    
    if response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        response, variety = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        #image = artist(city)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
        
    reply = response.choices[0].message.content
    history += [{"role":"assistant", "content":reply}]

    #run the text_to_speech function in the background to avoid slowing down of the chatbot
    thread = threading.Thread(target=text_to_speech(reply))
    thread.start()
    
    return history #, message_from_speech

# More involved Gradio code as we're not using the preset Chat interface!

with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=200, type="messages")
        #message_from_speech = gr.Textbox(label="Chat")
        #image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Button("Press button to start speaking")
    # with gr.Row():
    #     clear = gr.Button("Clear")

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history
    
    entry.click(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
       chat, inputs=chatbot, outputs=[chatbot])#, message_from_speech])        


ui.launch(inbrowser=True)

# -------Below code if doing only voice chat-------------
def voice_chat(audio_file):
    # Step 1: Transcribe audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        user_input = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that.", None

    # Step 2: Use GPT to generate a response
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": user_input}]
    )
    reply = response.choices[0].message.content

    # Step 3: Convert GPT text to speech
    tts = gTTS(reply)
    temp_audio_path = tempfile.mktemp(suffix=".mp3")
    tts.save(temp_audio_path)

    return reply, temp_audio_path

iface = gr.Interface(
    fn=voice_chat,
    inputs=gr.Audio(sources="microphone", type="filepath", label="Speak Here"),
    outputs=[
        gr.Text(label="Chatbot Response"),
        gr.Audio(label="Voice Response", type="filepath")
    ],
    title="🗣️ Voice Chat with GPT",
    description="Speak into the mic, GPT replies with both text and voice."
)

iface.launch()