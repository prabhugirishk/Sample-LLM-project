# !pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
# !pip install -q requests bitsandbytes==0.46.0 transformers==4.48.3 accelerate==1.3.0 openai
# imports

import os
import requests
from IPython.display import Markdown, display, update_display
from openai import OpenAI
from google.colab import drive
from huggingface_hub import login
from google.colab import userdata
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
# Constants

AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# If using Google colab
# New capability - connect this Colab to Google Drive

drive.mount("/content/drive")
audio_filename = "audio_extract.mp3"

audio_file = open(audio_filename, "rb") 
# Sign in to HuggingFace Hub

hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)


# If using Colab - Sign in to OpenAI using Secrets in Colab
#
# openai_api_key = userdata.get('OPENAI_API_KEY')
# openai = OpenAI(api_key=openai_api_key) #else read from .env


# transcription = openai.audio.transcriptions.create(model=AUDIO_MODEL, file=audio_file, response_format="text")
# print(transcription)

#If not using Openai, use open source model

AUDIO_MODEL = "openai/whisper-medium"
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
speech_model.to('cuda') # if using Nvida and colab
processor = AutoProcessor.from_pretrained(AUDIO_MODEL)

pipe = pipeline(
    "automatic-speech-recognition",
    model=speech_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device='cuda',
)
audio_file = open(audio_filename, "rb")
result = pipe(audio_filename, return_timestamps=True)

transcription = result["text"]

#pass this transcription to the user_prompt

system_message = "You are an assistant that produces minutes of meetings from transcripts, with summary, key discussion points, takeaways and action items with owners, in markdown."
user_prompt = f"Below is an extract transcript of a Denver council meeting. Please write minutes in markdown, including a summary with attendees, location and date; discussion points; takeaways; and action items with owners.\n{transcription}"

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="auto", quantization_config=quant_config)
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer) //to generate streaming response

response = tokenizer.decode(outputs[0])
display(Markdown(response))

