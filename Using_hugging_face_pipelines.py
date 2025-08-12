# Imports
# if this gives an "ERROR" about pip dependency conflicts, ignore it! It doesn't affect anything.
#!pip install -q --upgrade torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
#!pip install -q --upgrade transformers==4.55.0 datasets==3.2.0 diffusers

import torch
from google.colab import userdata #if using google.colab
from huggingface_hub import login
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio

#get a hugging face token on hugginface.co
hf_token = userdata.get('HF_TOKEN')
login(hf_token, add_to_git_credential=True)

# Sentiment Analysis

classifier = pipeline("sentiment-analysis") # if using google.colab -> pipeline("sentiment-analysis", device="cuda")
result = classifier("I'm extremely enthusiastic to learn about LLM!!")
print(result)

# Named Entity Recognition

ner = pipeline("ner", grouped_entities=True, device="cuda")
result = ner("Narendra Modi is the 18th Prime Minister of India. Prior to Modi was Manmohan Singh")
print(result)

# Question Answering with Context

question_answerer = pipeline("question-answering", device="cuda")
result = question_answerer(question="Who was the 18th Prime Minister of India?", context="Narendra Modi is the 18th Prime Minister of India.")
print(result)

# Text Summarization

summarizer = pipeline("summarization", device="cuda")
text = """Madhvacharya  pronounced ; 1199–1278 CE[4] or 1238–1317 CE[5]), also known as Purna Prajna and Ānanda Tīrtha, was an Indian philosopher, 
theologian and the chief proponent of the Dvaita (dualism) school of Vedanta.[1][6] Madhva called his philosophy Tattvavāda meaning "arguments from a realist viewpoint".[6]
Madhvacharya was born at Pajaka near Udupi on the west coast of Karnataka state in 13th-century India.[7] 
As a teenager, he became a Sanyasi (monk) joining Brahma-sampradaya guru Achyutapreksha, of the Ekadandi order.[1][3] 
Madhva studied the classics of Hindu philosophy, and wrote commentaries on the Principal Upanishads, the Bhagavad Gita and the Brahma Sutras (Prasthanatrayi),[1]
 and is credited with thirty seven works in Sanskrit.[8] His writing style was of extreme brevity and condensed expression. 
 His greatest work is considered to be the Anuvyakhyana, a philosophical supplement to his bhasya on the Brahma Sutras composed with a poetic structure.[7] 
In some of his works, he proclaimed himself to be an avatar of Vayu, the son of god Vishnu.[9][10
"""
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
print(summary[0]['summary_text'])

# Translation

translator = pipeline("translation_en_to_fr", device="cuda")
result = translator("The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API.")
print(result[0]['translation_text'])

# Another translation, showing a model being specified
# All translation models are here: https://huggingface.co/models?pipeline_tag=translation&sort=trending

translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es", device="cuda")
result = translator("The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API.")
print(result[0]['translation_text'])

# Classification

classifier = pipeline("zero-shot-classification", device="cuda")
result = classifier("Hugging Face's Transformers library is amazing!", candidate_labels=["technology", "sports", "politics"])
print(result)
# Text Generation

generator = pipeline("text-generation", device="cuda")
result = generator("If there's one thing I want you to remember about using HuggingFace pipelines, it's")
print(result[0]['generated_text'])


# Image Generation

image_gen = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
    ).to("cuda")

text = "A class of Data Scientists learning about AI, in the surreal style of Salvador Dali"
image = image_gen(prompt=text).images[0]
image

# Audio Generation

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = synthesiser("Hi to an artificial intelligence engineer, on the way to mastery!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
Audio("speech.wav")
