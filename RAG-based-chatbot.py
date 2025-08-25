#!/usr/bin/env python
# coding: utf-8

# ## RAG based Chatbot that uses the local Knowledge base and responds
# 
# ## Chroma or FAISS!

# imports

import os
import glob
from dotenv import load_dotenv
import gradio as gr


# install faiss-cpu if needed!
# get_ipython().system('pip install faiss-cpu')


# imports for langchain

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from openai import OpenAI
from langchain_community.embeddings import OllamaEmbeddings


# price is a factor for our company, so we're going to use a low cost model

MODEL = "gpt-4o-mini"
db_name = "vector_db"
OLLAMA_MODEL = 'llama3.2'


# Load environment variables in a file called .env

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
openai_ollama = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')


# Read in documents using LangChain's loaders
# Take everything in all the sub-folders of our knowledgebase

folders = glob.glob("knowledge-base/*")

text_loader_kwargs = {'encoding': 'utf-8'}
# If that doesn't work, some Windows users might need to uncomment the next line instead
# text_loader_kwargs={'autodetect_encoding': True}

documents = []
for folder in folders:
    doc_type = os.path.basename(folder)
    loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)


len(chunks)

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}")


# ## A sidenote on Embeddings, and "Auto-Encoding LLMs"
# 
# We will be mapping each chunk of text into a Vector that represents the meaning of the text, known as an embedding.
# 
# OpenAI offers a model to do this, which we will use by calling their API with some LangChain code.
# 
# Put the chunks of data into a Vector Store that associates a Vector Embedding with each chunk
# Chroma is a popular open source Vector Database based on SQLLite

#embeddings = OpenAIEmbeddings()
# OLLAMA_MODEL of llama3.2 is good for chat, but not good for embeddings
# OllamaEmbeddings () of Langchain can be directly used to create embeddings instead of OpenAI()
embeddings = OllamaEmbeddings(model="mxbai-embed-large") 
# Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
# Create vectorstore

# If using Chroma 
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

# If using FAISS
#vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

# total_vectors = vectorstore.index.ntotal
# dimensions = vectorstore.index.d

# print(f"There are {total_vectors} vectors with {dimensions:,} dimensions in the vector store")

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")


# ## Time to use LangChain to bring it all together

# create a new Chat with OpenAI
#llm = ChatOpenAI(temperature=0.7, model_name=MODEL)
llm = ChatOpenAI(
    #openai_api_key="YOUR_API_KEY_HERE",
    model="llama3.2",  # The name of the model on the local server
    base_url="http://localhost:11434/v1"
)
# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
#retriever = vectorstore.as_retriever()
retriever = vectorstore.as_retriever(search_kwargs={"k": 25}) # k is the number of chunks to use. Default chunk size is small (3 or 4)

# putting it together: set up the conversation chain with the GPT 3.5 LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)


# ## Now we will bring this up in Gradio using the Chat interface -
# # A quick and easy way to prototype a chat with an LLM


# Wrapping that in a function

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]


# And in Gradio:

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)





