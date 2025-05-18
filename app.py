import streamlit as st
import pickle
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ChatMessageHistory
from prompt import Prompt 
from chatbot import chatbot

st.title("Chatbot")

with open("documents_saved.pkl", "rb") as f:
    doc = pickle.load(f)

import openai
from openai import OpenAI
import os

os.environ["GEMINI_API_KEY"] = "AIzaSyDeqhwridj7KC0raINvs04-TvjIGla5d10"
gemini_api_key = os.getenv("GEMINI_API_KEY")


link = "/content/documents_saved.pkl"
link_store = "/tmp/ai_qdrant111_new"
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="query",
    return_messages=True
  )
model_name = "gemini-2.0-flash"

@st.cache_resource
def get_embedding():
  return GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=os.getenv("GEMINI_API_KEY")
)
embeddings = get_embedding()

@st.cache_resource
def get_prompt(api_key, model_name):
    return Prompt(api_key, model_name)

prompt = get_prompt(gemini_api_key, model_name)

@st.cache_resource
def get_chatbot(link, _embeddings, link_store):
    return chatbot(link, _embeddings, link_store)  

bot = get_chatbot(link, embeddings, link_store)

chatbot = get_chatbot(link, embeddings, link_store)

@st.cache_resource
def get_llm_teach_exercises(_memory):
  return prompt.teach_exercises(_memory)

@st.cache_resource
def get_llm_teach_detail(_memory):
  return prompt.teach_detail(_memory)
  
@st.cache_resource
def get_llm_teach_history(_memory):
  return prompt.teach_history(_memory)

@st.cache_resource
def get_llm_do(_memory):
  return prompt.do_exercises(_memory)



llm_teach_exercises = get_llm_teach_exercises(memory)
llm_teach_detail = get_llm_teach_detail(memory)
llm_teach_history = get_llm_teach_history(memory)
llm_do = get_llm_do(memory)


def QA (query, llm, k):
    llm_chain = llm
    history_messages = llm_chain.memory.chat_memory.messages if hasattr(llm_chain.memory.chat_memory, 'messages') else []
    history_str = "\n".join([f"{his.type}: {his.content}" for his in history_messages])

    result_docs = bot.retriever_similar(query, k)
    docs_format = bot.format_context(result_docs)

    response = None

    try:
      response = llm_chain.invoke({"query": query, "context": docs_format})
      if (len(history_messages) > 4 ):
          summary_his = prompt.summary_history(response['text'])
          llm_chain.memory.chat_memory.add_user_message(query)
          llm_chain.memory.chat_memory.add_ai_message(summary_his)

    except Exception as e:
      print("Error:", e)
    return response['text']

def teacher(query):
    check_prompt = prompt.determine_query(query)
    if check_prompt  == '1':
      llm = llm_teach_exercises
    elif check_prompt  == '3':
      llm = llm_teach_detail
    elif check_prompt  == '2':
      llm = llm_do
    else:
      llm = llm_teach_history
    
    return QA(query, llm, 10)


# query = st.text_input("Nhập câu hỏi:", value="")

st.markdown("""
    <style>
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background-color: #111;
    }
    .user-message {
        
        margin-bottom: 5px;
        color: #ffffff;
        padding: 10px;
        border-radius: 19px;
        border: solid 0.1px;
        float: right;
        clear: both;
        background-color: #6e6a6c;
        display: inline-block;
        margin-top: 35px;
    }
    .system-message {
        text-align: left;
        color: #ffffff;    
        padding: 19px;
        text-align: left;
        border-radius: 19px;
        border: solid 0.1px;
    }
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 55%;
        background-color: #111;
        padding: 10px;
    }
    <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    </style>
""", unsafe_allow_html=True)





if 'history' not in st.session_state:
  st.session_state['history'] = []


with st.container():
    for sender, message in (st.session_state['history'][:-2] if len(st.session_state['history']) > 2 else []):
      if isinstance(sender, str) and isinstance(message, str):
          if sender == 'Bạn':
              st.markdown(f'<div class="user-message">{message}</div>', unsafe_allow_html=True)
          else:
              st.markdown(f'<div class="system-message">{message}</div>', unsafe_allow_html=True)
      else:
          st.write("Lỗi cấu trúc dữ liệu trong lịch sử trò chuyện.")

query = st.chat_input("")
if query:
  answer = teacher(query)
  
  st.session_state['history'].append(("Bạn", query))
  st.markdown(f'<div class="user-message"> {query}</div>', unsafe_allow_html=True)

  st.session_state['history'].append(("AI", answer))
  st.markdown(f'<div class="system-message"> {answer}</div>', unsafe_allow_html=True)






