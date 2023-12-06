# Code refactored from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
import json
import openai
from streamlit_chat import message
import streamlit as st
from langchain.vectorstores import FAISS
import openai, os
from langchain.embeddings import OpenAIEmbeddings
from Agents import *
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings
from Agents import Get_response
from streamlit_chat import message

openai.api_key = "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" ## To configure OpenAI API
os.environ["OPENAI_API_KEY"] = "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" ## To configure langchain con

llm = ChatOpenAI(temperature=0.3,streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()], model="gpt-3.5-turbo-16k")

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")


#khởi tạo lịch sử chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[{'role':"system", 'content':'You are a expert in RUST programming'}]
    
# nếu bấm reset chat thì sẽ xóa lịch sử
if st.sidebar.button("Reset Chat"):
    # Save the chat history to the DataFrame before clearing it
    path_history='chat_history.txt'
    with open(path_history, "w") as json_file:
        json.dump(str(st.session_state.chat_history), json_file)
    # Clear the chat messages and reset the full response
    st.session_state.chat_history=[st.session_state.get("chat_history",[])[0]]
    st.session_state.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,
            return_messages=True,
            output_key="output"
        )
    
    
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=10,
            return_messages=True,
            output_key="output"
        )
else:
    memory=st.session_state.get("memory")
        
# tạo cột bên trái chứa tên demo

with st.sidebar:
    st.title(':red[POLKADOT-GPT] Demo  :robot_face:')

# in ra lịch sử chat
a=0
history=st.session_state.get("chat_history",[])
if len(history)>1:
    for msg in history[1:]:
        if a%2==0:
                message(msg["content"], is_user=True,key="user"+str(a))
        else:
                message(msg["content"], is_user=False,key="ai"+str(a))
        a+=1

# load embedding
if 'load_rust' not in st.session_state:
    rust=  FAISS.load_local("HuggingFace_embedding_RUSTGPT", embedding)
    st.session_state.load=rust
else:
    rust=st.session_state.get('load_rust')
    
if 'load_polkadot' not in st.session_state:
    polkadot=  FAISS.load_local("HuggingFace_embedding_Polkadot", embedding)
    st.session_state.load=polkadot
else:
    polkadot=st.session_state.get('load_polkadot')
    
# nhập câu hỏi và tạo câu trả lời
if prompt := st.chat_input("Enter your question"):
    print("==========QUESTION=============")
    print(prompt,"\n================================")
    message(prompt, is_user=True)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    final_answer=Get_response(prompt,llm,rust,polkadot,memory).output()
    message(final_answer, is_user=False)
    
    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})