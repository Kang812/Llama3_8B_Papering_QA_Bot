import streamlit as st
import random
import os
import time
from transformers import TextStreamer
from unsloth import FastLanguageModel

st.title("Llama3 Papering QA Bot")

@st.cache_resource()
def load_model():
    max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "/workspace/papering_qa/work_dir2/checkpoint-500",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map = "auto",)
    
    FastLanguageModel.for_inference(model)
    return model, tokenizer
    
def response_generator(prompt):
    model, tokenizer = load_model()
    q = f"### 질문: {prompt}\n\n### 답변:"
    gened = model.generate(
        **tokenizer(
            q, 
            return_tensors='pt', 
            return_token_type_ids=False).to('cuda'), 
        max_new_tokens=512,
        early_stopping=True,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id)
    
    answer = tokenizer.decode(gened[0])
    
    start_index = answer.find("답변")
    end_index = answer.find("<|end_of_text|>")
    answer = answer[start_index:end_index].replace("답변:", "").strip()
    
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message["content"])

if prompt := st.chat_input("메시지를 입력해주세요."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})