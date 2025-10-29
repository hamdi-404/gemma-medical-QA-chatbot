import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Load model and tokenizer ---
@st.cache_resource
def load_model():
    model_path = "gemma_medical_merged"  # your merged folder
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return model, tokenizer

model, tokenizer = load_model()

# --- App title ---
st.title("🩺 Medical QA Chatbot by Eng: Hamdi.")
st.caption("Ask any medical question (powered by fine-tuned Gemma model).")

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat history ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# --- Chat input ---
prompt = st.chat_input("Ask a medical question...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate model response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            input_text = f"Question: {prompt}\nAnswer:"
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            answer = tokenizer.decode(output[0], skip_special_tokens=True)
            # Clean up
            answer = answer.split("Answer:")[-1].strip()
            st.markdown(answer)

    # Add model response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
