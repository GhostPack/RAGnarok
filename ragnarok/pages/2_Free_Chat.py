import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from streamlit_cookies_manager import CookieManager
import time
import torch


cookies = CookieManager()
while not cookies.ready():
    time.sleep(1)

if "llm_model" not in cookies:
    st.error("Please select a LLM model on the main settings page.")
    st.stop()

if "llm_temperature" not in cookies:
    st.error("Please select a LLM model temperature on the main settings page.")
    st.stop()

llm_generation_kwargs = {
    "max_tokens": 512,
    "stream": True, 
    "temperature": float(cookies["llm_temperature"]),
    "echo": False
}

# check for GPU presence
if torch.cuda.is_available():
    # traditional Nvidia cuda GPUs
    device = torch.device("cuda:0")
    n_gpu_layers = int(cookies["n_gpu_layers"])
elif torch.backends.mps.is_available():
    # for macOS M1/M2s
    device = torch.device("mps")
    n_gpu_layers = int(cookies["n_gpu_layers"])
else:
    device = torch.device("cpu")
    n_gpu_layers = 0

@st.cache_resource
def get_llm(llm_model_path, n_gpu_layers):
    llm = Llama(
        model_path=llm_model_path,
        n_ctx=8192,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    return llm

try:
    if cookies["llm_model"] == "Intel/neural-chat-7b-v3-3":
        llm_model_path = hf_hub_download("TheBloke/neural-chat-7B-v3-3-GGUF", filename="neural-chat-7b-v3-3.Q5_K_M.gguf", local_files_only=True)
    elif cookies["llm_model"] == "DeepSeek-R1-Distill-Qwen-7B":
        llm_model_path = hf_hub_download("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF", filename="DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf", local_files_only=True)
    elif cookies["llm_model"] == "openchat-3.5-1210":
        llm_model_path = hf_hub_download("TheBloke/openchat-3.5-1210-GGUF", filename="openchat-3.5-1210.Q8_0.gguf", local_files_only=True)
    elif cookies["llm_model"] == "Starling-LM-7B-alpha":
        llm_model_path = hf_hub_download("TheBloke/Starling-LM-7B-alpha-GGUF", filename="starling-lm-7b-alpha.Q5_K_M.gguf", local_files_only=True)
    else:
        llm_model = cookies["llm_model"]
        st.error(f"Invalid llm_model: {llm_model}")
except:
    with st.spinner("Downloading LLM model (this will take some time)..."):
        if cookies["llm_model"] == "Intel/neural-chat-7b-v3-3":
            llm_model_path = hf_hub_download("TheBloke/neural-chat-7B-v3-3-GGUF", filename="neural-chat-7b-v3-3.Q5_K_M.gguf")
        elif cookies["llm_model"] == "DeepSeek-R1-Distill-Qwen-7B":
            llm_model_path = hf_hub_download("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF", filename="DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf")
        elif cookies["llm_model"] == "openchat-3.5-1210":
            llm_model_path = hf_hub_download("TheBloke/openchat-3.5-1210-GGUF", filename="openchat-3.5-1210.Q8_0.gguf")
        elif cookies["llm_model"] == "Starling-LM-7B-alpha":
            llm_model_path = hf_hub_download("TheBloke/Starling-LM-7B-alpha-GGUF", filename="starling-lm-7b-alpha.Q5_K_M.gguf")
        else:
            llm_model = cookies["llm_model"]
            st.error(f"Invalid llm_model_path: {llm_model}")

llm = get_llm(llm_model_path, n_gpu_layers)

st.title("Free Chat With Selected Model")
st.warning('*WARNING: results not guaranteed to be correct!*', icon="⚠️")

if "freeform_messages" not in st.session_state:
    st.session_state.freeform_messages = []

for message in st.session_state.freeform_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("<enter a question>"):
    st.session_state.freeform_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if "neural-chat" in cookies["llm_model"]:
            single_turn_prompt = f"### System:\nYou are a helpful assistant chatbot.\n### User:\n{prompt}\n### Assistant:\n"
        else:
            single_turn_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"

        with st.spinner("LLM is processing the prompt..."):
            start = time.time()
            stream = llm.create_completion(single_turn_prompt, **llm_generation_kwargs)
            for output in stream:
                full_response += (output['choices'][0]['text'] or "").split("### Assistant:\n")[-1]
                message_placeholder.markdown(full_response + "▌")

            end = time.time()
            print(f"LLM generation completed in {(end - start):.2f} seconds")

    st.session_state.freeform_messages.append({"role": "assistant", "content": f"{full_response}"})
