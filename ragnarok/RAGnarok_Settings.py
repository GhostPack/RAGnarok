import requests
from requests.auth import HTTPBasicAuth
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
from streamlit_cookies_manager import CookieManager
import torch


cookies = CookieManager()
while not cookies.ready():
    time.sleep(1)

columns = st.columns(2)
with columns[0]:
    st.image("./img/ragnarok.png")
st.markdown("#### *[RAGnarok](https://github.com/GhostPack/RAGnarok) is a Retrieval-Augmented Generation LLM ChatBot powered by a Nemesis instance.*")
st.info("**NOTE**: RAGnarok does not have conversational memory, each question is independent.")
st.divider()


def wait_for_nemesis(nemesis_url, nemesis_user, nemesis_password, wait_timeout = 3):
    retries = 3
    success = False
    while retries > 0:
        try:
            if nemesis_url.startswith("https://"):
                result = requests.get(
                    f"{nemesis_url}nlp/ready",
                    auth=HTTPBasicAuth(nemesis_user, nemesis_password),
                    timeout=wait_timeout,
                    verify=False
                )
            else:
                result = requests.get(
                    f"{nemesis_url}nlp/ready",
                    auth=HTTPBasicAuth(nemesis_user, nemesis_password),
                    timeout=wait_timeout
                )
            if result.status_code == 401:
                st.error(f"Invalid Nemesis credentials!")
            elif result.status_code != 200:
                st.warning(f"Error connecting to Nemesis instance {nemesis_url}: '{result.status_code}'")
            else:
                success = True
                retries = 0
                st.success("Successfully connected to Nemesis instance!")
            break
        except Exception as e:
            st.warning(f"Encountered an exception while trying to connect to Nemesis instance {nemesis_url}: '{e}', trying again in {wait_timeout} seconds...")
            time.sleep(wait_timeout)
            retries = retries - 1
            continue
    if not success:
        st.error(f"Error connecting to {nemesis_url}, please check your connection/credentials!")

default_value = cookies["nemesis_url"] if "nemesis_url" in cookies else ""
nemesis_url = st.text_input(
    label="Nemesis URL",
    help="The Nemesis endpoint",
    value=default_value
)
cols = st.columns(2)
with cols[0]:
    default_value = cookies["nemesis_username"] if "nemesis_username" in cookies else ""
    nemesis_username = st.text_input(
        label="Nemesis Username",
        help="This is the `basic_auth_user` value in nemesis.config.",
        value=default_value
    )
with cols[1]:
    default_value = cookies["nemesis_password"] if "nemesis_password" in cookies else ""
    nemesis_password = st.text_input(
        label="Nemesis Password",
        help="This is the `basic_auth_password` value in nemesis.config.",
        value=default_value
    )
if nemesis_url and nemesis_username and nemesis_password:
    if not nemesis_url.endswith("/"):
        nemesis_url = f"{nemesis_url}/"
    wait_for_nemesis(nemesis_url, nemesis_username, nemesis_password)
    cookies["nemesis_url"] = nemesis_url
    cookies["nemesis_username"] = nemesis_username
    cookies["nemesis_password"] = nemesis_password
    st.divider()


cols = st.columns(2)
with cols[0]:
    default_index = 0
    if "llm_model" in cookies:
        if "neural-chat" in cookies["llm_model"].lower():
            default_index = 0
        elif "deepseek" in cookies["llm_model"].lower():
            default_index = 1   
        elif "openchat" in cookies["llm_model"].lower():
            default_index = 2
        elif "starling" in cookies["llm_model"].lower():
            default_index = 2
    llm_model = st.selectbox(
        label='LLM model to use',
        options=('Intel/neural-chat-7b-v3-3', 'DeepSeek-R1-Distill-Qwen-7B', 'openchat-3.5-1210', 'Starling-LM-7B-alpha'),
        help="The core LLM to use for chat over retrieved document snippets.",
        index=default_index
    )
    cookies["llm_model"] = llm_model

with cols[1]:
    llm_temperature_default_value = float(cookies["llm_temperature"]) if "llm_temperature" in cookies else 0.1
    llm_temperature = st.slider(
        label="LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=llm_temperature_default_value,
        help="The temperate for the core LLM. Higher means more 'creative', lower means more repeatable."
    )
    cookies["llm_temperature"] = llm_temperature

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

cols = st.columns(2)
with cols[0]:
    if device == "cuda":
        st.success("Using CUDA GPU!")
    elif device == "mps":
        st.success("Using MPS GPU!")
    else:
        st.warning("No GPU detected, generation may be slow!")
with cols[1]:
    if device == "cuda" or device == "mps":
        default_n_gpu_layers = int(cookies["n_gpu_layers"]) if "n_gpu_layers" in cookies else 4
        n_gpu_layers = st.slider(
            label="Number of GPU layers to offload to the GPU", 
            min_value=1,
            max_value=32,
            value=default_n_gpu_layers,
            help="Number of GPU layers to offload to the GPU. More _usually_ means faster generation, but may cause out-of-memory errors."
        )
        cookies["n_gpu_layers"] = n_gpu_layers

cols = st.columns(2)
with cols[0]:
    default_index = 0
    if "embedding_model" in cookies:
        if cookies["embedding_model"] == 'TaylorAI/gte-tiny':
            default_index = 0
        elif cookies["embedding_model"] == 'Harmj0y/nemesis-gte-tiny':
            default_index = 1
        elif cookies["embedding_model"] == 'TaylorAI/bge-micro-v2':
            default_index = 2
        elif cookies["embedding_model"] == 'thenlper/gte-small':
            default_index = 3
    embedding_model = st.selectbox(
        label='Embedding model to use',
        options=('TaylorAI/gte-tiny', 'Harmj0y/nemesis-gte-tiny', 'TaylorAI/bge-micro-v2', 'thenlper/gte-small'),
        help="Embedding model to use. TaylorAI/gte-tiny is the Nemesis Default.",
        index=default_index
    )
    cookies["embedding_model"] = embedding_model

with cols[1]:
    k_similarity_default_value = int(cookies["k_similarity"]) if "k_similarity" in cookies else 30
    k_similarity = st.slider(
        label="Initial K search",
        min_value=1,
        max_value=100,
        value=k_similarity_default_value,
        help="The number of similar indexed documents to pull from the Nemesis backend before performing reranking. More documents casts a wide net but takes more time."
    )
    cookies["k_similarity"] = k_similarity

cols = st.columns(3)
with cols[0]:
    default_index = 0
    if "reranking_model" in cookies:
        if cookies["reranking_model"] == "Harmj0y/nemesis-reranker":
            default_index = 0
        elif cookies["reranking_model"] == "BAAI/bge-reranker-base":
            default_index = 1
    reranking_model = st.selectbox(
        label='Reranking model to use',
        options=('nemesis-reranker', 'bge-reranker-base',),
        help="Model to use to rerank results before sending to the LLM.",
        index=default_index
    )
    if reranking_model == "nemesis-reranker":
        reranking_model = "Harmj0y/nemesis-reranker"
    elif reranking_model == "bge-reranker-base":
        reranking_model = "BAAI/bge-reranker-base"
    cookies["reranking_model"] = reranking_model
with cols[1]:
    min_doc_results_default_value = int(cookies["min_doc_results"]) if "min_doc_results" in cookies else 1
    min_doc_results = st.slider(
        label="Minimum number of documents to supply to the LLM", 
        min_value=1,
        max_value=15,
        value=min_doc_results_default_value,
        help="The minimum number of document results to feed to the LLM's context. More means slower response generation, higher provides more context (but possibly more irrelevant information)."
    )
    cookies["min_doc_results"] = min_doc_results
with cols[2]:
    max_doc_results_default_value = int(cookies["max_doc_results"]) if "max_doc_results" in cookies else 5
    max_doc_results = st.slider(
        label="Maximum number of documents to supply to the LLM", 
        min_value=1,
        max_value=15,
        value=max_doc_results_default_value,
        help="The maximum number of document results to feed to the LLM's context. More means slower response generation, higher provides more context (but possibly more irrelevant information)."
    )
    cookies["max_doc_results"] = max_doc_results

st.divider()

if st.button("Clear Cookies?"):
    cookies.clear()

st.divider()

cookies.save()