import time
import ntpath
import os.path
import requests
import json
from requests.auth import HTTPBasicAuth

from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llama_cpp import Llama
import numpy as np
import streamlit as st
from streamlit_cookies_manager import CookieManager
import torch
import asyncio
from ollama import AsyncClient



cookies = CookieManager()
while not cookies.ready():
    time.sleep(1)

if "mode" not in cookies:
    st.error("Your cookies are broken, please go back to the main settings page.")
    st.stop()

if "mode_index" not in st.session_state:
    st.session_state["mode_index"] = 0  # Default to the first mode
def get_mode_index(cookies):
    if "mode" in cookies:
        if cookies["mode"] == "Local LLM":
            return 0
        elif cookies["mode"] == "Remote Ollama Server (Local Reranker)":
            return 1
        elif cookies["mode"] == "Remote Ollama Server (No Reranker)":
            return 2
    return 0  # Default to 0 if no mode is set

mode_index = get_mode_index(cookies)

if mode_index == 0:
    if "llm_model" not in cookies and "ollama_model" not in cookies:
        st.error("Please select a LLM model on the main settings page.")
        st.stop()
    if "llm_temperature" not in cookies:
        st.error("Please select a LLM model temperature on the main settings page.")
        st.stop()

if mode_index == 0 or mode_index == 1:
    if "reranking_model" not in cookies:
        st.error("Please select an reranking model on the main settings page.")
        st.stop()
if mode_index == 1 or mode_index == 2:
    if "ollama_url" not in cookies:
        st.error("Please select an Ollama server on the main settings page.")
        st.stop()
    if "ollama_model" not in cookies:
        st.error("Please select an Ollama model on the main settings page.")
        st.stop()
    if cookies["ollama_model"] == "":
        st.error("Please select an Ollama model on the main settings page.")
        st.stop()

if "nemesis_url" not in cookies or "nemesis_username" not in cookies or "nemesis_password" not in cookies:
    st.error("Please enter Nemesis connection info on the main settings page.")
    st.stop()

if "k_similarity" not in cookies:
    st.error("Please select an k_similarity on the main settings page.")
    st.stop()

if "min_doc_results" not in cookies:
    st.error("Please select an min_doc_results on the main settings page.")
    st.stop()

if "max_doc_results" not in cookies:
    st.error("Please select an max_doc_results on the main settings page.")
    st.stop()




########################################################
#
# Model Delarations
#
########################################################
n_gpu_layers = 0
temp = ""
min_doc_results = int(cookies['min_doc_results'])
max_doc_results = int(cookies["max_doc_results"])
min_similarity_score = 0

@st.cache_resource
def get_llm(llm_model_path, n_gpu_layers):
    llm = Llama(
        model_path=llm_model_path,
        n_ctx=8192,
        n_gpu_layers=n_gpu_layers,
        verbose=False
    )
    return llm

@st.cache_resource
def get_ollama_client(ollama_url):
    client = AsyncClient(ollama_url)
    return client

# either local LLM or remote Ollama server with local reranker
if mode_index == 0 or mode_index == 1:
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
    def get_reranker(reranking_model, device):
        rerank_tokenizer = AutoTokenizer.from_pretrained(reranking_model)
        print(f"device: {device}")
        rerank_model = AutoModelForSequenceClassification.from_pretrained(reranking_model).to(device)
        return (rerank_tokenizer, rerank_model)
    temp = cookies["reranking_model"]
    with st.spinner(f"Downloading/loading reranking model {temp} ..."):
        (rerank_tokenizer, rerank_model) = get_reranker(cookies["reranking_model"], device)

# only local LLM
if mode_index == 0:
    llm_generation_kwargs = {
        "max_tokens": 512,
        "stream": True, 
        "temperature": float(cookies["llm_temperature"]),
        "echo": False
    }
    try:
        if cookies["llm_model"] == "Intel/neural-chat-7b-v3-3":
            llm_model_path = hf_hub_download("TheBloke/neural-chat-7B-v3-3-GGUF", filename="neural-chat-7b-v3-3.Q5_K_M.gguf", local_files_only=True)
        elif cookies["llm_model"] == "openchat-3.5-0106":
            llm_model_path = hf_hub_download("TheBloke/openchat-3.5-0106-GGUF", filename="openchat-3.5-0106.Q5_K_M.gguf", local_files_only=True)
        elif cookies["llm_model"] == "Starling-LM-7B-alpha":
            llm_model_path = hf_hub_download("TheBloke/Starling-LM-7B-alpha-GGUF", filename="starling-lm-7b-alpha.Q5_K_M.gguf", local_files_only=True)
        else:
            llm_model = cookies["llm_model"]
            st.error(f"Invalid llm_model: {llm_model}")
    except:
        with st.spinner("Downloading LLM model (this will take some time)..."):
            if cookies["llm_model"] == "Intel/neural-chat-7b-v3-3":
                llm_model_path = hf_hub_download("TheBloke/neural-chat-7B-v3-3-GGUF", filename="neural-chat-7b-v3-3.Q5_K_M.gguf")
            elif cookies["llm_model"] == "openchat-3.5-0106":
                llm_model_path = hf_hub_download("TheBloke/openchat-3.5-0106-GGUF", filename="openchat-3.5-0106.Q5_K_M.gguf")
            elif cookies["llm_model"] == "Starling-LM-7B-alpha":
                llm_model_path = hf_hub_download("TheBloke/Starling-LM-7B-alpha-GGUF", filename="starling-lm-7b-alpha.Q5_K_M.gguf")
            else:
                llm_model = cookies["llm_model"]
                st.error(f"Invalid llm_model_path: {llm_model}")

    llm = get_llm(llm_model_path, n_gpu_layers)


with st.sidebar:
    if mode_index == 0 or mode_index == 1:
        if torch.cuda.is_available():
            st.success("Using CUDA GPU!")
        elif torch.backends.mps.is_available():
            st.success("Using MPS GPU!")
        else:
            st.warning("No GPU detected, generation may be slow!")
    file_path_include = st.text_input("Enter file name/path pattern to include in the initial search:")
    file_path_exclude = st.text_input("Enter file name/path pattern to exclude from the initial search:")
    st.write("**Note**: _wildcard == \\*, use | to separate multiple terms, e.g. C:\\Temp\\\\*|\\*.pdf_")
    st.write("_You MUST surround any file name with \\*'s, i.e., \*A_Process_is_No_One.pdf\*_")

st.title("RAGnarok Chat")
st.warning('*WARNING: results not guaranteed to be correct! Verify answers with the supplied sources.*', icon="⚠️")

if "private_messages" not in st.session_state:
    st.session_state.private_messages = []

for message in st.session_state.private_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("<enter a question>"):
    st.session_state.private_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        results = []

        try:
            # data = {
            #     "search_phrase": prompt,
            #     "num_results": cookies['k_similarity'],
            # }
            
            # if file_path_include:
            #     data["file_path_include"] = file_path_include
            # if file_path_exclude:
            #     data["file_path_exclude"] = file_path_exclude

            # TODO: change this to query the local llm or remote Ollama server to get a list of search terms, then use those terms to query Nemesis
            
            client = get_ollama_client(cookies["ollama_url"])
            single_turn_nlp_prompt = f"### System:\nYou are a helpful LLM who knows how to distil user queries down to an individual search term. The user is going to ask you a question, your job is to determine what the best search term is based on the users question. Only respond with the search term, nothing else. Do not preface the response with classifiers like 'search string' or 'search_term', only output the term itself. \n### User:\n{prompt}\n### Assistant:\n"
            async def nlp_term_chat():
                nlp_search_term_full_response = ""
                message = {'role': 'user', 'content': single_turn_nlp_prompt}
                async for part in await AsyncClient().chat(model=cookies['ollama_model'], messages=[message], stream=True):
                    nlp_search_term_full_response += part['message']['content']
                return nlp_search_term_full_response
            nlp_search_term = asyncio.run(nlp_term_chat())


            url = "{}/hasura/v1/graphql".format(cookies["nemesis_url"])
            payload = json.dumps({
            "query": "\n            query SearchDocuments($searchQuery: String!, $pathPattern: String!, $agentPattern: String!, $project: String, $startDate: timestamptz, $endDate: timestamptz) {\n              search_documents(\n                args: {\n                  search_query: $searchQuery,\n                  path_pattern: $pathPattern,\n                  agent_pattern: $agentPattern,\n                  project_name: $project,\n                  start_date: $startDate,\n                  end_date: $endDate,\n                  max_results: 100\n                }\n              ) {\n                object_id\n                chunk_number\n                content\n                file_name\n                path\n                extension\n                project\n                agent_id\n                timestamp\n              }\n            }\n          ",
            "variables": {
                "searchQuery": nlp_search_term,
                "pathPattern": "%",
                "agentPattern": "%",
                "project": None,
                "startDate": None,
                "endDate": None
            }
            })
            # TODO: add the X-Hasura-Admin-Secret to the settings page and use that instead of hardcoding it
            headers = {
            'Content-Type': 'application/json',
            'X-Hasura-Admin-Secret': 'pass456',
            }

            response = requests.request("POST", url, headers=headers, data=payload, verify=False)
            if response.status_code != 200:
                st.error(f"Error calling Nemesis GraphQL API: {response.status_code} - {response.text}")
                st.stop()
            else:
                results = response.json()
                if len(results["data"]["search_documents"]) == 0:
                    st.error("No documents found matching the search criteria.")
                    st.stop()
                else:
                    results = results["data"]["search_documents"]
                    print(f"Found {len(results)} documents matching the search criteria.")



            # url = f'{cookies["nemesis_url"]}nlp/hybrid_search'
            # with st.spinner("Searching for initial documents from Nemesis..."):
            #     if url.startswith("https://"):
            #         response = requests.get(
            #             url,
            #             json=data,
            #             auth=HTTPBasicAuth(cookies["nemesis_username"], cookies["nemesis_password"]),
            #             verify=False
            #         )
            #     else:
            #         response = requests.get(
            #             url,
            #             json=data,
            #             auth=HTTPBasicAuth(cookies["nemesis_username"], cookies["nemesis_password"])
            #         )
            #     if response.status_code == 200:
            #         results = response.json()
            #         if "error" in results:
            #             if results["error"] == "index_not_found_exception":
            #                 st.error(f"No documents have been indexed!")
            #             else:
            #                 error = results["error"]
            #                 st.error(f"Error calling text search {url} with search_phrase '{prompt}' : {error}")
            #             st.stop()
            #     else:
            #         st.error(f"Error calling text search {url} with search_phrase '{prompt}' : {response.status_code}")
            #         st.stop()
        
        except Exception as e:
            st.error(f"Error calling text search {url} with search_phrase '{prompt}' : {e}")
            st.stop()

        documents = results
        if mode_index == 0 or mode_index == 1:
            # Reranking process
            pairs = []
            for document in documents:
                pairs += [[prompt, document["text"].replace('"""', '"').replace("'''", "'")]]

            start = time.time()
            inputs = rerank_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            print(inputs)
            with st.spinner("Reranking documents..."):
                rerank_scores = (rerank_model(**inputs, return_dict=True).logits.view(-1,).float()).tolist()
                end = time.time()
                print(f"Reranker evaluated in {(end - start):.2f} seconds")

        final_results = []

        for i in range(len(documents)):
            document = documents[i]
            rerank_score = rerank_scores[i] if (mode_index == 0  or mode_index == 1) else 0.1
            originating_file_name = ntpath.basename(document["path"])
            originating_object_id = document["object_id"]
            nemesis_url = cookies["nemesis_url"]
            # link to the file in Nemesis
            file_page_link = f"{nemesis_url}files/{originating_object_id}"
            # snippet_id = document["id"]
            # direct link to the snippet
            # snippet_link = f"{nemesis_url}/kibana/app/discover#/?_a=(filters:!((query:(match_phrase:(_id:'{snippet_id}')))),index:'45fbaacb-9ef9-4cd1-b837-bc8ab0448220')&_g=(time:(from:now-1y%2Fd,to:now))"
            
            final_results += [
                [
                    rerank_score,
                    originating_file_name,
                    file_page_link,
                    # snippet_link,
                    document["content"].replace('"""', '"').replace("'''", "'")
                ]
            ]
    
        # ensure results are sorted with the highest similiary score first
        final_results.sort(key=lambda x: x[0], reverse=True)

        # start at the minimum number of documents required
        i_range = min_doc_results if min_doc_results < len(final_results) else len(final_results)
        sources = final_results[0:i_range]

        responses_generated = False

        # add in results over the minimum similiary score
        for result in final_results[i_range:]:
            if result[0] > min_similarity_score:
                sources.append(result)

        # finally, cap the final results so we don't go over the LLM context
        #   default == 15 (15*510tokens == 7650, leaving room for prompt overhead)
        sources = sources[:max_doc_results]

        print(f"\nNumber of source snippets: {len(sources)}\n")

        sources_formatted = []
        for i in range(len(sources)):
            source = sources[i]
            sources_formatted += [f"{i+1}. [{source[1]}]({source[2]}) (score: {source[0]:.2f})"]
        sources_formatted_final = "\n".join(sources_formatted)

        #######################################################
        #
        # LLM prompting
        #
        #######################################################

        template = """
You are a helpful LLM who knows how to reason over source text blocks. Generate a coherent and informative response based on the following source blocks.

Each Source Block starts with the Source Block number, followed by a Similarity Score reflecting the block's relevance to the overall prompt, followed by the originating Filename, finally followed by the source Text itself.

The similarity scores represent the model's confidence in the relevance of each source block. Higher scores indicate higher perceived similarity. Utilize the information in all source blocks to enhance your answer, but if any source blocks contain contradictory information use the information from the source block with the higher Similarity Score.

Only answer questions using the source below and if you're not sure of an answer, you can say "I don't know based on the retrieved sources".
"""

        for i in range(len(sources)):
            # final_result_score, originating_file_name, file_page_link, snippet_link, final_result_text = sources[i]
            final_result_score, originating_file_name, file_page_link, final_result_text = sources[i]

            template += f"""
Source Block: {i+1}
Similarity Score: {final_result_score}
Filename: {originating_file_name}
Text:
\"\"\"
{final_result_text}
\"\"\"

---
"""     

        template += f"""
Question: {prompt}
"""

        if "neural-chat" in cookies["llm_model"]:
            single_turn_prompt = f"### System:\nYou are a helpful LLM who knows how to reason over source text blocks.\n### User:\n{template}\n### Assistant:\n"
        else:
            single_turn_prompt = f"GPT4 Correct User: {template}<|end_of_turn|>GPT4 Correct Assistant:"

        with st.spinner("LLM is processing the prompt..."):
            start = time.time()
            if mode_index == 0:
                stream = llm.create_completion(single_turn_prompt, **llm_generation_kwargs)
                for output in stream:
                    full_response += (output['choices'][0]['text'] or "").split("### Assistant:\n")[-1]
                    message_placeholder.markdown(full_response + "▌")
            else:
                client = get_ollama_client(cookies["ollama_url"])
                async def chat():
                    full_response = ""
                    message = {'role': 'user', 'content': single_turn_prompt}
                    async for part in await AsyncClient().chat(model=cookies['ollama_model'], messages=[message], stream=True):
                        print(part['message']['content'], end='', flush=True)
                        full_response += part['message']['content']
                    return full_response
                full_response = asyncio.run(chat())

            
            end = time.time()
  
            message_placeholder.markdown(f"{full_response}\n\n*Sources:*\n\nSearch Term: {nlp_search_term}\n\n{sources_formatted_final}\n\n_Generation time: {(end - start):.2f} seconds_\n")
            
            print(f"LLM generation completed in {(end - start):.2f} seconds")

        st.session_state.private_messages.append({"role": "assistant", "content": f"{full_response}\n\n*Sources:*\n{sources_formatted_final}\n\n_Generation time: {(end - start):.2f} seconds_\n"})
