import streamlit as st
from utils.retriever_pipeline import retrieve_documents
from utils.doc_handler import process_documents
from sentence_transformers import CrossEncoder
import openai
import torch
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

def translate_to_english(text):
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": f"Translate this to English for document understanding:\n{text}"}],
        temperature=0,
    )
    return response.choices[0].message["content"]

def translate_to_vietnamese(text):
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": f"Translate this to Vietnamese:\n{text}"}],
        temperature=0,
    )
    return response.choices[0].message["content"]

def route_query(query):
    query_lower = query.lower()

    if any(keyword in query_lower for keyword in [
        "xin chào", "chào", "hi", "hello",
        "giờ giấc", "thời gian", "mấy giờ",
        "thời tiết", "nắng", "mưa",
    ]):
        return "chatgpt"
    return "default"

device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = None
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
except Exception as e:
    st.error(f"❌ Không thể tải CrossEncoder: {str(e)}")

st.set_page_config(page_title="NHOM07 RAG-Pro", layout="wide")
st.markdown("""<style>.stApp { background-color: #f4f4f9; }</style>""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files and not st.session_state.documents_loaded:
        with st.spinner("Processing documents..."):
            process_documents(uploaded_files, reranker, openai.api_key)
            st.success("✅ Documents processed!")

    st.markdown("---")
    st.header("⚙️ RAG Settings")
    st.session_state.rag_enabled = st.checkbox("RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("HyBird Search + Neural Reranking", value=True)
    st.session_state.enable_graph_rag = st.checkbox("GraphRAG", value=True)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 Thị Trường Lao Động - by Manhtuong")
st.caption("Powered by OpenAI API")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Đặt câu hỏi ở đây..."):
    route = route_query(prompt)
    translated_prompt = translate_to_english(prompt)
    chat_history = "\n".join([msg["content"] for msg in st.session_state.messages[-5:]])
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        if route == "chatgpt":
            messages = [
                {"role": "system", "content": "You are a friendly assistant that answers casual greetings, time-related questions, weather queries, or provides career advice."},
                {"role": "user", "content": translated_prompt}
            ]
        else: 
            context = ""
            if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
                try:
                    docs = retrieve_documents(translated_prompt, None, "gpt-3.5-turbo", chat_history)
                    if st.session_state.enable_hyde and st.session_state.hyde_query:
                        with st.expander("🔍 Câu hỏi đã mở rộng bằng HyDE"):
                            st.markdown(f"```\n{translate_to_vietnamese(st.session_state.hyde_query)}\n```")
                    context = "\n".join(f"[Source {i+1}]: {doc.page_content}" for i, doc in enumerate(docs))
                except Exception as e:
                    st.error(f"❌ Retrieval Error: {str(e)}")

            messages = [
                {"role": "system", "content": f"You are a helpful assistant that answers questions based on document context.\nContext:\n{context}"},
                {"role": "user", "content": f"{chat_history}\n{translated_prompt}"}
            ]

        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=st.session_state.temperature,
                stream=True,
            )

            for chunk in response:
                token = chunk['choices'][0]['delta'].get('content', '')
                full_response += token
                response_placeholder.markdown(full_response + " ")

            translated_response = translate_to_vietnamese(full_response)
            response_placeholder.markdown(translated_response)
            st.session_state.messages.append({"role": "assistant", "content": translated_response})

        except Exception as e:
            st.error(f"❌ OpenAI error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "⚠️ Đã xảy ra lỗi khi tạo phản hồi."})