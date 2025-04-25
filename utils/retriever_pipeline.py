import streamlit as st
from utils.build_graph import retrieve_from_graph
from langchain_core.documents import Document
import requests
import os
from dotenv import load_dotenv
import openai

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def expand_query(query, uri=None, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": f"Giả sử bạn đã có câu trả lời cho câu hỏi sau. Hãy tạo câu trả lời hợp lý nhất có thể để dùng cho tìm kiếm thông tin:\n\n{query}"}],
            temperature=0.5
        )
        hypothetical_answer = response.choices[0].message["content"]
        return f"{query}\n{hypothetical_answer}"
    except Exception as e:
        st.error(f"❌ HyDE failed: {str(e)}")
        return query




def retrieve_documents(query, uri, model, chat_history=""):
    
    expanded_query = expand_query(f"{chat_history}\n{query}", uri, model) 

    if st.session_state.enable_hyde:
        expanded_query = expand_query(f"{chat_history}\n{query}", uri, model)
        st.session_state.hyde_query = expanded_query
    else:
        expanded_query = query
        st.session_state.hyde_query = None

    docs = st.session_state.retrieval_pipeline["ensemble"].invoke(expanded_query)

    if st.session_state.enable_graph_rag:
        G = st.session_state.retrieval_pipeline.get("knowledge_graph")
        graph_results = retrieve_from_graph(query, G)

        st.write(f"🧠 GraphRAG trả về các nút liên quan: {graph_results}")

        graph_docs = [Document(page_content=node) for node in graph_results]

        if graph_docs:
            docs = graph_docs + docs


    if st.session_state.enable_reranking:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = st.session_state.retrieval_pipeline["reranker"].predict(pairs)


        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    else:
        ranked_docs = docs

    return ranked_docs[:st.session_state.max_contexts]
