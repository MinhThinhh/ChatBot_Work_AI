import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from utils.build_graph import build_knowledge_graph
from rank_bm25 import BM25Okapi
import os
import re
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import nltk


try:
    nltk.download('punkt', quiet=True)
except:
    pass

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
    return text.strip()


def process_document_batch(file_batch):
    documents = []
    for file in file_batch:
        try:
            st.info(f"Đang xử lý file: {file.name}")
            file_path = os.path.join("temp", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            if file.name.endswith(".pdf"):
                st.write("Đang đọc file PDF...")
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".docx"):
                st.write("Đang đọc file DOCX...")
                loader = Docx2txtLoader(file_path)
            elif file.name.endswith(".txt"):
                st.write("Đang đọc file TXT...")
                loader = TextLoader(file_path)
            else:
                st.warning(f"Định dạng file không được hỗ trợ: {file.name}")
                continue

            docs = loader.load()
            if not docs:
                st.warning(f"Không tìm thấy nội dung trong file: {file.name}")
                continue

            st.success(f"Đã đọc được {len(docs)} trang/phần từ {file.name}")

            for doc in docs:
                doc.page_content = preprocess_text(doc.page_content)
                if not doc.page_content.strip():
                    st.warning(f"Trang/phần trong {file.name} không có nội dung sau khi xử lý")
                    continue
                doc.metadata['source'] = file.name
                documents.append(doc)

            os.remove(file_path)
            st.success(f"Đã xử lý xong file: {file.name}")
        except Exception as e:
            st.error(f"Lỗi xử lý file {file.name}: {str(e)}")

    st.info(f"Tổng số documents đã xử lý trong batch: {len(documents)}")
    return documents

def process_documents(uploaded_files, reranker, openai_api_key):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True

    if not uploaded_files:
        st.error("Không có file nào được tải lên!")
        return

    st.info(f"Bắt đầu xử lý {len(uploaded_files)} files...")

    if not os.path.exists("temp"):
        os.makedirs("temp")

    batch_size = 4
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        file_batches = [uploaded_files[i:i + batch_size] for i in range(0, len(uploaded_files), batch_size)]
        documents = []
        with st.spinner('Đang xử lý documents...'):
            for batch_docs in tqdm(executor.map(process_document_batch, file_batches), total=len(file_batches)):
                documents.extend(batch_docs)

    st.info(f"Số lượng documents sau khi xử lý: {len(documents)}")

    if not documents:
        st.error("Không có documents nào được xử lý thành công!")
        return

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    texts = text_splitter.split_documents(documents)

    for i, text in enumerate(texts):
        text.metadata['chunk_id'] = i
        text.metadata['total_chunks'] = len(texts)

    text_contents = [doc.page_content for doc in texts]

    if not texts:
        st.error("Không tìm thấy văn bản hợp lệ sau khi xử lý documents!")
        return

    st.info(f"Đang xử lý {len(texts)} đoạn văn bản...")

    if not any(text.page_content.strip() for text in texts):
        st.error("Không có nội dung hợp lệ trong các documents!")
        return

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        test_embedding = embeddings.embed_query("test")
        if not test_embedding:
            st.error("Failed to generate embeddings! Please check your OpenAI API key.")
            return
    except Exception as e:
        st.error(f"Error with OpenAI embeddings: {str(e)}")
        return

    try:
        vector_store = Chroma.from_documents(
            texts,
            embedding=embeddings,
            persist_directory="./chroma_store"
        )
        vector_store.persist()
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return

    bm25_retriever = BM25Retriever.from_texts(
        text_contents,
        bm25_impl=BM25Okapi,
        preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split()
    )

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_store.as_retriever(search_kwargs={"k": 5})
        ],
        weights=[0.4, 0.6]
    )

    st.session_state.retrieval_pipeline = {
        "ensemble": ensemble_retriever,
        "reranker": reranker,
        "texts": text_contents,
        "knowledge_graph": build_knowledge_graph(texts)
    }

    st.success(f"✅ Đã xử lý thành công {len(uploaded_files)} files!")
    st.write(f"📊 Tổng số chunks: {len(texts)}")
    st.write(f"📊 Kích thước trung bình mỗi chunk: {sum(len(t.page_content) for t in texts)/len(texts):.0f} ký tự")

    st.session_state.documents_loaded = True
    st.session_state.processing = False

    if "knowledge_graph" in st.session_state.retrieval_pipeline:
        G = st.session_state.retrieval_pipeline["knowledge_graph"]
        st.write(f"🔗 Total Nodes: {len(G.nodes)}")
        st.write(f"🔗 Total Edges: {len(G.edges)}")
        st.write(f"🔗 Sample Nodes: {list(G.nodes)[:10]}")
        st.write(f"🔗 Sample Edges: {list(G.edges)[:10]}")
