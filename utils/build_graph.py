import streamlit as st
import networkx as nx
import re

def build_knowledge_graph(docs):
    G = nx.Graph()
    for doc in docs:
        entities = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', doc.page_content)
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                G.add_edge(entities[i], entities[i + 1])
    return G


def retrieve_from_graph(query, G, top_k=5):
    st.write(f"🔎 Searching GraphRAG for: {query}")

    query_words = query.lower().split()
    matched_nodes = [node for node in G.nodes if any(word in node.lower() for word in query_words)]
    
    if matched_nodes:
        related_nodes = []
        for node in matched_nodes:
            related_nodes.extend(list(G.neighbors(node))) 
        
        st.write(f"🟢 GraphRAG Matched Nodes: {matched_nodes}")
        return related_nodes[:top_k]
    
    st.write(f"❌ No graph results found for: {query}")
    return []
