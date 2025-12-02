import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(page_title="AI Knowledge Base Agent", layout="wide")

# -------------------------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# -------------------------------------------------
# VECTOR STORE
# -------------------------------------------------
VECTOR_DIM = 384
index = faiss.IndexFlatL2(VECTOR_DIM)
documents = []


# -------------------------------------------------
# TEXT CHUNKING
# -------------------------------------------------
def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# -------------------------------------------------
# EMBEDDING + FAISS HELPERS
# -------------------------------------------------
def embed_text(text):
    return embedder.encode([text])[0]


def add_document_to_index(text, source):
    vector = embed_text(text)
    index.add(np.array([vector]).astype("float32"))
    documents.append({"text": text, "source": source})


# -------------------------------------------------
# SAFE SEARCH (fix for IndexError)
# -------------------------------------------------
def search_docs(query, k=5):
    if len(documents) == 0:
        return []

    q_vec = embed_text(query).astype("float32")
    distances, idxs = index.search(np.array([q_vec]), k)

    results = []
    for i, dist in zip(idxs[0], distances[0]):
        if i == -1:   # skip invalid index
            continue
        if 0 <= i < len(documents):
            results.append({**documents[i], "score": float(dist)})

    return results


# -------------------------------------------------
# LLM CLIENT (Groq → OpenAI → Ollama)
# -------------------------------------------------
def call_llm(prompt, system="You are an expert AI assistant."):

    # --- GROQ ---
    if "GROQ_API_KEY" in st.secrets:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {st.secrets.GROQ_API_KEY}"}

        payload = {
            "model": "llama-3.3-70b-versatile",  # Working 2025 model
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800
        }

        r = requests.post(url, headers=headers, json=payload)
        data = r.json()

        if "choices" not in data:
            st.error("Groq API Error:")
            st.json(data)
            return "⚠️ Groq API returned an error."

        return data["choices"][0]["message"]["content"]

    # --- OPENAI ---
    if "OPENAI_API_KEY" in st.secrets:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {st.secrets.OPENAI_API_KEY}"}

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
        }

        r = requests.post(url, headers=headers, json=payload)
        return r.json()["choices"][0]["message"]["content"]

    # --- OLLAMA (Local Fallback) ---
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1:8b",
        "messages": [{"role": "user", "content": prompt}]
    }
    r = requests.post(url, json=payload)
    return r.json()["message"]["content"]


# -------------------------------------------------
# RAG PIPELINE
# -------------------------------------------------
def rag_pipeline(query):
    matches = search_docs(query, k=5)

    if not matches:
        return "No relevant documents found. Please upload files.", []

    matches = matches[:3]  # reduce context

    context = "\n\n".join(
        [f"Source: {m['source']}\n{m['text']}" for m in matches]
    )

    prompt = f"""
You are a Knowledge Base Expert AI.
Use ONLY the context below to answer and cite the sources.

CONTEXT:
{context}

QUESTION:
{query}

RULES:
- Do NOT hallucinate.
- Only use info from context.
- Add citations like (source).
"""

    answer = call_llm(prompt)
    return answer, matches


# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("📚 AI Knowledge Base Agent")
st.write("Upload documents → ask anything → get precise RAG answers with citations.")

# -------------------------------------------------
# SIDEBAR UPLOAD
# -------------------------------------------------
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF | TXT | DOCX files", accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
            try:
                text = f.read().decode(errors="ignore")

                chunks = chunk_text(text)
                for chunk in chunks:
                    add_document_to_index(chunk, f.name)

            except:
                st.error(f"❌ Could not read: {f.name}")

        st.success("Documents added successfully!")


# -------------------------------------------------
# CHAT INTERFACE
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask your knowledge base...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        answer, sources = rag_pipeline(query)
        st.write(answer)

        # CLEAN SOURCES UI
        with st.expander("📄 Sources Used"):
            if not sources:
                st.info("No matching sources.")
            else:
                for i, s in enumerate(sources, 1):
                    snippet = s["text"][:180].strip().replace("\n", " ")
                    st.markdown(
                        f"""
                        **{i}. {s['source']}**  
                        _{snippet}..._
                        """
                    )
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer})










# import streamlit as st
# import chromadb
# import requests

# st.set_page_config(page_title="AI Knowledge Base Agent", layout="wide")

# # -------------------------
# # ChromaDB Client
# # -------------------------
# client = chromadb.Client()
# collection = client.get_or_create_collection(
#     name="kb_store",
#     metadata={"hnsw:space": "cosine"}
# )

# # -------------------------
# # HuggingFace Embeddings (SAFE)
# # -------------------------
# def get_embedding(text):
#     HF_TOKEN = st.secrets["HF_API_KEY"]

#     url = "https://router.huggingface.co/hf-inference/models/intfloat/e5-small-v2"

#     headers = {"Authorization": f"Bearer {HF_TOKEN}"}

#     payload = {
#         "inputs": text[:500],    # safe chunk length
#         "parameters": {
#             "task": "feature-extraction"   # 👈 REQUIRED FIX
#         }
#     }

#     try:
#         r = requests.post(url, headers=headers, json=payload, timeout=30)

#         # Try JSON decode
#         try:
#             data = r.json()
#         except:
#             st.error("❌ HF returned non-JSON response")
#             st.code(r.text[:400])
#             return None

#         # HF-level error
#         if isinstance(data, dict) and "error" in data:
#             st.error("❌ HuggingFace API Error")
#             st.json(data)
#             return None

#         return data[0]   # embedding vector

#     except Exception as e:
#         st.error(f"❌ HF request failed: {e}")
#         return None





# # -------------------------
# # Add document into ChromaDB
# # -------------------------
# def add_document(text, source):
#     emb = get_embedding(text)
#     if emb is None:
#         return

#     collection.add(
#         documents=[text],
#         metadatas=[{"source": source}],
#         ids=[source + "_" + str(len(text))]
#     )

# # -------------------------
# # Groq LLM
# # -------------------------
# def groq_llm(system, prompt):
#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {"Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}"}
#     payload = {
#         "model": "llama-3.3-70b-versatile",
#         "messages": [
#             {"role": "system", "content": system},
#             {"role": "user", "content": prompt}
#         ],
#         "max_tokens": 700
#     }

#     r = requests.post(url, headers=headers, json=payload)
#     data = r.json()

#     if "choices" not in data:
#         st.error("❌ Groq LLM Error")
#         st.json(data)
#         return "Groq error occurred."

#     return data["choices"][0]["message"]["content"]

# # -------------------------
# # RAG Search Pipeline
# # -------------------------
# def rag_query(query):
#     q_embed = get_embedding(query)
#     if q_embed is None:
#         return "Embedding failed. Try again.", []

#     results = collection.query(query_embeddings=[q_embed], n_results=3)

#     if len(results["documents"][0]) == 0:
#         return "No relevant documents found. Upload files first.", []

#     docs = results["documents"][0]
#     metas = results["metadatas"][0]

#     context = ""
#     for d, m in zip(docs, metas):
#         context += f"Source: {m['source']}\n{d}\n\n"

#     prompt = f"""
# Use ONLY the following context to answer. Cite using (source).

# Context:
# {context}

# Question:
# {query}
# """

#     answer = groq_llm("You are a precise RAG assistant.", prompt)
#     return answer, list(zip(docs, metas))

# # -------------------------
# # Streamlit UI
# # -------------------------
# st.title("📚 AI Knowledge Base (Groq + HuggingFace + ChromaDB)")
# st.write("Upload documents → Ask questions → Get precise RAG answers.")

# # Upload panel
# with st.sidebar:
#     st.header("📄 Upload Files")
#     files = st.file_uploader("Upload TXT / PDF / DOCX", accept_multiple_files=True)

#     if files:
#         for f in files:
#             text = f.read().decode(errors="ignore")

#             # Safe chunk size (HF-friendly)
#             chunks = [text[i:i+500] for i in range(0, len(text), 500)]

#             for c in chunks:
#                 add_document(c, f.name)

#         st.success("Documents indexed successfully!")

# # Chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])

# # User input
# query = st.chat_input("Ask something about your uploaded documents...")

# if query:
#     st.session_state.messages.append({"role": "user", "content": query})

#     with st.chat_message("assistant"):
#         answer, sources = rag_query(query)
#         st.write(answer)

#         with st.expander("📄 Sources Used"):
#             for i, (doc, meta) in enumerate(sources, 1):
#                 snippet = doc[:180].replace("\n", " ")
#                 st.markdown(f"**{i}. {meta['source']}** — _{snippet}..._")

#     st.session_state.messages.append({"role": "assistant", "content": answer})







# import streamlit as st
# import os
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np
# import requests

# # -------------------------------------------------
# # STREAMLIT CONFIG
# # -------------------------------------------------
# st.set_page_config(page_title="AI Knowledge Base Agent", layout="wide")

# # -------------------------------------------------
# # LOAD EMBEDDING MODEL
# # -------------------------------------------------
# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# embedder = load_embedding_model()

# # -------------------------------------------------
# # VECTOR STORE
# # -------------------------------------------------
# VECTOR_DIM = 384
# index = faiss.IndexFlatL2(VECTOR_DIM)
# documents = []


# # -------------------------------------------------
# # TEXT CHUNKING
# # -------------------------------------------------
# def chunk_text(text, chunk_size=800, overlap=200):
#     chunks = []
#     start = 0
#     length = len(text)

#     while start < length:
#         end = min(start + chunk_size, length)
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start += chunk_size - overlap

#     return chunks


# # -------------------------------------------------
# # EMBEDDING + FAISS HELPERS
# # -------------------------------------------------
# def embed_text(text):
#     return embedder.encode([text])[0]


# def add_document_to_index(text, source):
#     vector = embed_text(text)
#     index.add(np.array([vector]).astype("float32"))
#     documents.append({"text": text, "source": source})


# # -------------------------------------------------
# # SAFE SEARCH (fix for IndexError)
# # -------------------------------------------------
# def search_docs(query, k=5):
#     if len(documents) == 0:
#         return []

#     q_vec = embed_text(query).astype("float32")
#     distances, idxs = index.search(np.array([q_vec]), k)

#     results = []
#     for i, dist in zip(idxs[0], distances[0]):
#         if i == -1:   # skip invalid index
#             continue
#         if 0 <= i < len(documents):
#             results.append({**documents[i], "score": float(dist)})

#     return results


# # -------------------------------------------------
# # LLM CLIENT (Groq → OpenAI → Ollama)
# # -------------------------------------------------
# def call_llm(prompt, system="You are an expert AI assistant."):

#     # --- GROQ ---
#     if "GROQ_API_KEY" in st.secrets:
#         url = "https://api.groq.com/openai/v1/chat/completions"
#         headers = {"Authorization": f"Bearer {st.secrets.GROQ_API_KEY}"}

#         payload = {
#             "model": "llama-3.3-70b-versatile",  # Working 2025 model
#             "messages": [
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": prompt}
#             ],
#             "max_tokens": 800
#         }

#         r = requests.post(url, headers=headers, json=payload)
#         data = r.json()

#         if "choices" not in data:
#             st.error("Groq API Error:")
#             st.json(data)
#             return "⚠️ Groq API returned an error."

#         return data["choices"][0]["message"]["content"]

#     # --- OPENAI ---
#     if "OPENAI_API_KEY" in st.secrets:
#         url = "https://api.openai.com/v1/chat/completions"
#         headers = {"Authorization": f"Bearer {st.secrets.OPENAI_API_KEY}"}

#         payload = {
#             "model": "gpt-4o-mini",
#             "messages": [
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": prompt}
#             ],
#         }

#         r = requests.post(url, headers=headers, json=payload)
#         return r.json()["choices"][0]["message"]["content"]

#     # --- OLLAMA (Local Fallback) ---
#     url = "http://localhost:11434/api/chat"
#     payload = {
#         "model": "llama3.1:8b",
#         "messages": [{"role": "user", "content": prompt}]
#     }
#     r = requests.post(url, json=payload)
#     return r.json()["message"]["content"]


# # -------------------------------------------------
# # RAG PIPELINE
# # -------------------------------------------------
# def rag_pipeline(query):
#     matches = search_docs(query, k=5)

#     if not matches:
#         return "No relevant documents found. Please upload files.", []

#     matches = matches[:3]  # reduce context

#     context = "\n\n".join(
#         [f"Source: {m['source']}\n{m['text']}" for m in matches]
#     )

#     prompt = f"""
# You are a Knowledge Base Expert AI.
# Use ONLY the context below to answer and cite the sources.

# CONTEXT:
# {context}

# QUESTION:
# {query}

# RULES:
# - Do NOT hallucinate.
# - Only use info from context.
# - Add citations like (source).
# """

#     answer = call_llm(prompt)
#     return answer, matches


# # -------------------------------------------------
# # STREAMLIT UI
# # -------------------------------------------------
# st.title("📚 AI Knowledge Base Agent")
# st.write("Upload documents → ask anything → get precise RAG answers with citations.")

# # -------------------------------------------------
# # SIDEBAR UPLOAD
# # -------------------------------------------------
# with st.sidebar:
#     st.header("📄 Upload Documents")
#     uploaded_files = st.file_uploader(
#         "Upload PDF | TXT | DOCX files", accept_multiple_files=True
#     )

#     if uploaded_files:
#         for f in uploaded_files:
#             try:
#                 text = f.read().decode(errors="ignore")

#                 chunks = chunk_text(text)
#                 for chunk in chunks:
#                     add_document_to_index(chunk, f.name)

#             except:
#                 st.error(f"❌ Could not read: {f.name}")

#         st.success("Documents added successfully!")


# # -------------------------------------------------
# # CHAT INTERFACE
# # -------------------------------------------------
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])

# query = st.chat_input("Ask your knowledge base...")

# if query:
#     st.session_state.messages.append({"role": "user", "content": query})

#     with st.chat_message("user"):
#         st.write(query)

#     with st.chat_message("assistant"):
#         answer, sources = rag_pipeline(query)
#         st.write(answer)

#         # CLEAN SOURCES UI
#         with st.expander("📄 Sources Used"):
#             if not sources:
#                 st.info("No matching sources.")
#             else:
#                 for i, s in enumerate(sources, 1):
#                     snippet = s["text"][:180].strip().replace("\n", " ")
#                     st.markdown(
#                         f"""
#                         **{i}. {s['source']}**  
#                         _{snippet}..._
#                         """
#                     )
#                     st.markdown("---")

#     st.session_state.messages.append({"role": "assistant", "content": answer})
