# 📚 AI Knowledge Base RAG Agent (Streamlit + Groq)

## 🌐 Live App  
👉 **https://suhasgowda24-kb-ai-app-cghexq.streamlit.app/**

[![Streamlit App](https://img.shields.io/badge/Launch_Streamlit_App-brightgreen?logo=streamlit)](https://suhasgowda24-kb-ai-app-cghexq.streamlit.app/)

A fully functional Knowledge-Base RAG system using:

- Streamlit
- Groq Llama 3.3 models
- FAISS vector search
- MiniLM embeddings
- Chunk-based retrieval
- Beautiful source viewer

## 🚀 Deployment on Streamlit Cloud

1. Upload these files to your GitHub repo:
   - app.py
   - requirements.txt
   - .streamlit/secrets.toml

2. Go to: https://share.streamlit.io

3. Select your repository and choose `app.py`.

4. Your app will deploy automatically.

## 🔐 Secrets

In `.streamlit/secrets.toml`:

GROQ_API_KEY="your_key"
OPENAI_API_KEY="your_key"


## ▶ Local Usage

```bash
pip install -r requirements.txt
streamlit run app.py


📄 Features

RAG (Retrieval-Augmented Generation)

Chunking for large documents

FAISS vector database

Clean Sources UI

Groq/OpenAI/Ollama fallback

Works with PDFs, TXTs, DOCX

Supports long documents
