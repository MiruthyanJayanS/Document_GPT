# Document GPT

A "Chat with your PDF" style tool that runs **entirely within a single Jupyter Notebook**, designed to run on **Google Colab's free T4 GPU** — keeping the whole pipeline local to the notebook environment without needing separate backend/frontend services.

## 📌 About

Document GPT lets you upload a document and interact with it conversationally — ask questions, get summaries, extract information — similar in spirit to other "chat with PDF" tools, but built to run end-to-end inside one `.ipynb` file. Everything from document processing to model inference happens within the notebook, leveraging Colab's free T4 GPU for acceleration.

This makes it a self-contained, easy-to-run alternative that doesn't require setting up a separate environment, server, or deployment — just open the notebook and run it on Colab.

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Environment | Jupyter Notebook (Google Colab) |
| Hardware | T4 GPU (Google Colab) |

> Add specific libraries used for embeddings/LLM inference (e.g. LangChain, sentence-transformers, a local LLM, FAISS/Chroma for vector storage) once confirmed.

## ✨ Features

- End-to-end document Q&A pipeline contained in a single notebook
- Runs on Google Colab's free-tier T4 GPU — no local GPU or paid API required
- Self-contained: no separate backend, frontend, or deployment setup needed

## 🚀 Getting Started

### Prerequisites
- A Google account (to use Google Colab)
- The notebook file from this repository

### Running on Google Colab
1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `.ipynb` file from this repository (or open it directly from GitHub via Colab's "Open from GitHub" option)
3. In Colab, go to **Runtime → Change runtime type → select T4 GPU**
4. Run all cells in order
5. Upload your document when prompted, and start asking questions

## 🎯 Learning Outcomes

- Building a self-contained document Q&A pipeline using free cloud GPU resources
- Practical experience with document processing and LLM-based question answering
- Designing for accessibility — making the tool runnable by anyone with just a Google account, no local setup

## 📄 License

This repository is for personal learning and portfolio purposes.
