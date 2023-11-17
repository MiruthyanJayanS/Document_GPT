from llm import *

def load_pdf_data(file_path):
    loader = PyMuPDFLoader(file_path=file_path)
    docs = loader.load()
    return docs

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents=documents)
    return chunks

def load_embedding_model(model_path, normalize_embedding=True):
    return HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device':'cuda'},
        encode_kwargs={
            'normalize_embeddings': normalize_embedding
        }
    )

def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(storing_path)
    return vectorstore

prompt = """
### System:
You are an AI Assistant that follows instructions extremely well. \
Help as much as you can.

### User:
{prompt}

### Response:

"""

template = """
### System:
You are a respectful and honest assistant. You have to answer the user's \
questions using only the context provided to you. If you don't know the answer, \
just say you don't know. Don't try to make up an answer.

### Context:
{context}

### User:
{question}

### Response:
"""

def load_qa_chain(retriever, llm, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def get_response(query, chain):
    response = chain({'query': query})
    wrapped_text = textwrap.fill(response['result'], width=100)
    print(wrapped_text)