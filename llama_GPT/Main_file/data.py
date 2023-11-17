from llm import *
from embedding import * 
import os
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

embed = load_embedding_model(model_path="all-MiniLM-L6-v2")
# #google colab importing document
# from google.colab import drive
# drive.mount("/content/drive", force_remount=True)

# # loading and splitting the documents
# docs = load_pdf_data(file_path="/stat7.pdf")
# documents = split_docs(documents=docs)
pdf_file_path = "/path/to/your/local/stat7.pdf"

# Check if the file exists
if not os.path.exists(pdf_file_path):
    print("Error: The specified file does not exist.")
else:
    # Load and split the documents
    docs = load_pdf_data(file_path=pdf_file_path)
    documents = split_docs(documents=docs)
# creating vectorstore
vectorstore = create_embeddings(documents, embed)

# converting vectorstore to a retriever
retriever = vectorstore.as_retriever()

# Creating the prompt from the template which we created before
prompt = PromptTemplate.from_template(template)

# Creating the chain
chain = load_qa_chain(retriever, llm, prompt)

# Getting user query and generating response
user_query = "Which one is the best industry"
get_response(user_query, chain)