import streamlit as st
import os
import fitz 
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document import DocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

load_dotenv()

st.set_page_config(layout="wide")

# Initialize the Hugging Face model pipeline with Qwen2-0.5B
pipe = pipeline(task="text-generation", model="./Qwen2-0.5B")
local_llm = HuggingFacePipeline(pipeline=pipe)

# Initialize Hugging Face embeddings model
embedding_model = HuggingFaceEmbeddings()

# Qdrant vector store initialization
def init_qdrant():
    return Qdrant.from_memory(embedding_model)

vectorstore = init_qdrant()

# Text splitter for long documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Function to extract text from multiple PDFs
def extract_text_from_multiple_pdfs(uploaded_files):
    extracted_texts = []
    for uploaded_file in uploaded_files:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf") 
        text = ""
        for page in doc:
            text += page.get_text()  # Extract text from each page
        extracted_texts.append(text)
    return " ".join(extracted_texts)

# Function to split text and add to the vector store
def split_and_store_text(text):
    chunks = text_splitter.split_text(text)
    documents = [{"content": chunk} for chunk in chunks]
    vectorstore.add_documents(documents)

# Setup RetrievalQA Chain
def init_retrieval_chain():
    retriever = DocumentRetriever(vectorstore)
    return RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=retriever)

retrieval_chain = init_retrieval_chain()

@st.cache_resource
def llm_pipeline():
    return local_llm

def process_summary(extracted_text):
    split_and_store_text(extracted_text)  # Store text in the vector store
    instruction = "Summarize the following text: "
    result = retrieval_chain.run(instruction)  # Retrieve and summarize
    return result

def process_template(extracted_text):
    first_visit_date = "placeholder for first visit date"
    last_visit_date = "placeholder for last visit date"
    instruction = f"The first visit date was {first_visit_date} and the last visit date was {last_visit_date}. Analyze the text: "
    result = retrieval_chain.run(instruction)
    return result

def main():
    st.title("Medical Document Analyzer with RAG")

    # Upload multiple PDF files
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Extracting Text..."):
            extracted_text = extract_text_from_multiple_pdfs(uploaded_files)

        st.success("Text Extracted")

        col1, col2 = st.columns([1, 2])

        if st.button('Process Documents'):
            with col1:
                with st.spinner("Summarizing"):
                    summary = process_summary(extracted_text)
                st.success("Summarized")
                st.write(summary)
            
            with col2:
                with st.spinner("Generating Template"):
                    template = process_template(extracted_text)
                st.success("Template Generated")
                st.write(template)

    st.write("Upload your files")

if __name__ == "__main__":
    main()
