import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
import textwrap 
from langchain_community.document_loaders import PDFMinerLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS

st.set_page_config(layout="wide")


checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.float32
)


persist_directory = "db"

@st.cache_resource
def data_ingestion():
    loader = None
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    documents = loader.load()
    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    print("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    print(f"Creating embeddings. May take some minutes...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 
    
    
@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_summary():
    response = ''
    instruction = "Summarize all the information in the uploaded documents in concise and clear language. Read and think carefully."
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer

def process_template():
    response = ''
    instruction = "the first visit date was {} and the last vist date was {}."
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer



def main():
    
    st.title("Medical doc Analyzer")
    
    uploaded_file = st.file_uploader("", type=["pdf"], accept_multiple_files=False)
        
    if uploaded_file is not None:
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
        with st.spinner("Processing"):
            
            data_ingestion()
        st.success("Data Accepted")

        col1, col2= st.columns([1,2])
        if st.button('Process Documents'):
            with col1:
                with st.spinner("Summarizing"):    
                    summary = process_summary()
                st.success("Summarized")
                st. write(summary)
            with col2:
                with st.spinner("Generating Template"):    
                    template = process_template()
                st.success("Generated")
                st. write(template)
                
    st.write("Upload your files")



if __name__ == "__main__":
    main()