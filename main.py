import streamlit as st
import os
import fitz 
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")

# Initialize the Hugging Face model pipeline with Qwen2-0.5B
pipe = pipeline(task="text-generation", model="./Qwen2-0.5B")
local_llm = HuggingFacePipeline(pipeline=pipe)

def extract_text_from_multiple_pdfs(uploaded_files):
    """Extract text from multiple uploaded PDFs using PyMuPDF."""
    extracted_texts = []
    for uploaded_file in uploaded_files:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf") 
        text = ""
        for page in doc:
            text += page.get_text()  # Extract text from each page
        extracted_texts.append(text)
    return " ".join(extracted_texts)

@st.cache_resource
def llm_pipeline():
    """Initialize the LLM pipeline using HuggingFacePipeline."""
    return local_llm

def process_summary(extracted_text):
    """Generate a summary from the extracted text using the model."""
    instruction = "Summarize the following text in concise and clear language: " + extracted_text
    llm = llm_pipeline()
    generated_text = llm(instruction)
    
    
    if isinstance(generated_text, str):
        return generated_text
    elif isinstance(generated_text, list) and len(generated_text) > 0:
        return generated_text[0]
    else:
        return "Unexpected output format from the model."

def process_template(extracted_text):
    """Generate a template from the extracted text using the model."""
    first_visit_date = "placeholder for first visit date"
    last_visit_date = "placeholder for last visit date"
    instruction = f"The first visit date was {first_visit_date} and the last visit date was {last_visit_date}. Analyze the text: {extracted_text}"
    llm = llm_pipeline()
    generated_text = llm(instruction)
    
   
    if isinstance(generated_text, str):
        return generated_text
    elif isinstance(generated_text, list) and len(generated_text) > 0:
        return generated_text[0]
    else:
        return "Unexpected output format from the model."

def main():
    """Main function to run the Streamlit app."""
    st.title("Medical Document Analyzer")

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
