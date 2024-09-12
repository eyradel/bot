import streamlit as st
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key (ensure your .env file contains the OPENAI_API_KEY variable)
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(layout="wide")

# Initialize OpenAI LLM with the API key
@st.cache_resource
def openai_llm():
    return OpenAI(api_key=openai_api_key, temperature=0.3)

def extract_text_from_multiple_pdfs(uploaded_files):
    """Extract text from multiple uploaded PDFs using PyMuPDF."""
    extracted_texts = []
    for uploaded_file in uploaded_files:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")  # Open PDF from uploaded file
        text = ""
        for page in doc:
            text += page.get_text()  # Extract text from each page
        extracted_texts.append(text)
    return " ".join(extracted_texts)

def process_summary(extracted_text):
    """Generate a summary from the extracted text using OpenAI LLM."""
    prompt_template = """
    Summarize the following text in concise and clear language:
    {text}
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    llm = openai_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text=extracted_text)
    return summary

def process_template(extracted_text):
    """Generate a template analysis from the extracted text using OpenAI LLM."""
    prompt_template = """
    The first visit date was {first_visit_date} and the last visit date was {last_visit_date}. 
    Analyze the following text:
    {text}
    """
    first_visit_date = "placeholder for first visit date"
    last_visit_date = "placeholder for last visit date"
    prompt = PromptTemplate(
        input_variables=["first_visit_date", "last_visit_date", "text"], 
        template=prompt_template
    )
    llm = openai_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    template_analysis = chain.run(first_visit_date=first_visit_date, last_visit_date=last_visit_date, text=extracted_text)
    return template_analysis

def main():
    """Main function to run the Streamlit app."""
    st.title("Medical Document Analyzer")

    # Upload multiple PDF files
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        # Extract text from multiple PDFs using PyMuPDF
        with st.spinner("Extracting Text..."):
            extracted_text = extract_text_from_multiple_pdfs(uploaded_files)

        st.success("Text Extracted")

        col1, col2 = st.columns([1, 2])

        # Button to process documents
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
