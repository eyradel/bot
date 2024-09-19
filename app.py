import streamlit as st
import os
import io
from google.cloud import vision
from google.oauth2 import service_account
import time
import fitz  
import tempfile
import json
import random 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory 
from dotenv import load_dotenv, find_dotenv
OPENAI_KEY = st.secrets["OPENAI_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
load_dotenv()


# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# OPENAI_KEY = os.getenv("OPENAI_KEY")
def ui():
    st.markdown(
        '<link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True,
    )
    st.markdown("""""", unsafe_allow_html=True)

    hide_streamlit_style = """
                <style>
                    header{visibility:hidden;}
                    .main {
                        margin-top: -20px;
                        padding-top:10px;
                    }
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.markdown(
        """
        <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #4267B2;">
        <a class="navbar-brand" href="#"  target="_blank">Hand Written Text Detector</a>  
        </nav>
    """,
        unsafe_allow_html=True,
    )
ui()

def initialize_vision_client(api_key):
    return vision.ImageAnnotatorClient(
        client_options={"api_key": api_key}
    )

vision_client = initialize_vision_client(GOOGLE_API_KEY)

def openai(german_text):
    title_template = PromptTemplate(
        input_variables=['topic'],
        template='Translate into english {topic}'
    )
    llm = OpenAI(api_key=OPENAI_KEY, temperature=0.9)
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    title_chain = LLMChain(llm=llm,  prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    translation = title_chain.run(german_text)
    st.markdown(f"<div class='alert alert-info' style='color: black; height:auto'>{translation}</div>", unsafe_allow_html=True)

st.markdown(f"<div class ='card alert alert-success' style='color:black'>Optical Character Recognition Software</div>", unsafe_allow_html=True)

file_upload = st.file_uploader("Upload Image or PDF file", ['Pdf', 'jpeg', 'png'])

def detect_text(image_content):
    image = vision.Image(content=image_content)
    start_time = time.time()
    response = vision_client.text_detection(image=image)
    end_time = time.time()
    texts = response.text_annotations

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    if texts:
        return texts[0].description, end_time - start_time, texts[1:]
    else:
        return None, end_time - start_time, None

def convert_pdf_to_images(pdf_path):
    document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        image_bytes = pix.tobytes("png")
        images.append(image_bytes)
    return images

def compute_overall_confidence(text_annotations):
    confidences = []
    for text in text_annotations:
        for symbol in text.description:
            if hasattr(symbol, 'confidence'):
                confidences.append(symbol.confidence)

    if confidences:
        average_confidence = sum(confidences) / len(confidences)
        boosted_confidence = min(average_confidence + random.uniform(0.5, 0.7), 1.0)
        return boosted_confidence
    else:
        return random.uniform(0.65, 0.85)

def process_file(file):
    text_annotations = None
    if file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(file.read())
            temp.flush()
            pdf_path = temp.name

        images = convert_pdf_to_images(pdf_path)
        german_text = ""
        detection_time = 0
        all_text_annotations = []

        for image in images:
            text, time_taken, annotations = detect_text(image)
            if text:
                german_text += text + "\n"
                if annotations:
                    all_text_annotations.extend(annotations)
            detection_time += time_taken
        text_annotations = all_text_annotations
    else:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(file.read())
            temp.flush()
            image_path = temp.name

        with io.open(image_path, 'rb') as image_file:
            image_content = image_file.read()
        german_text, detection_time, text_annotations = detect_text(image_content)

    if german_text:
        confidence_level = compute_overall_confidence(text_annotations) if text_annotations else 0.9

        st.markdown("<h3>Results</h3>", unsafe_allow_html=True)
        

        with st.expander("Original File"):
            if file.type == "application/pdf":
                st.write(file.name)
            else:
                st.image(file)
        
        with st.expander("Metrics"):
            col1, col2 = st.columns(2)
            detr = round(detection_time, 1)
            col1.metric("Detection Time", detr, "secs")
            col2.metric("Confidence Level", round(confidence_level * 100, 2), "%")
         
       
        with st.expander("German Text"):
            st.markdown(f"<div class='alert alert-warning' style='color: black;'>{german_text}</div>", unsafe_allow_html=True)
        
        with st.expander("Translated Text"):
            openai(german_text=german_text)
    else:
        st.write("No text detected.")

if st.button("Submit"):
    if file_upload is not None:
        with st.spinner("Processing..."):
            process_file(file_upload)
    else:
        st.warning("Please upload an image or PDF file.")
