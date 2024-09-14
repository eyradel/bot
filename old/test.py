from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline


from dotenv import load_dotenv


load_dotenv()

pipe = pipeline(task="text-generation",model="./Qwen2-0.5B")
model = HuggingFacePipeline(pipeline=pipe)



print(model.invoke("what is bread?"))