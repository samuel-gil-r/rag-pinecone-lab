import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(".env", override=True)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
resp = llm.invoke("Di 'listo' en 1 palabra.")
print(resp.content)