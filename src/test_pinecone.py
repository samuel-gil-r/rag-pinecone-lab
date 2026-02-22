import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(".env", override=True)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

# lista indices y revisa si existe el tuyo
print("Indexes:", [i["name"] for i in pc.list_indexes()])
print("Using:", index_name)
print("OK ✅")