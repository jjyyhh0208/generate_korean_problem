from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

import os
import json
import pinecone
import uuid

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = "edu-rag-korean"
embeddings_model = OpenAIEmbeddings()

pinecone = Pinecone(api_key=pinecone_api_key)
pinecone_host = "https://edu-rag-korean-n7vjsiy.svc.aped-4627-b74a.pinecone.io"
index = pinecone.Index(pinecone_index_name, pinecone_host)

with open(
    "/Users/collegenie/Desktop/generate_korean_problem/concept.json",
    # "/Users/collegenie/Desktop/generate_korean_problem/masterpiece.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

vectors = []
for item in data:
    text = item["text"]
    vector = embeddings_model.embed_query(text)

    vector_data = {
        "id": str(uuid.uuid4()),
        "values": vector,
        "metadata": {
            "text": text,
            "text_type": item["text_type"],
            "main_category": item["main_category"],
            "problem_type": item["problem_type"],
        },
    }
    vectors.append(vector_data)

index.upsert(vectors=vectors)
