from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_teddynote.community.pinecone import upsert_documents
from langchain_upstage import UpstageEmbeddings
import numpy as np

import os
import pinecone
import uuid

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = "edu-rag-korean"
embeddings_model = OpenAIEmbeddings()

pinecone = Pinecone(api_key=pinecone_api_key)
pinecone_host = "https://edu-rag-korean-n7vjsiy.svc.aped-4627-b74a.pinecone.io"
index = pinecone.Index(pinecone_index_name, pinecone_host)


# PDF 로딩 및 텍스트 추출
pdf_filepath = r"/Users/collegenie/Desktop/RAG/국어(천재노)/3. 언어랑 국어랑 놀자/국어1-1(노)_3-0-0_지도서_천재교육.pdf"
loader = PyMuPDFLoader(pdf_filepath)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    length_function=len,
)

texts = []

for page in pages[71:76]:
    chunks = text_splitter.split_text(page.page_content)
    texts.extend(chunks)

# 벡터 및 메타데이터 업로드
for text, emb in zip(texts, embeddings_model.embed_documents(texts)):
    vector_id = str(uuid.uuid4())
    metadata = {"text": text, "main_category": "문법", "sub_category": "품사"}
    vector_data = {
        "id": vector_id,
        "values": emb.tolist() if isinstance(emb, np.ndarray) else emb,
        "metadata": metadata,
    }

    # index.upsert(vectors=[vector_data])
