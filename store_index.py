from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from src.helper import load_pdf, text_split, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name = index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

docsearch = PineconeVectorStore.from_texts(
    texts= [t.page_content for t in text_chunks],
    embedding=embeddings,
    index_name=index_name
)