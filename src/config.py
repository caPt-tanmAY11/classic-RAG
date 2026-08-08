from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 3

RELEVANCE_THRESHOLD = 0.8

CHROMA_DB_DIR = "chroma_db"