import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from src.config import (
    embeddings,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_DB_DIR,
)


console = Console()

DATA_DIR = Path("data/documents")


def load_documents():
    documents = []

    for pdf_file in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


def create_vector_store(chunks):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )

    return vector_store


if __name__ == "__main__":
    # Remove existing vector database
    db_path = Path(CHROMA_DB_DIR)

    if db_path.exists():
        console.print(
            "[yellow]Removing existing vector database...[/yellow]"
        )
        shutil.rmtree(db_path)

    # Load and split documents
    docs = load_documents()
    chunks = split_documents(docs)

    console.print(f"Loaded Pages : {len(docs)}")
    console.print(f"Total Chunks : {len(chunks)}")

    console.print("\n[cyan]Creating vector database...[/cyan]")

    # Create vector database
    vector_store = create_vector_store(chunks)

    console.print("[green]Done![/green]")

    console.print(
        f"\nStored {vector_store._collection.count()} chunks in ChromaDB."
    )

    console.print(
        "\n[bold green]✓ Vector database created successfully[/bold green]"
    )