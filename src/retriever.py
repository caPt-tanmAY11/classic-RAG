from langchain_chroma import Chroma
from langsmith import traceable

from src.config import (
    embeddings,
    TOP_K,
    RELEVANCE_THRESHOLD,
    CHROMA_DB_DIR,
)


vector_store = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings,
)

# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={
#         "k": 3,
#     },
# )

@traceable(
    name="Retrieve Documents",
    metadata={
        "top_k": TOP_K,
    },
)
def retrieve(query: str):
    results = vector_store.similarity_search_with_score(
        query=query,
        k=TOP_K,
    )

    return apply_relevance_guard(results)

@traceable(
    name="Relevance Guard",
    metadata={
        "relevance_threshold": RELEVANCE_THRESHOLD,
    },
)
def apply_relevance_guard(results):
    return [
        (doc, score)
        for doc, score in results
        if score <= RELEVANCE_THRESHOLD
    ]


if __name__ == "__main__":
    query = input("Question: ")

    documents = retrieve(query)

    print(f"\nRetrieved {len(documents)} documents.\n")

    for index, (doc, score) in enumerate(documents, start=1):
        print(f"\nDocument {index}")
        print("-" * 60)

        print(f"Score : {score:.4f}")

        print(f"Source: {doc.metadata['source']}")

        print(f"Page  : {doc.metadata['page']}")

        print()

        print(doc.page_content[:500])

        print("\n")