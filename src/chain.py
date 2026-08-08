from src.config import llm
from src.prompts import RAG_PROMPT
from src.retriever import retrieve
from langsmith import traceable


@traceable(name="RAG Pipeline")
def generate_answer(question: str):
    results = retrieve(question)

    if not results:
        return [], (
            "I couldn't find the answer in the provided documents."
        )

    context = "\n\n".join(
        doc.page_content
        for doc, _ in results
    )

    prompt = RAG_PROMPT.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    response = llm.invoke(prompt)

    return results, response.content