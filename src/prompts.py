from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I couldn't find the answer in the provided documents."

Do not make up information.

-------------------------
Context:
{context}
-------------------------

Question:
{question}

Answer:
"""
)