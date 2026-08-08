import subprocess
from pathlib import Path


MERMAID_DIAGRAM = r"""
flowchart TD

    subgraph INGEST["Document Ingestion"]
        A["PDF Documents"] --> B["PyPDFLoader"]
        B --> C["Text Splitting"]
        C --> D["Embeddings"]
        D --> E[("ChromaDB")]
    end

    subgraph QUERY["Query Pipeline"]
        F["User Query"] --> G["Retriever"]
        G --> E
        E --> H["Top-K Chunks"]
        H --> I{"Relevance Guard"}

        I -->|Relevant| J["Build Context"]
        I -->|Not Relevant| K["Fallback Response"]

        J --> L["RAG Prompt"]
        L --> M["LLM"]
        M --> N["Final Answer"]
        N --> O["Source Citations"]
    end
"""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"

MMD_FILE = IMAGES_DIR / "classic-rag-workflow.mmd"
PNG_FILE = IMAGES_DIR / "classic-rag-workflow.png"

IMAGES_DIR.mkdir(exist_ok=True)

MMD_FILE.write_text(
    MERMAID_DIAGRAM,
    encoding="utf-8",
)

subprocess.run(
    [
        "mmdc",
        "-i",
        str(MMD_FILE),
        "-o",
        str(PNG_FILE),
        "-p",
        str(PROJECT_ROOT / "scripts" / "puppeteer-config.json"),
    ],
    check=True,
)

print(f"Diagram saved to: {PNG_FILE}")