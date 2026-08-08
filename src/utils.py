from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

console = Console()


def show_banner():
    console.print(
        Panel(
            "[bold cyan]Classic RAG[/bold cyan]\n\n"
            "Ask questions about your document collection.\n\n"
            "Type [bold]'exit'[/bold] to quit.",
            title="AI Assistant",
            border_style="cyan",
            expand=False,
        )
    )

def show_retrieval():
    console.print()
    console.print(
        Rule(
            "[bold yellow]Retrieving Relevant Documents[/bold yellow]"
        )
    )

def show_chunk(index: int, doc, score: float):
    content = (
        f"[bold]Source:[/bold] {doc.metadata['source']}\n"
        f"[bold]Page:[/bold] {doc.metadata['page']}\n"
        f"[bold]Score:[/bold] {score:.4f}\n\n"
        f"{doc.page_content[:500]}"
    )

    console.print(
        Panel(
            content,
            title=f"Retrieved Chunk #{index}",
            border_style="green",
        )
    )

def show_answer(answer: str):
    console.print()

    console.print(
        Panel(
            answer,
            title="Final Answer",
            border_style="cyan",
        )
    )

def show_sources(results):
    if not results:
        return

    console.print()
    console.print(
        Rule("[bold blue]Sources[/bold blue]")
    )

    seen = set()

    for index, (doc, _) in enumerate(results, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")

        source_name = source.split("/")[-1]

        source_key = (source_name, page)

        if source_key in seen:
            continue

        seen.add(source_key)

        console.print(
            f"[bold cyan][{index}][/bold cyan] "
            f"{source_name} — "
            f"Page {page + 1 if page is not None else 'N/A'}"
        )

def show_error(message: str):
    console.print(
        Panel(
            message,
            title="ERROR",
            border_style="red",
        )
    )