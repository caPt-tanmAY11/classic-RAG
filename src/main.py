from src.chain import generate_answer
from src.utils import (
    console,
    show_banner,
    show_retrieval,
    show_chunk,
    show_answer,
    show_sources,
    show_error,
)


def main():
    show_banner()

    try:
        while True:
            question = console.input("\n[bold cyan]You:[/bold cyan] ")

            if question.lower() == "exit":
                console.print("\n👋 Goodbye!\n")
                break

            try:
                show_retrieval()

                results, answer = generate_answer(question)

                for index, (doc, score) in enumerate(results, start=1):
                    show_chunk(index, doc, score)

                show_answer(answer)
                show_sources(results)

            except Exception as exc:
                show_error(
                    "Something went wrong while processing your question.\n\n"
                    f"{exc}"
                )
                
    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()