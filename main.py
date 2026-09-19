"""CLI entry point — thin wrapper around RagPipeline."""
import yaml
from src.pipeline import RagPipeline


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    pipeline = RagPipeline(cfg)

    print("\nReady. Ask questions about 'Attention Is All You Need' (type 'exit' to quit).\n")

    while True:
        query = input("> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        result = pipeline.answer(query)
        if result["cache_hit"]:
            print(f"\n[cache hit — similarity={result['similarity']:.3f}]\n{result['answer']}\n")
        else:
            print(f"\n{result['answer']}\n")


if __name__ == "__main__":
    main()