from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from rag_pipeline.config import Settings
from rag_pipeline.corpus import OpenRAGBenchLoader
from rag_pipeline.pipeline import RAGPipeline


def _parser(defaults: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index a corpus and ask questions from it.")
    commands = parser.add_subparsers(dest="command", required=True)

    index = commands.add_parser("index", help="Build the Chroma index")
    index.add_argument("--corpus-dir", type=Path, default=defaults.corpus_dir)
    index.add_argument("--chroma-dir", type=Path, default=defaults.chroma_dir)
    index.add_argument("--collection", default=defaults.collection)
    index.add_argument("--embedding-model", default=defaults.embedding_model)
    index.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    index.add_argument("--chunk-overlap", type=int, default=defaults.chunk_overlap)
    index.add_argument("--batch-size", type=int, default=defaults.batch_size)
    index.add_argument("--limit-files", type=int)
    index.add_argument("--reset", action="store_true")

    ask = commands.add_parser("ask", help="Ask a question")
    ask.add_argument("query")
    ask.add_argument("--chroma-dir", type=Path, default=defaults.chroma_dir)
    ask.add_argument("--collection", default=defaults.collection)
    ask.add_argument("--embedding-model", default=defaults.embedding_model)
    ask.add_argument("--llm-model", default=defaults.llm_model)
    ask.add_argument("--top-k", type=int, default=defaults.top_k)
    return parser


def main(argv: list[str] | None = None) -> None:
    defaults = Settings.from_env()
    args = _parser(defaults).parse_args(argv)

    try:
        if args.command == "index":
            settings = replace(
                defaults,
                corpus_dir=args.corpus_dir.expanduser().resolve(),
                chroma_dir=args.chroma_dir.expanduser().resolve(),
                collection=args.collection,
                embedding_model=args.embedding_model,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
            )
            pipeline = RAGPipeline(settings)
            loader = OpenRAGBenchLoader(settings.corpus_dir)
            print("Building index...")
            total = pipeline.index(
                loader.load(limit_files=args.limit_files),
                reset=args.reset,
                progress=lambda count: print(f"Indexed {count} chunks", end="\r"),
            )
            print(f"Indexed {total} chunks.{' ' * 20}")
            return

        settings = replace(
            defaults,
            chroma_dir=args.chroma_dir.expanduser().resolve(),
            collection=args.collection,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            top_k=args.top_k,
        )
        answer, sources = RAGPipeline(settings).ask(args.query)
        print(f"\n{answer}\n\nSources:")
        titles = dict.fromkeys(source.metadata.get("title", "Untitled") for source in sources)
        for number, title in enumerate(titles, start=1):
            print(f"[{number}] {title}")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
