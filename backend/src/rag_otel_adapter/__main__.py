"""Normalize an OTLP/JSON trace file:  python -m rag_otel_adapter trace.otlp.jsonl

The file holds one OTLP/JSON export request per line, as written by the
reference app when RAG_TRACE_FILE is set. Prints the normalized records.
"""

import json
import sys
from pathlib import Path

from rag_otel_adapter import normalize


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m rag_otel_adapter TRACE_FILE.otlp.jsonl")
    lines = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    requests = [json.loads(line) for line in lines if line.strip()]
    print(json.dumps(normalize(requests), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
