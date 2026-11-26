# Python backend

`src/rag_diagnostic` contains the diagnostic engine, CLI, registry, OTLP receiver,
and UI read APIs. `src/rag_pipeline` is the reference RAG application;
`src/rag_otel_adapter` normalizes exported telemetry.

Install and run commands from the **repository root**, where `pyproject.toml`,
shared `docs/` contracts, `configs/`, `data/`, and `runs/` live:

```bash
python -m pip install -e '.[dev,server]'
python -m pytest
python scripts/dev.py
```

See [frontend setup](../frontend/README.md) for standalone server configuration.
