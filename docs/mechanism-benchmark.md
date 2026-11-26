# Technical mechanism benchmark

The default `fault-test-runner` assesses the stage-based diagnostic engine using
controlled captured executions. It includes one case for each of the 15 technical
mechanisms, one operational exception, one multi-fault case, and two controls:
19 executions in total. Cases have deterministic IDs and complete recorded
boundary facts for the injected mechanism. The suite runs the real engine,
Gemini judge, citation validation, and root-cause attribution.

This is an attribution benchmark on controlled trace interventions. It does not
execute a production retriever, train embeddings, or establish production-domain
accuracy. Model outputs are recorded execution artifacts; only the diagnostic
judge performs live inference. Expected mechanisms, stages, and statuses are
kept in a separate ground-truth file and never supplied to the judge.

## Run

Prepare and inspect the traces without making any Gemini requests:

```bash
rag-diagnose fault-test-runner --prepare-only \
  --config configs/mechanism-benchmark.json \
  --output-dir data/fault_runs/mechanisms
```

Execute the prepared suite (or omit `--resume` and use a fresh directory):

```bash
rag-diagnose fault-test-runner --resume \
  --config configs/mechanism-benchmark.json \
  --output-dir data/fault_runs/mechanisms
```

Run a single mechanism for a quick check:

```bash
rag-diagnose fault-test-runner \
  --mechanism OUTPUT_FORMAT_VIOLATION \
  --config configs/mechanism-benchmark.json \
  --output-dir data/fault_runs/format-check
```

`--mechanism` is repeatable. Normal execution uses the configured Gemini model
and credentials. This suite needs no downloaded corpus or separate RAG server.
Provider errors count as failed attempts and are saved; the CLI exits nonzero
when errors occur. `--resume` retries errors and skips successful cases only
when case content, settings, judge identity, and diagnostic source hash match.

To inspect how Gemini diagnoses a case, add `--log-llm`:

```bash
rag-diagnose fault-test-runner --mechanism CHUNK_FRAGMENTATION \
  --config configs/mechanism-benchmark.json \
  --output-dir data/fault_runs/chuck-fragmentation --resume --log-llm
```

Stderr prints `GEMINI JUDGE INPUT` (system instructions, actual contents after
artifact span conversion, model settings and JSON schema), `GEMINI JUDGE OUTPUT`
(raw response and completion metadata), and `GEMINI JUDGE VALIDATED VERDICT`
(accepted root cause and validation results). Output is untruncated. Each exchange
identifies a live API request, cached response, or saved verdict reused by resume.
Logging does not change judge cache keys or benchmark identity. `--prepare-only`
performs no diagnosis and therefore has no judge exchange to print.

To save the logs while keeping the score summary in the terminal, append
`2> gemini-judge.log`. Logs contain the supplied document and query text; they
exclude API keys, client objects, HTTP headers, and raw provider error messages.

Re-score saved observations using the current judge and engine:

```bash
rag-diagnose fault-test-runner \
  --replay-dir data/fault_runs/mechanisms \
  --config configs/mechanism-benchmark.json \
  --output-dir data/fault_runs/mechanisms-replay
```

## Cases

| Mechanism | Intervention and retained evidence |
|---|---|
| `MISSING_SOURCE_INFORMATION` | Complete original corpus contains only unrelated information; normalized and indexed records match it. |
| `EXTRACTION_LOSS` | Original document contains the limitation; normalized text and downstream chunks omit it. |
| `CHUNK_FRAGMENTATION` | A contrastive limitation is split mid-sentence into neighboring chunks with exact offsets; the first fragment is selected without its qualification. |
| `EMBEDDING_MISMATCH` | Query and document model configurations disagree; unrelated text is retrieved. |
| `INDEXING_FAILURE` | An answer-bearing generated/embedded chunk is missing from the complete acknowledged indexed ID set. |
| `TOP_K_EXCLUSION` | Deeper indexed candidates show the relevant chunk at rank 2, below actual top-K of 1. |
| `RERANKING_DEGRADATION` | The relevant chunk enters retrieval and is removed by the reranker. |
| `SIMILARITY_CUTOFF_EXCLUSION` | Relevant candidate score 0.61 falls below an explicit threshold of 0.7; a higher-scored unrelated candidate survives. |
| `TOKEN_BUDGET_EXCLUSION` | Relevant chunk needs 60 tokens when 70 of the 100-token budget are already used. |
| `CONTEXT_TRUNCATION` | Recorded pre-context contains the limitation and post-context omits it. |
| `CONTEXT_DELIVERY_LOSS` | Assembled context contains the facts, but exact model request delivers unrelated text. |
| `INCORRECT_EVIDENCE_EXTRACTION` | Delivered facts are contradicted by the captured answer. |
| `PARTIAL_EVIDENCE_EXTRACTION` | Query asks for methods and limitation; both reach generation, but the response supplies only methods. |
| `OUTPUT_FORMAT_VIOLATION` | JSON-only instruction receives a Markdown-fenced JSON response. |
| `SPECIFICITY_MISMATCH` | Query explicitly asks for technical method names; answer gives only a generic description. |
| Operational fallback | A recorded generation transport exception is attributed to the actual `generation` stage. |
| Multiple faults | Embedding mismatch precedes an independent output format violation; embedding must remain primary. |
| Compliant-format control | Valid JSON should clear the format mechanism without inventing another fault. |
| Insufficient-evidence control | Relevant deeper retrieval is withheld; Top-K attribution must remain `UNKNOWN`. |

## Scoring and artifacts

`report.json` uses benchmark schema `3.0`. Diagnostic verdicts remain schema
`2.0`; captured executions remain schema `1.0`.

- `primary_accuracy`: exact primary mechanism **and** technical stage on fault
  executions, including the multi-fault case. Abstentions and errors count wrong.
- `stage_accuracy`: primary technical stage accuracy on the same fault executions.
- `suite_accuracy`: exact root, complete fault set, and target-status correctness
  across all executions, including controls. Extra fault claims count wrong.
- `accuracy_when_diagnosed` and `fault_abstention_rate`: coverage alongside accuracy.
- `mechanism_precision` / `mechanism_recall`: all supported fault labels, including
  contributors; unknown or missing expected faults reduce recall. Errors also
  contribute false negatives.
- `control_accuracy`: correct specified clear/unknown assessment with no invented
  faults. An expected abstention on the evidence-withheld control is correct.
- `per_mechanism` and `confusion`: individual results and root attribution pairs.

The output directory contains `traces/`, separate `ground_truth.json`,
`verdicts/`, resumable `results/`, and `report.json`. Verdicts retain evidence,
original trace, model response, source revision and settings. A test judge can be
injected in Python to verify runner plumbing; mock accuracy is not semantic judge
accuracy. The full-corpus FP1–FP7 runner remains available only with
`assessment_mode: "legacy_fp"` and the original manifest arguments.

## Diagnostic Accuracy page

Open **Diagnostic Accuracy** from the dashboard and select **Run fault suite**.
The page runs the full technical suite with the diagnostic server's judge
settings. Each run uses a fresh judge cache and may consume Gemini quota.
One run per tenant can be queued or running at a time.

Completed reports are saved under the server data directory in
`accuracy/<tenant>/<run_id>/`. Use **Test run** to open a report and **Compare
with** to compare overall scores and per-mechanism precision. The page shows
runs launched from the UI. CLI runs retain their original output directories.

Per-mechanism precision divides correct fault labels by all predictions for that
label, including contributing faults. A dash indicates no predictions for that
label. Cases with execution errors count against overall accuracy, and an
expected abstention on a control counts as correct if all checks pass.
