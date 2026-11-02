# RAG Diagnostic Tool

When the RAG pipeline gets a question wrong, this tool works out which stage caused it.
It names the stage, shows the evidence, and gives a confidence score. When the evidence
isn't strong enough to say, it abstains instead of guessing.

This is a starting point. It covers the core of what the SOW asks for, and the tests
show the logic holds up end to end. Some pieces still need tuning, and those are listed
at the bottom.

## Folder layout

```text
src/rag_diagnostic/
├── trace.py          Data types for what each stage produces
├── adapter.py        The interface a pipeline fills in so the tool can read it
├── cases.py          Reads the cases to diagnose and the labels used for scoring
├── config.py         The tool's own settings
├── locate.py         Finds where the answer passage ended up
├── judge.py          Decides whether an answer is right
├── checks.py         One check per stage, following the passage through the pipeline
├── diagnose.py       Runs the fix, sets the confidence, and builds the verdict
├── taxonomy.py       The named failure modes and how to fix each one
├── evaluate.py       Scores the tool against labeled cases
├── runlog.py         Saves a record of every run
├── cli.py            The rag-diagnose command
└── adapters/
    ├── toy.py            A small practice pipeline you can break on purpose
    └── open_ragbench.py  Connects to rag_pipeline

examples/
├── toy_cases.jsonl   13 practice cases, one for each kind of failure
└── toy_labels.jsonl  The right answer for each practice case

configs/diagnostic.json   Example settings
docs/diagnostic-tool.md   This file
tests/                    test_diagnose.py, test_diagnostic_parts.py,
                          test_open_ragbench_adapter.py
runs/                     Created when you run the tool; holds the run records
```

## How the pieces fit together

A case comes in with a question, the reference answer, and where the answer lives in
the documents. The adapter runs the pipeline and hands back what each stage produced.
The judge decides whether the answer is wrong. If it is, the checks follow the answer
passage through each stage to find where it got lost. Then `diagnose.py` fixes that
stage and reruns the pipeline to confirm it. The result is a verdict, and it gets saved
along with the code version and settings that produced it.

## How a diagnosis works

1. **Run and judge.** The tool runs the pipeline and judges the answer. If the answer
   is right, there's nothing to diagnose.
2. **Check each stage.** The tool follows the answer passage through the pipeline. Is it
   in the parsed text? Is it whole inside one chunk? Is that chunk in the index? Did
   retrieval rank it high enough? Did the reranker keep it? Did its text make it into
   the context? Each check comes back as ok, fault, unknown, or not applicable, along
   with the numbers behind it.
3. **Test the model on its own.** The tool gives the model only the answer passage. If
   it still gets the question wrong, generation is at fault, no matter what else
   happened.
4. **Fix and rerun.** A failed check makes a stage a suspect, but it doesn't prove that
   stage caused the wrong answer. To confirm it, the tool fixes the first stage that
   failed and reruns everything after it:
   - Parsing or chunking: a perfect chunk with the passage is put first.
   - Indexing or retrieval: the real answer chunk is put first.
   - Reranking: the reranker is skipped.
   - Context: the top chunks go in whole, with nothing cut.

   If the passage now reaches the model, that stage is confirmed. If it still doesn't,
   another stage must be broken too, so the tool abstains.
5. **Set the confidence.** The confidence comes from how the fix went:

   | What happened | Confidence |
   |---|---|
   | The fix corrected the answer | 0.9 |
   | The fix got the passage through, but the model still failed the model-only test | 0.8, plus generation at 0.85 |
   | The fix got the passage through, and the model passed the model-only test | 0.7, plus generation at 0.6 |
   | The fix didn't get the passage through | 0.4 |

   An unclear check lowers the confidence by 0.2. Anything under the `abstain_below`
   setting (0.6 by default) becomes an abstention, and the tool still names its
   leading suspects.

## What the tool reads

The tool reads two files. The case file holds the questions to diagnose. The label file
holds the right answers, and only the scoring code reads it. Keeping them apart means
the tool can't peek at the answer while it's working.

A case looks like this:

```json
{"case_id": "c001",
 "query": "Who built the Velra lighthouse?",
 "reference_answer": "Mara Quill",
 "gold": [{"doc_id": "harbor",
           "evidence": "The Velra lighthouse was built in 1874 by the engineer Mara Quill."}],
 "pipeline": {"chunk_size": 5}}
```

A label looks like this:

```json
{"case_id": "c001", "stages": ["chunking"], "fault_type": "chunk_split_evidence"}
```

A few rules:

- `evidence` should be the shortest passage that supports the answer. A whole section
  is too long, since it would almost always look split across chunks.
- `pipeline` holds the settings for running that case. It shouldn't name the fault.
  The tool rejects any case that includes `stages` or `fault_type`.
- `stages` can list more than one stage. A case with nothing wrong uses an empty list.

The format can still change. It's written down here so the tool has something concrete
to read.

## Connecting a pipeline

To connect a pipeline, subclass `PipelineAdapter` and fill in these methods:
`describe`, `raw_text`, `parsed_text`, `chunks`, `indexed_ids`, `retrieve`,
`assemble_context` and `generate`. If the pipeline has a reranker, fill in `rerank` too.
Then set `retrieval_k` and `context_k`.

Next, write a function that takes a case's pipeline settings and returns your adapter.
Pass it to the command as `--adapter your.module:function`. The practice pipeline in
`adapters/toy.py` is the shortest full example.

## Running it

From the repo root, after `python -m pip install -e ".[dev]"`:

```bash
python -m pytest

rag-diagnose diagnose --cases examples/toy_cases.jsonl \
  --adapter rag_diagnostic.adapters.toy:build --case-id toy-10

rag-diagnose evaluate --cases examples/toy_cases.jsonl --labels examples/toy_labels.jsonl \
  --adapter rag_diagnostic.adapters.toy:build --split all
```

On the practice cases, the tool answers 11 of the 13 and gets all 11 right. It abstains
on two:

- `toy-11` has two broken stages before the model, retrieval and the reranker. Fixing
  retrieval alone isn't enough, so the tool abstains instead of guessing.
- `toy-13` hides the original document, so the tool can't tell whether the answer was
  lost in parsing or was never there.

This only shows that the logic works. It doesn't say anything yet about accuracy on
real data.

The test in `tests/test_open_ragbench_adapter.py` runs the adapter against the real
`rag_pipeline` code and a real ChromaDB index. It swaps in a simple embedder and a
stand-in for the OpenAI call, so it doesn't need any downloads or an API key.

## How this lines up with the SOW

| What the SOW asks for | Where it is |
|---|---|
| A verdict with the stage, evidence and confidence | `diagnose.py` |
| Reasoning an engineer can check | `Verdict.render()` and the check results in `checks.py` |
| A controlled comparison, not just recorded output | `fix_and_rerun()` in `diagnose.py` |
| The tool reports its own accuracy, including misses | `evaluate.py` and `rag-diagnose evaluate` |
| Abstains instead of guessing | `abstain_below` in the config |
| Works on pipelines it didn't build | `adapter.py` |
| Same input, same verdict | `config.py`, the saved LLM judge results, and the repeat-run tests |
| Every run saved with code version and settings | `runlog.py` |
| Named failure modes with fixes | `taxonomy.py` |
| Scored on held-out cases | `split_cases()` in `evaluate.py` |
| Handles more than one broken stage | Partly. See below. |

## What still needs work

- **More than one broken stage.** The tool catches one broken stage plus a broken
  model. When two stages before the model are broken, it abstains. Handling that case
  means fixing stages one after another until the passage gets through, then removing
  each fix to see which ones were really needed.
- **Split `rag_pipeline` into stages.** Right now `ask()` runs retrieval and generation
  together, so the adapter copies a few small pieces of it. Those are marked `MIRRORS`.
  Once each stage is its own function, the adapter should call those instead, so the
  tool always checks the code that actually runs.
- **Set temperature 0 in the pipeline.** The adapter already does. `ask()` doesn't, so
  the two can give different answers.
- **Tune the confidence scores.** The numbers at the top of `diagnose.py` are starting
  guesses. Tune them on the tuning cases so that a verdict at 0.9 is right about 90%
  of the time, then measure once on the held-out cases.
- **Check the LLM judge.** Before trusting it, grade a sample by hand and see how often
  you agree with it.
- **Break retrieval into finer causes.** Right now, a bad embedding model, the index
  type, and query rewriting all show up as `retrieval`. Telling them apart means
  swapping one at a time, like rerunning retrieval with a second embedding model.
- **Add more pipeline options.** A reranker, hybrid search, and a second embedding
  model in `rag_pipeline` would give the reranking and retrieval checks real cases to
  work on.
- **Find answer passages for Open RAGBench.** Check what the dataset's labels point to.
  If they point to a whole section, you'll still need the exact sentence that supports
  each answer.
- **Grow the list of failure modes.** Add new ones as testing turns up failures the
  list doesn't cover.
