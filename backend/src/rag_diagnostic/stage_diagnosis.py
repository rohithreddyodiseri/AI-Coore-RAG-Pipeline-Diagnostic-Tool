"""Evidence reconstruction and validated technical root-cause attribution."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

from rag_diagnostic.mechanisms import FAILURE_TAXONOMY, MECHANISM_STAGE, ROW_PROPERTIES, VERSION
from rag_diagnostic.runlog import RunRecord, append_jsonl


def reconstruct(trace):
    from rag_diagnostic.diagnostic_engine import trace_artifacts

    artifacts = trace_artifacts(trace)
    boundaries = {}
    if "execution_status" in trace:
        artifacts["boundary.execution"] = {
            "stage": "operational",
            "kind": "manifest",
            "text": json.dumps({"status": trace["execution_status"]}),
        }
    for stage, record in trace.get("stages", {}).items():
        # Vectors are already summarized by trace_artifacts; retain full data in source trace.
        if stage in ("document_embedding", "query_embedding"):
            record = {k: v for k, v in record.items() if k != "output_data"} | {
                "chunk_ids": [
                    item["chunk_id"]
                    for item in record.get("output_data", [])
                    if isinstance(item, dict) and "chunk_id" in item
                ]
            }
        key = f"boundary.{stage}"
        artifacts[key] = {
            "stage": stage,
            "kind": "boundary",
            "text": json.dumps(record, sort_keys=True, allow_nan=False),
        }
        boundaries[stage] = record
    if "output_constraints" in trace:
        artifacts["boundary.output_constraints"] = {
            "stage": "generation",
            "kind": "boundary",
            "text": json.dumps(trace["output_constraints"], sort_keys=True),
        }
    delivered = trace.get("stages", {}).get("generation", {}).get("input_data", {})
    if isinstance(delivered.get("delivered_context"), str):
        artifacts["generation.delivered_context"] = {
            "stage": "generation",
            "kind": "context",
            "text": delivered["delivered_context"],
        }
        artifacts["generation.delivery_manifest"] = {
            "stage": "generation",
            "kind": "manifest",
            "text": json.dumps({"characters": len(delivered["delivered_context"])}),
        }
    return artifacts, boundaries


def unknown(mechanism, reason):
    return dict(
        mechanism=mechanism,
        stage=MECHANISM_STAGE[mechanism],
        status="UNKNOWN",
        designation=None,
        confidence=0.0,
        evidence=[],
        contradicting_evidence=[],
        rationale=reason,
        unresolved_questions=[reason],
    )


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _output(trace, stage):
    return trace.get("stages", {}).get(stage, {}).get("output_data")


def technical_check(mechanism, trace, row, artifacts):
    """Verify observable prerequisites. Semantic relevance remains a judge hypothesis."""
    refs = {c["artifact_id"] for c in row["evidence"]}

    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    def cite(*keys):
        require(all(k in refs for k in keys), f"must cite boundaries: {keys}")

    fault = row["status"] == "FAULT"
    if mechanism in (
        "TOP_K_EXCLUSION",
        "RERANKING_DEGRADATION",
        "SIMILARITY_CUTOFF_EXCLUSION",
        "TOKEN_BUDGET_EXCLUSION",
    ):
        for name in ("retrieval", "reranking", "deep_retrieval"):
            candidates = trace.get(name) if name == "deep_retrieval" else _output(trace, name)
            if candidates is None:
                continue
            ids = [c.get("chunk_id") for c in candidates]
            require(
                all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)),
                "rank/selection attribution requires unique recorded chunk IDs",
            )
            require(
                all(c.get("score") is None or _finite(c["score"]) for c in candidates),
                "candidate scores must be finite numbers",
            )
    if mechanism == "MISSING_SOURCE_INFORMATION":
        cite("corpus.manifest", "query")
        require(artifacts["corpus.manifest"]["complete"], "original corpus view is incomplete")
    elif mechanism == "EXTRACTION_LOSS":
        cite("boundary.parsing")
        corpus = trace.get("corpus", {})
        parsed = _output(trace, "parsing")
        require(
            isinstance(parsed, list) and trace["stages"]["parsing"].get("complete") is True,
            "complete normalized document boundary required",
        )
        sources = [
            artifacts[c["artifact_id"]] | {"quote": c["quote"]}
            for c in row["evidence"]
            if c["artifact_id"].startswith("corpus.document.")
        ]
        require(bool(sources), "must cite original document content")
        for source in sources:
            texts = [
                p["text"]
                for p in parsed
                if p.get("doc_id", p.get("metadata", {}).get("doc_id")) == source["doc_id"]
            ]
            require(bool(texts), "source-to-normalized document identity is unobserved")
            if fault:
                require(
                    source["quote"] not in "\n".join(texts),
                    "cited source passage survived normalization",
                )
        require(bool(corpus.get("documents")), "original source unavailable")
    elif mechanism == "CHUNK_FRAGMENTATION":
        cite("boundary.chunking", "boundary.parsing")
        chunks, parsed = _output(trace, "chunking"), _output(trace, "parsing")
        require(
            isinstance(chunks, list) and isinstance(parsed, list),
            "normalized text and chunks required",
        )
        cited_ids = {artifacts[c["artifact_id"]].get("chunk_id") for c in row["evidence"]}
        selected = [c for c in chunks if c["chunk_id"] in cited_ids]
        require(len(selected) >= 2, "cite at least two neighboring chunks")
        require(
            len({c["chunk_id"] for c in selected}) == len(selected),
            "chunk lineage IDs are ambiguous",
        )
        for c in selected:
            meta = c.get("metadata", {}) | c
            matches = [
                p["text"]
                for p in parsed
                if p.get("doc_id", p.get("metadata", {}).get("doc_id")) == c["doc_id"]
                and p.get("metadata", {}).get("section") == meta.get("section")
            ]
            require(
                len(matches) == 1, "normalized document/section lineage is ambiguous or missing"
            )
            text = matches[0]
            start, end = meta.get("start_offset"), meta.get("end_offset")
            require(
                text is not None
                and type(start) is int
                and type(end) is int
                and 0 <= start < end <= len(text)
                and text[start:end] == c["text"],
                "invalid chunk lineage or offsets",
            )
        require(
            len({(c["doc_id"], c.get("metadata", {}).get("section")) for c in selected}) == 1,
            "chunks must belong to the same normalized section",
        )
        numbers = sorted(c.get("metadata", {}).get("chunk", -100) for c in selected)
        require(
            all(b == a + 1 for a, b in zip(numbers, numbers[1:], strict=False)),
            "neighboring chunk lineage required",
        )
    elif mechanism == "EMBEDDING_MISMATCH":
        cite("boundary.document_embedding", "boundary.query_embedding")
        stages = trace["stages"]
        doc = stages.get("document_embedding", {}).get("configuration", {})
        query = stages.get("query_embedding", {}).get("configuration", {})
        fields = ("model", "revision", "dimension", "preprocessing", "normalized")
        comparable = [k for k in fields if doc.get(k) is not None and query.get(k) is not None]
        mismatch = any(doc[k] != query[k] for k in comparable)
        require(
            mismatch if fault else all(k in comparable for k in fields) and not mismatch,
            "configuration mismatch unproven or compatibility evidence incomplete",
        )
    elif mechanism == "INDEXING_FAILURE":
        cite("boundary.indexing", "boundary.chunking", "boundary.document_embedding")
        index = _output(trace, "indexing") or {}
        require(
            index.get("complete") is True and isinstance(index.get("indexed_ids"), list),
            "complete acknowledged indexed ID set required",
        )
        chunks, embeddings = _output(trace, "chunking"), _output(trace, "document_embedding")
        require(
            isinstance(chunks, list) and isinstance(embeddings, list),
            "generated ID mappings required",
        )
        generated = {c["chunk_id"] for c in chunks}
        embedded = {c["chunk_id"] for c in embeddings}
        require(
            generated <= embedded, "embedding output incomplete; indexing cause is not isolated"
        )
        missing = generated - set(index["indexed_ids"])
        require(bool(missing) if fault else not missing, "index membership contradicts assessment")
    elif mechanism == "TOP_K_EXCLUSION":
        cite("retrieval.manifest")
        cutoff = artifacts["retrieval.manifest"]["cutoff"]
        require(type(cutoff) is int, "observed retrieval cutoff required")
        if fault:
            require(
                any(
                    k.startswith("deep.chunk.")
                    and artifacts[k]["rank"] > cutoff
                    and artifacts[k]["chunk_id"] not in artifacts["retrieval.manifest"]["chunk_ids"]
                    for k in refs
                ),
                "no cited deeper-ranked excluded indexed chunk",
            )
        else:
            require(
                any(k.startswith("deep.chunk.") for k in refs),
                "clearing top-K requires observed deeper evidence",
            )
    elif mechanism == "RERANKING_DEGRADATION":
        cite("retrieval.manifest", "reranking.manifest")
        if fault:
            before = [artifacts[k] for k in refs if k.startswith("retrieval.chunk.")]
            after = {
                a["chunk_id"]: a["rank"]
                for k, a in artifacts.items()
                if k.startswith("reranking.chunk.")
            }
            require(
                any(a["chunk_id"] not in after or after[a["chunk_id"]] > a["rank"] for a in before),
                "no cited candidate demotion or removal",
            )
    elif mechanism in ("SIMILARITY_CUTOFF_EXCLUSION", "TOKEN_BUDGET_EXCLUSION"):
        cite("boundary.selection")
        selection = _output(trace, "selection") or {}
        decisions = selection.get("decisions", [])
        reason = "similarity_threshold" if mechanism.startswith("SIMILARITY") else "token_budget"
        candidates = {
            a["chunk_id"]: a
            for k, a in artifacts.items()
            if k.startswith(("retrieval.chunk.", "reranking.chunk."))
        }
        exclusions = [
            d for d in decisions if d.get("selected") is False and d.get("reason") == reason
        ]
        for d in exclusions:
            require(
                d.get("chunk_id") in candidates, "selection decision names an unobserved candidate"
            )
            if reason == "similarity_threshold":
                require(
                    all(_finite(d.get(k)) for k in ("score", "threshold"))
                    and d["score"] == candidates[d["chunk_id"]]["score"]
                    and d["score"] < d["threshold"],
                    "score/threshold decision is inconsistent",
                )
            else:
                require(
                    all(
                        type(d.get(k)) is int and d[k] >= 0
                        for k in ("token_count", "used_tokens", "budget")
                    )
                    and d["used_tokens"] + d["token_count"] > d["budget"],
                    "token budget arithmetic is inconsistent",
                )
        if fault:
            cited = {artifacts[k].get("chunk_id") for k in refs}
            require(
                any(d["chunk_id"] in cited for d in exclusions),
                "cite relevant excluded candidate and decision",
            )
        else:
            require(
                selection.get("complete") is True
                and len(decisions) == len(candidates)
                and {d.get("chunk_id") for d in decisions} == set(candidates)
                and not exclusions,
                "complete candidate decisions required to clear exclusion",
            )
    elif mechanism == "CONTEXT_TRUNCATION":
        cite("boundary.context_assembly")
        context = trace["stages"]["context_assembly"]
        pre, post = context.get("pre_truncation"), context.get("post_truncation")
        require(
            isinstance(pre, str) and isinstance(post, str), "before/after truncation text required"
        )
        require(
            (pre != post and post in pre) if fault else pre == post,
            "truncation boundary contradicts claim",
        )
    elif mechanism == "CONTEXT_DELIVERY_LOSS":
        cite("context.text", "boundary.generation")
        require(
            "generation.delivered_context" in refs or "generation.delivery_manifest" in refs,
            "delivered context citation required",
        )
        assembled = artifacts["context.text"]["text"]
        delivered = artifacts["generation.delivered_context"]["text"]
        prompt = trace["stages"]["generation"]["input_data"].get("prompt")
        require(
            isinstance(prompt, str) and delivered in prompt,
            "delivered context not found in exact request",
        )
        require(
            assembled not in delivered if fault else assembled in delivered,
            "delivery boundary contradicts claim",
        )
    elif mechanism in ("INCORRECT_EVIDENCE_EXTRACTION", "PARTIAL_EVIDENCE_EXTRACTION"):
        cite("generation.delivered_context", "generation.answer", "boundary.generation")
        if mechanism.startswith("PARTIAL"):
            cite("query")
        delivered = artifacts["generation.delivered_context"]["text"]
        prompt = trace["stages"]["generation"]["input_data"].get("prompt")
        require(
            bool(delivered.strip()) and isinstance(prompt, str) and delivered in prompt,
            "actual delivered evidence is unavailable",
        )
    elif mechanism in ("OUTPUT_FORMAT_VIOLATION", "SPECIFICITY_MISMATCH"):
        cite("generation.answer")
        require(
            "query" in refs or "generation.instructions" in refs,
            "explicit request constraint citation required",
        )
        constraint = trace.get("output_constraints", {})
        if mechanism == "OUTPUT_FORMAT_VIOLATION":
            from rag_diagnostic.diagnostic_engine import _deterministic_checks

            observed = _deterministic_checks(artifacts).get("FP5")
            if observed is not None:
                require(
                    (observed["status"] == "fault") == fault,
                    "explicit JSON format check contradicts assessment",
                )
        if mechanism == "OUTPUT_FORMAT_VIOLATION" and constraint.get("type") == "json":
            cite("boundary.output_constraints")
            try:
                value = json.loads(artifacts["generation.answer"]["text"])
                valid = (
                    not constraint.get("required_keys")
                    or isinstance(value, dict)
                    and all(k in value for k in constraint["required_keys"])
                )
            except ValueError:
                valid = False
            require(not valid if fault else valid, "JSON output constraint contradicts assessment")
    elif mechanism == "UNCLASSIFIED_OPERATIONAL_FAILURE":
        if not fault:
            cite("boundary.execution")
            observed = trace.get("stages", {})
            require(
                trace.get("execution_status") == "completed"
                and all(
                    observed.get(s, {}).get("status") == "completed"
                    for s in ("retrieval", "generation")
                )
                and not any(
                    s.get("status") == "failed" or s.get("error") for s in observed.values()
                ),
                "terminal successful boundaries required to clear operational failure",
            )
            return
        require(
            any(
                k.startswith("boundary.")
                and trace["stages"].get(k[9:], {}).get("status") == "failed"
                and trace["stages"][k[9:]].get("error")
                for k in refs
            ),
            "observed stage exception required",
        )


def validate(raw, artifacts, trace, threshold):
    from rag_diagnostic.diagnostic_engine import _citations

    if not isinstance(raw, dict) or set(raw) != {"evaluations"}:
        raise ValueError("invalid stage assessment schema")
    rows = raw["evaluations"]
    if (
        not isinstance(rows, list)
        or len(rows) != len(MECHANISM_STAGE)
        or any(not isinstance(r, dict) for r in rows)
        or sorted(r.get("mechanism", "") for r in rows) != sorted(MECHANISM_STAGE)
    ):
        raise ValueError("every technical mechanism must be assessed exactly once")
    results = []
    for mechanism in MECHANISM_STAGE:
        row = next(r for r in rows if r["mechanism"] == mechanism)
        try:
            if (
                set(row) != set(ROW_PROPERTIES)
                or row["stage"] != MECHANISM_STAGE[mechanism]
                or row["status"] not in ("FAULT", "CLEAR", "UNKNOWN", "NOT_APPLICABLE")
                or row["designation"] not in ("primary", "contributing", None)
                or not _finite(row["confidence"])
                or not 0 <= row["confidence"] <= 1
                or not isinstance(row["rationale"], str)
                or not row["rationale"].strip()
                or not isinstance(row["unresolved_questions"], list)
                or not all(isinstance(q, str) for q in row["unresolved_questions"])
            ):
                raise ValueError("invalid mechanism assessment")
            if row["status"] == "UNKNOWN":
                results.append(unknown(mechanism, row["rationale"]))
                continue
            if row["confidence"] < threshold:
                raise ValueError("below confidence threshold")
            row = {
                **row,
                "evidence": _citations(row["evidence"], artifacts),
                "contradicting_evidence": _citations(row["contradicting_evidence"], artifacts)
                if row["contradicting_evidence"]
                else [],
            }
            if row["status"] != "FAULT" and row["designation"] is not None:
                raise ValueError("non-fault cannot have causal designation")
            if row["contradicting_evidence"] and row["status"] == "FAULT":
                raise ValueError("contradictory evidence requires resolution before attribution")
            if mechanism == "CHUNK_FRAGMENTATION" and row["status"] == "FAULT":
                # Resolve structural lineage behind already cited chunks. These
                # records prove offsets/identity; they never establish semantic
                # fragmentation without a judge fault hypothesis.
                refs = {c["artifact_id"] for c in row["evidence"]}
                cited_chunks = [artifacts[k] for k in refs if k.startswith("chunking.chunk.")]
                if len(cited_chunks) >= 2:
                    additions = [
                        {"artifact_id": key, "quote": artifacts[key]["text"]}
                        for key in ("boundary.chunking", "boundary.parsing")
                        if key in artifacts and key not in refs
                    ]
                    row["evidence"] = [*row["evidence"], *additions]
                    # Reject rather than repair broken or ambiguous provenance.
                    technical_check(mechanism, trace, row, artifacts)
                    if additions:
                        row["derived_evidence"] = additions
                        row["validation_notes"] = [
                            "Python attached recorded parsing/chunking boundaries after "
                            "verifying cited chunk IDs, section lineage and exact offsets."
                        ]
            unused = trace.get("stages", {}).get("reranking", {}).get("status") == "not_used"
            if mechanism == "RERANKING_DEGRADATION" and unused:
                if row["status"] != "NOT_APPLICABLE":
                    raise ValueError("unused reranker must be NOT_APPLICABLE")
            elif row["status"] == "NOT_APPLICABLE":
                names = {
                    "normalization": "parsing",
                    "embedding": "document_embedding",
                    "context": "context_assembly",
                }
                stage = names.get(row["stage"], row["stage"])
                if trace.get("stages", {}).get(stage, {}).get("status") != "not_used":
                    raise ValueError("not-applicable requires explicitly unused stage")
            else:
                technical_check(mechanism, trace, row, artifacts)
            row["technical_stage"] = row["stage"]
            if mechanism == "UNCLASSIFIED_OPERATIONAL_FAILURE" and row["status"] == "FAULT":
                row["technical_stage"] = next(
                    c["artifact_id"][9:]
                    for c in row["evidence"]
                    if c["artifact_id"].startswith("boundary.")
                    and trace["stages"].get(c["artifact_id"][9:], {}).get("status") == "failed"
                )
            results.append(row)
        except (ValueError, TypeError, KeyError) as error:
            results.append(unknown(mechanism, f"Abstained: {error}."))

    # Machine-checkable output constraints take precedence over the judge's
    # semantic hypothesis. Exact parsing can also repair a judge row whose
    # citations failed validation.
    from rag_diagnostic.diagnostic_engine import _deterministic_checks

    format_check = _deterministic_checks(artifacts).get("FP5")
    if format_check is not None:
        replacement = {
            "mechanism": "OUTPUT_FORMAT_VIOLATION",
            "stage": "generation",
            "technical_stage": "generation",
            "status": "FAULT" if format_check["status"] == "fault" else "CLEAR",
            "designation": None,
            "confidence": 1.0,
            "evidence": format_check["evidence"],
            "contradicting_evidence": [],
            "rationale": format_check["rationale"],
            "unresolved_questions": [],
        }
        index = next(
            index
            for index, row in enumerate(results)
            if row["mechanism"] == "OUTPUT_FORMAT_VIOLATION"
        )
        results[index] = replacement

    order = [
        "corpus",
        "normalization",
        "parsing",
        "chunking",
        "embedding",
        "document_embedding",
        "indexing",
        "query_embedding",
        "retrieval",
        "reranking",
        "selection",
        "context",
        "context_assembly",
        "generation",
        "operational",
    ]
    faults = sorted(
        (r for r in results if r["status"] == "FAULT"),
        key=lambda r: (
            order.index(r["technical_stage"]) if r["technical_stage"] in order else len(order)
        ),
    )
    # Rows are already in pipeline/taxonomy order; never trust model primary selection.
    for i, row in enumerate(faults):
        row["designation"] = "primary" if i == 0 else "contributing"
    return results


@dataclass
class StageVerdict:
    run_id: str
    query: str
    status: str
    root_cause: dict | None
    contributing_failures: list
    evaluations: list
    unresolved_issues: list
    observations: dict
    run: RunRecord
    answer: str
    schema_version: str = "2.0"

    @property
    def confidence(self):
        return self.root_cause["confidence"] if self.root_cause else 0.0

    @property
    def stages(self):
        return tuple(
            dict.fromkeys(r["technical_stage"] for r in self.evaluations if r["status"] == "FAULT")
        )

    @property
    def failure_categories(self):
        """Compatibility for offline scoring: these are technical mechanism labels."""
        return tuple(r["mechanism"] for r in self.evaluations if r["status"] == "FAULT")

    @property
    def suspects(self):
        return ()

    @property
    def primary_fault_stage(self):
        return self.root_cause["technical_stage"] if self.root_cause else None

    def to_dict(self, include_run=True):
        result = asdict(self)
        result.update(
            primary_fault_stage=self.primary_fault_stage,
            failure_mechanism=self.root_cause["mechanism"] if self.root_cause else None,
            confidence=self.root_cause["confidence"] if self.root_cause else 0.0,
            attribution_complete=not self.unresolved_issues,
        )
        if not include_run:
            result.pop("run")
        return result

    def render(self):
        root = self.root_cause
        return "\n".join(
            [
                f"Run: {self.run_id}",
                f"Verdict: {self.status}",
                f"Root cause: {root['mechanism']} at {root['stage']} ({root['confidence']:.2f})"
                if root
                else "Root cause: unresolved",
                *[f"{r['mechanism']}: {r['status']} — {r['rationale']}" for r in self.evaluations],
            ]
        )


def diagnose_stages(engine, trace, run_id=None, pipeline=None, run=None):
    pipeline = pipeline or trace.get("pipeline", {"adapter": "captured_trace"})
    run = run or RunRecord.capture(engine.config, SimpleNamespace(describe=lambda: pipeline))
    try:
        from rag_diagnostic.diagnostic_engine import _reject_oracles

        _reject_oracles(pipeline)
        artifacts, boundaries = reconstruct(engine._judge_view(trace))
        source = hashlib.sha256()
        root = Path(__file__).parent.parent
        for path in sorted(root.rglob("*.py")):
            source.update(str(path.relative_to(root)).encode())
            source.update(path.read_bytes())
        payload = dict(
            artifacts=artifacts,
            boundaries=boundaries,
            taxonomy=FAILURE_TAXONOMY,
            taxonomy_version=VERSION,
            config=engine.config.to_dict(),
            code_revision=run.code_revision,
            source_sha256=source.hexdigest(),
            pipeline=pipeline,
        )
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        observations = dict(
            input=payload,
            trace=trace,
            judge=engine.judge.describe(),
            input_fingerprint=hashlib.sha256(encoded.encode()).hexdigest(),
        )
        rows = [unknown(m, "Judge has not established a verdict.") for m in MECHANISM_STAGE]
        if len(encoded) <= engine.config.max_judge_input_chars:
            response = engine.judge.evaluate_trace(payload)
            observations["judge_response"] = asdict(response)
            try:
                if response.completion_status != "completed":
                    raise ValueError("incomplete/refused judge response")
                rows = validate(
                    json.loads(response.raw_output),
                    artifacts,
                    engine._judge_view(trace),
                    engine.config.abstain_below,
                )
            except (ValueError, TypeError, KeyError) as error:
                rows = [unknown(m, str(error)) for m in MECHANISM_STAGE]
        else:
            rows = [unknown(m, "max_judge_input_chars exceeded") for m in MECHANISM_STAGE]
        faults = sorted(
            (r for r in rows if r["status"] == "FAULT"), key=lambda r: r["designation"] != "primary"
        )
        unresolved = [
            dict(mechanism=r["mechanism"], questions=r["unresolved_questions"])
            for r in rows
            if r["status"] == "UNKNOWN"
        ]
        verdict = StageVerdict(
            run_id or trace.get("run_id", observations["input_fingerprint"][:12]),
            trace["query"],
            "diagnosed" if faults else "abstained" if unresolved else "no_failure",
            faults[0] if faults else None,
            faults[1:],
            rows,
            unresolved,
            observations,
            run,
            trace.get("final_answer", trace.get("answer", "")),
        )
        append_jsonl(Path(engine.config.runs_dir) / "diagnoses.jsonl", verdict.to_dict())
        from rag_diagnostic.judge_logging import log_validated_verdict

        log_validated_verdict(verdict.to_dict())
        return verdict
    except Exception as error:
        engine.record_error(trace, run, error, run_id)
        raise
