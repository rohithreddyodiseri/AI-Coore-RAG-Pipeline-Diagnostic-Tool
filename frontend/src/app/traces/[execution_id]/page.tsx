"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import PipelineGraph from "@/components/PipelineGraph";
import examples from "@/data/ui-api.examples.json";

const trace = examples.TraceDetail;

type Artifact = {
  artifact_id: string;
  kind: string;
  availability: string;
  content: unknown;
};

type DiagnosisState =
  | "completed"
  | "pending"
  | "failed"
  | "insufficient_evidence";

const artifacts = examples.artifacts_by_id as Record<string, Artifact>;

const stateMessages: Record<
  Exclude<DiagnosisState, "completed">,
  { title: string; message: string }
> = {
  pending: {
    title: "Diagnosis pending",
    message:
      "The diagnostic result is not ready. Recorded stages and artifacts remain available to inspect.",
  },
  failed: {
    title: "Diagnosis failed",
    message:
      "The diagnostic job did not complete. This does not establish a fault in the RAG pipeline. No diagnostic error details were supplied in this preview.",
  },
  insufficient_evidence: {
    title: "Insufficient evidence",
    message:
      "The available evidence does not support a reliable failure-stage attribution. No stage or confidence score is assigned in this preview.",
  },
};

export default function TraceDetailPage() {
  const params = useParams<{ execution_id: string }>();

  const [selectedNodeId, setSelectedNodeId] = useState(
    trace.graph.nodes[0].node_id
  );
  const [selectedArtifactId, setSelectedArtifactId] = useState<
    string | null
  >(null);
  const [diagnosisState, setDiagnosisState] =
    useState<DiagnosisState>("completed");

  if (params.execution_id !== trace.execution.execution_id) {
    return (
      <main className="min-h-screen bg-slate-950 p-8 text-slate-100">
        <h1 className="text-2xl font-semibold">Trace not found</h1>
        <p className="mt-3 text-slate-400">
          This demo only contains execution exec_42.
        </p>
        <Link
          href="/traces"
          className="mt-5 inline-block text-cyan-400"
        >
          Back to trace explorer
        </Link>
      </main>
    );
  }

  const verdict =
    diagnosisState === "completed"
      ? trace.latest_live_diagnosis.verdict
      : null;

  const selectedNode =
    trace.graph.nodes.find((node) => node.node_id === selectedNodeId) ??
    trace.graph.nodes[0];

  const visibleArtifactIds = selectedArtifactId
    ? [selectedArtifactId]
    : selectedNode.artifact_ids;

  const stateMessage =
    diagnosisState !== "completed"
      ? stateMessages[diagnosisState]
      : null;

  function selectNode(nodeId: string) {
    setSelectedNodeId(nodeId);
    setSelectedArtifactId(null);
  }

  function inspectEvidence(artifactId: string) {
    const owner = trace.graph.nodes.find((node) =>
      node.artifact_ids.some((id) => id === artifactId)
    );

    if (owner) {
      setSelectedNodeId(owner.node_id);
    }

    setSelectedArtifactId(artifactId);
  }

  return (
    <main className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <div className="mx-auto max-w-7xl space-y-6">
        <Link href="/traces" className="text-sm text-cyan-400">
          ← Back to trace explorer
        </Link>

        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="mt-3 text-3xl font-semibold">
              Execution {trace.execution.execution_id}
            </h1>
            <p className="mt-3 break-all font-mono text-xs text-slate-400">
              Trace ID: {trace.execution.trace_id}
            </p>
          </div>

          <span className="rounded-full border border-amber-400/30 px-4 py-2 text-sm text-amber-300">
            Demo data
          </span>
        </header>

        <section className="rounded-xl border border-slate-800 bg-slate-900 p-5">
          <label
            htmlFor="diagnosis-preview"
            className="text-sm font-medium"
          >
            Demo diagnosis state
          </label>

          <select
            id="diagnosis-preview"
            value={diagnosisState}
            onChange={(event) => {
              setDiagnosisState(event.target.value as DiagnosisState);
              setSelectedArtifactId(null);
            }}
            className="mt-3 block w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm sm:max-w-sm"
          >
            <option value="completed">Completed — supplied verdict</option>
            <option value="pending">Pending — UI preview</option>
            <option value="failed">Failed — UI preview</option>
            <option value="insufficient_evidence">
              Insufficient evidence — UI preview
            </option>
          </select>

          <p className="mt-3 text-xs text-slate-400">
            Only the completed verdict comes from the supplied fixture.
            Other options preview display states using the same recorded
            trace. No diagnostic job is started.
          </p>
        </section>

        <section className="grid gap-4 rounded-xl border border-slate-800 bg-slate-900 p-5 sm:grid-cols-4">
          <div>
            <p className="text-xs text-slate-400">Environment</p>
            <p className="mt-2">{trace.execution.environment}</p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Runtime status</p>
            <p className="mt-2">{trace.execution.status}</p>
          </div>
          <div>
            <p className="text-xs text-slate-400">Total duration</p>
            <p className="mt-2">{trace.execution.duration_ms} ms</p>
          </div>
          <div>
            <p className="text-xs text-slate-400">
              {diagnosisState === "completed"
                ? "Diagnosis"
                : "Diagnosis preview"}
            </p>
            <p className="mt-2">
              {diagnosisState.replaceAll("_", " ")}
              {trace.execution.flagged ? " · Flagged" : ""}
            </p>
          </div>
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900 p-5">
          <h2 className="font-semibold">Observed stage dependencies</h2>

          <PipelineGraph
            graph={trace.graph}
            selectedNodeId={selectedNodeId}
            primaryStage={verdict?.primary_stage ?? ""}
            onSelectNode={selectNode}
          />

          <p className="mt-3 text-xs text-slate-400">
            The fixture supplies stage dependencies and durations, but no
            parent-span tree, per-stage timestamps, or exception details.
          </p>
        </section>

        <div className="grid items-start gap-5 lg:grid-cols-2">
          <section
            aria-live="polite"
            className="min-w-0 rounded-xl border border-slate-800 bg-slate-900 p-5"
          >
            <h2 className="font-semibold">Artifact inspector</h2>

            <p className="mt-2 text-sm text-slate-400">
              Selected stage: {selectedNode.label}
            </p>

            <p className="mt-2 text-xs text-slate-400">
              Runtime: {selectedNode.status} · {selectedNode.duration_ms} ms
            </p>

            {selectedArtifactId && (
              <button
                type="button"
                onClick={() => setSelectedArtifactId(null)}
                className="mt-4 text-sm text-cyan-400 hover:underline"
              >
                Show all artifacts for this stage
              </button>
            )}

            {visibleArtifactIds.length === 0 && (
              <p className="mt-5 text-sm text-slate-400">
                No artifacts were supplied for this stage.
              </p>
            )}

            {visibleArtifactIds.map((id) => {
              const artifact = artifacts[id];

              return (
                <div
                  key={id}
                  className="mt-4 rounded-lg border border-slate-700 bg-slate-950 p-4"
                >
                  <h3 className="break-all font-mono text-xs text-cyan-400">
                    {id}
                  </h3>

                  <p className="mt-2 text-xs text-slate-400">
                    Availability: {artifact?.availability ?? "unknown"}
                  </p>

                  <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words text-xs leading-6 text-slate-300">
                    {artifact?.availability === "available"
                      ? typeof artifact.content === "string"
                        ? artifact.content
                        : JSON.stringify(artifact.content, null, 2)
                      : "Evidence content is unavailable."}
                  </pre>
                </div>
              );
            })}
          </section>

          <section
            aria-live="polite"
            className="rounded-xl border border-amber-400/30 bg-slate-900 p-5"
          >
            <h2 className="font-semibold">Verdict inspector</h2>

            {stateMessage && (
              <div
                role="status"
                className={`mt-4 rounded-lg border p-4 ${
                  diagnosisState === "failed"
                    ? "border-red-400/30 bg-red-400/10"
                    : diagnosisState === "pending"
                      ? "border-cyan-400/30 bg-cyan-400/10"
                      : "border-amber-400/30 bg-amber-400/10"
                }`}
              >
                <h3 className="font-medium">{stateMessage.title}</h3>
                <p className="mt-3 text-sm leading-6 text-slate-300">
                  {stateMessage.message}
                </p>
                <p className="mt-3 text-xs text-slate-400">
                  UI preview only.
                </p>
              </div>
            )}

            {verdict && (
              <>
                <p className="mt-3 text-sm text-amber-300">
                  {verdict.status} · {verdict.category} ·{" "}
                  {verdict.primary_stage.replaceAll("_", " ")}
                </p>

                <p className="mt-3 text-sm leading-6 text-slate-300">
                  {verdict.summary}
                </p>

                <p className="mt-4 text-sm text-cyan-400">
                  Evidence score: {(verdict.confidence * 100).toFixed(0)}%
                </p>

                <p className="mt-1 text-xs text-slate-400">
                  Uncalibrated score; not a probability of correctness.
                </p>

                <h3 className="mt-6 text-sm font-semibold">
                  Supporting evidence
                </h3>

                {verdict.evidence_trail.length === 0 && (
                  <p className="mt-3 text-sm text-amber-300">
                    No supporting evidence was supplied.
                  </p>
                )}

                {verdict.evidence_trail.map((evidence, index) => (
                  <div key={index} className="mt-4">
                    <p className="text-sm leading-6 text-slate-300">
                      {evidence.finding}
                    </p>

                    {evidence.citations.map((citation, citationIndex) => (
                      <blockquote
                        key={`${citation.artifact_id}-${citationIndex}`}
                        className="mt-3 border-l-2 border-cyan-400 pl-3"
                      >
                        <p className="text-sm text-slate-300">
                          “{citation.quote}”
                        </p>

                        <button
                          type="button"
                          onClick={() =>
                            inspectEvidence(citation.artifact_id)
                          }
                          className="mt-2 break-all text-left font-mono text-xs text-cyan-400 underline"
                        >
                          Inspect {citation.artifact_id}
                        </button>
                      </blockquote>
                    ))}
                  </div>
                ))}

                <h3 className="mt-6 text-sm font-semibold">
                  Suggested next steps
                </h3>

                {verdict.recommendations.map((recommendation, index) => (
                  <p
                    key={index}
                    className="mt-3 text-sm leading-6 text-slate-300"
                  >
                    {recommendation.action}
                  </p>
                ))}

                <p className="mt-5 text-xs text-amber-300">
                  Unresolved categories:{" "}
                  {verdict.unresolved_categories.join(", ") || "None"}
                </p>
              </>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}