"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import ApiStatus from "@/components/ApiStatus";
import PipelineGraph from "@/components/PipelineGraph";
import { useApi } from "@/hooks/use-api";
import type { Artifact, TraceDetail } from "@/lib/api";

function ArtifactInspector({ executionId, id }: { executionId: string; id: string }) {
  const result = useApi<Artifact>(`traces/${encodeURIComponent(executionId)}/artifacts/${encodeURIComponent(id)}`);
  return <article className="mt-4 rounded-lg border border-slate-700 bg-slate-950 p-4">
    <h3 className="break-all font-mono text-xs text-cyan-400">{id}</h3>
    <ApiStatus loading={result.loading} error={result.error} />
    {result.data && <>
      <p className="mt-2 text-xs text-slate-400">Availability: {result.data.availability}</p>
      <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words text-xs leading-6 text-slate-300">{result.data.availability === "available" ? typeof result.data.content === "string" ? result.data.content: JSON.stringify(result.data.content, null, 2): "Evidence content is unavailable."}</pre>
    </>}
  </article>;
}

function diagnosisPending(trace: TraceDetail) {
  return ["queued", "running"].includes(trace.execution.diagnosis_state) ||
    (trace.execution.flagged && trace.execution.diagnosis_state === "not_requested");
}

export default function TraceDetailPage() {
  const { execution_id: executionId } = useParams<{ execution_id: string }>();
  const result = useApi<TraceDetail>(`traces/${encodeURIComponent(executionId)}`, {
    pollIntervalMs: 3000, pollWhile: diagnosisPending,
  });
  const [requesting, setRequesting] = useState(false);
  const [requestError, setRequestError] = useState("");

  async function requestDiagnosis() {
    if (requesting) return;
    setRequesting(true);
    setRequestError("");
    try {
      const response = await fetch(`/api/traces/${encodeURIComponent(executionId)}/diagnosis`, {
        method: "POST",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.message ?? "Could not request diagnosis.");
      result.refresh();
    } catch (error) {
      setRequestError(error instanceof Error ? error.message : "Could not request diagnosis.");
    } finally {
      setRequesting(false);
    }
  }
  const [selectedNodeId, setSelectedNodeId] = useState("");
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null);
  const trace = result.data;
  const selectedNode = trace?.graph.nodes.find((node) => node.node_id === selectedNodeId) ?? trace?.graph.nodes[0];
  const verdict = trace?.latest_live_diagnosis?.verdict;
  const artifactIds = selectedArtifactId ? [selectedArtifactId] : selectedNode?.artifact_ids ?? [];

  return <main className="min-h-screen bg-slate-950 p-6 text-slate-100">
    <div className="mx-auto max-w-7xl space-y-6">

      <Link href="/traces" className="text-sm text-cyan-400">← Back to trace explorer</Link>

      <ApiStatus loading={result.loading} error={result.error} />

      {trace && <>

        <header>
          <h1 className="mt-3 break-all text-2xl font-semibold">Execution {trace.execution.execution_id}</h1>
          <p className="mt-3 break-all font-mono text-xs text-slate-400">Trace ID: {trace.execution.trace_id ?? "Unavailable"}</p>
          <div className="mt-4 flex gap-5 text-sm text-cyan-400">
            <button onClick={result.refresh}>Refresh trace</button>
            <Link href={`/spans?execution_id=${encodeURIComponent(executionId)}`}>Inspect recorded spans →</Link>
          </div>
        </header>

        <section className="grid gap-4 rounded-xl border border-slate-800 bg-slate-900 p-5 sm:grid-cols-4">
          {[["Environment", trace.execution.environment ?? "Unknown"], ["Runtime status", trace.execution.status], ["Duration", trace.execution.duration_ms == null ? "Unknown" : `${trace.execution.duration_ms.toFixed(2)} ms`], ["Diagnosis", trace.execution.diagnosis_state.replaceAll("_", " ")]].map(([label, value]) => <div key={label}>
            <p className="text-xs text-slate-400">{label}</p>
            <p className="mt-2">{value}</p>
          </div>)}
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900 p-5">
          <h2 className="font-semibold">Observed stages</h2>
          {trace.graph.nodes.length > 0 ? <PipelineGraph graph={trace.graph} selectedNodeId={selectedNode?.node_id ?? ""} primaryStage={verdict?.primary_stage ?? ""} onSelectNode={(id) => { setSelectedNodeId(id); setSelectedArtifactId(null); }} /> : <p className="mt-4 text-slate-400">No supported stage spans were captured. Inspect the recorded spans for available metadata.</p>}
          <p className="mt-3 text-xs text-slate-400">Edges show captured parent relationships between stages. A completed stage can still contribute to a wrong answer.</p>
        </section>

        <div className="grid items-start gap-5 lg:grid-cols-2">

          <section className="min-w-0 rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="font-semibold">Artifact inspector</h2>
            <p className="mt-2 text-sm text-slate-400">Selected stage: {selectedNode?.label ?? "None captured"}</p>
            {selectedArtifactId && <button className="mt-3 text-sm text-cyan-400" onClick={() => setSelectedArtifactId(null)}>Show stage artifacts</button>}
            {artifactIds.length === 0 && <p className="mt-4 text-sm text-slate-400">No artifacts captured for this stage.</p>}
            {artifactIds.map((id) => <ArtifactInspector key={id} executionId={executionId} id={id} />)}
            {Boolean(trace.execution_artifact_ids?.length) && <h3 className="mt-5 font-semibold">Execution artifacts</h3>}
            {trace.execution_artifact_ids?.filter((id) => !artifactIds.includes(id)).map((id) => <ArtifactInspector key={id} executionId={executionId} id={id} />)}
          </section>

          <section className="rounded-xl border border-amber-400/30 bg-slate-900 p-5">
            <h2 className="font-semibold">Verdict inspector</h2>
            {["not_requested", "failed", "insufficient_evidence"].includes(trace.execution.diagnosis_state) && (
              <button onClick={() => void requestDiagnosis()} disabled={requesting}
                className="mt-4 rounded-lg bg-cyan-400 px-4 py-2 text-sm text-slate-950 disabled:opacity-50">
                {requesting ? "Requesting…" : trace.execution.diagnosis_state === "not_requested" ? "Run diagnosis" : "Retry diagnosis"}
              </button>
            )}
            {requestError && <p role="alert" className="mt-3 text-sm text-red-300">{requestError}</p>}
            {["queued", "running"].includes(trace.execution.diagnosis_state) && (
              <p role="status" className="mt-3 text-sm text-cyan-300">Diagnosis {trace.execution.diagnosis_state}. This view updates automatically.</p>
            )}
            {!verdict ? <p role="status" className="mt-4 text-sm text-slate-300">{trace.latest_live_diagnosis?.error?.message ?? (trace.execution.diagnosis_state === "not_requested" ? "No diagnostic run has been requested for this execution. Telemetry and runtime errors alone do not establish a diagnostic root cause.": `Diagnosis ${trace.execution.diagnosis_state.replaceAll("_", " ")}.`)}</p> : <>
              <p className="mt-4 text-amber-300">{verdict.status} · {verdict.mechanism ?? verdict.category ?? "No mechanism established"} · {verdict.primary_stage?.replaceAll("_", " ") ?? "No stage assigned"}</p>
              <p className="mt-3 text-sm text-slate-300">{verdict.summary}</p>
              <p className="mt-4 text-sm text-cyan-400">Evidence score: {verdict.confidence == null ? "Unavailable": `${(verdict.confidence * 100).toFixed(0)}%`}</p>
              <p className="mt-1 text-xs text-slate-400">Uncalibrated score; not a probability of correctness.</p>
              {verdict.evidence_trail.map((evidence, index) => <div key={index} className="mt-5 text-sm">
                <p>{evidence.finding}</p>
                {evidence.citations.map((citation, i) => <blockquote key={i} className="mt-3 border-l-2 border-cyan-400 pl-3">
                  <p>{citation.quote ?? "Quote withheld"}</p>
                  <button onClick={() => setSelectedArtifactId(citation.artifact_id)} className="mt-2 break-all text-left text-xs text-cyan-400 underline">Inspect {citation.artifact_id}</button>
                </blockquote>)}
              </div>)}
              {Boolean(verdict.evaluations?.length) && (
                <details className="mt-5 text-sm">
                  <summary className="cursor-pointer text-cyan-300">All mechanism checks</summary>
                  {verdict.evaluations?.map((evaluation) => (
                    <div key={evaluation.mechanism} className="mt-4 border-t border-slate-700 pt-3">
                      <p className="font-mono text-xs">{evaluation.mechanism} · {evaluation.status}</p>
                      <p className="mt-2 text-slate-300">{evaluation.rationale}</p>
                    </div>
                  ))}
                </details>
              )}
              {verdict.recommendations.map((item, i) => <p key={i} className="mt-4 text-sm text-slate-300">{item.action}</p>)}
            </>}
          </section>
        </div>
      </>}
    </div>
  </main>;
}
