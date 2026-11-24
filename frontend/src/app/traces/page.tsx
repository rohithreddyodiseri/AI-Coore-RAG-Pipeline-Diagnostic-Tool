"use client";

import Link from "next/link";
import { useState } from "react";
import PipelineGraph from "@/components/PipelineGraph";
import examples from "@/data/ui-api.examples.json";

const traces = [examples.TraceDetail];

type Artifact = {
  artifact_id: string;
  kind: string;
  availability: string;
  content: unknown;
};

const artifacts = examples.artifacts_by_id as Record<string, Artifact>;

export default function TraceExplorerPage() {
  const [search, setSearch] = useState("");
  const [environment, setEnvironment] = useState("all");
  const [status, setStatus] = useState("all");
  const [flagged, setFlagged] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedNodeId, setSelectedNodeId] = useState("");

  const filteredTraces = traces.filter(({ execution }) => {
    const searchable = [
      execution.execution_id,
      execution.trace_id,
      execution.app_id,
    ]
      .join(" ")
      .toLowerCase();

    return (
      searchable.includes(search.trim().toLowerCase()) &&
      (environment === "all" || execution.environment === environment) &&
      (status === "all" || execution.status === status) &&
      (flagged === "all" ||
        execution.flagged === (flagged === "flagged"))
    );
  });

  const selectedTrace = filteredTraces.find(
    ({ execution }) => execution.execution_id === selectedId
  );

  const selectedNode =
    selectedTrace?.graph.nodes.find(
      (node) => node.node_id === selectedNodeId
    ) ?? selectedTrace?.graph.nodes[0];

  const verdict = selectedTrace?.latest_live_diagnosis.verdict;

  function resetFilters() {
    setSearch("");
    setEnvironment("all");
    setStatus("all");
    setFlagged("all");
  }

  const fieldClass =
    "mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-cyan-400";

  return (
    <main className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <div className="mx-auto max-w-7xl space-y-6">
        <Link href="/" className="text-sm text-cyan-400">
          ← Back to dashboard
        </Link>

        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="mt-4 text-3xl font-semibold">Trace explorer</h1>
            <p className="mt-2 text-sm text-slate-400">
              Search executions and inspect their stages and evidence.
            </p>
          </div>

          <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-4 py-2 text-sm text-amber-300">
            Demo data · 1 supplied trace
          </span>
        </header>

        <section className="grid gap-4 rounded-xl border border-slate-800 bg-slate-900 p-5 sm:grid-cols-2 lg:grid-cols-4">
          <label className="text-sm">
            Search execution, trace, or app ID
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="exec_42"
              className={fieldClass}
            />
          </label>

          <label className="text-sm">
            Environment
            <select
              value={environment}
              onChange={(event) => setEnvironment(event.target.value)}
              className={fieldClass}
            >
              <option value="all">All environments</option>
              <option value="staging">Staging</option>
              <option value="production">Production</option>
            </select>
          </label>

          <label className="text-sm">
            Runtime status
            <select
              value={status}
              onChange={(event) => setStatus(event.target.value)}
              className={fieldClass}
            >
              <option value="all">All statuses</option>
              {Array.from(
                new Set(traces.map((trace) => trace.execution.status))
              ).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>

          <label className="text-sm">
            Diagnostic flag
            <select
              value={flagged}
              onChange={(event) => setFlagged(event.target.value)}
              className={fieldClass}
            >
              <option value="all">All executions</option>
              <option value="flagged">Flagged</option>
              <option value="unflagged">Not flagged</option>
            </select>
          </label>
        </section>

        <div className="flex items-center justify-between">
          <p role="status" className="text-sm text-slate-400">
            {filteredTraces.length} matching execution(s)
          </p>

          <button
            type="button"
            onClick={resetFilters}
            className="text-sm text-cyan-400 hover:underline"
          >
            Clear filters
          </button>
        </div>

        <section className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-slate-800 text-slate-400">
              <tr>
                <th className="p-4">Execution</th>
                <th className="p-4">App</th>
                <th className="p-4">Environment</th>
                <th className="p-4">Duration</th>
                <th className="p-4">Runtime</th>
                <th className="p-4">Diagnosis</th>
              </tr>
            </thead>

            <tbody>
              {filteredTraces.map(({ execution }) => (
                <tr
                  key={execution.execution_id}
                  className={`border-b border-slate-800 last:border-0 ${
                    selectedId === execution.execution_id
                      ? "bg-cyan-400/10"
                      : ""
                  }`}
                >
                  <td className="p-4">
                    <button
                      type="button"
                      aria-pressed={selectedId === execution.execution_id}
                      onClick={() => {
                        setSelectedId(execution.execution_id);
                        setSelectedNodeId("");
                      }}
                      className="font-mono text-cyan-400 hover:underline"
                    >
                      {execution.execution_id}
                    </button>
                  </td>
                  <td className="p-4">{execution.app_id}</td>
                  <td className="p-4">{execution.environment}</td>
                  <td className="p-4">{execution.duration_ms} ms</td>
                  <td className="p-4">{execution.status}</td>
                  <td className="p-4">
                    {execution.diagnosis_state}
                    {execution.flagged && (
                      <span className="ml-2 text-xs text-amber-300">
                        Flagged
                      </span>
                    )}
                  </td>
                </tr>
              ))}

              {filteredTraces.length === 0 && (
                <tr>
                  <td colSpan={6} className="p-8 text-center text-slate-400">
                    No executions match these filters. Try clearing them.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>

        {!selectedTrace && filteredTraces.length > 0 && (
          <p className="rounded-xl border border-slate-800 p-6 text-sm text-slate-400">
            Select an execution ID above to inspect its details.
          </p>
        )}

        {selectedTrace && verdict && selectedNode && (
          <>
            <section className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <h2 className="text-lg font-semibold">
                Execution {selectedTrace.execution.execution_id}
              </h2>

              <p className="mt-2 break-all font-mono text-xs text-slate-400">
                Trace ID: {selectedTrace.execution.trace_id}
              </p>

              <p className="mt-2 text-xs text-slate-400">
                Started: {selectedTrace.execution.started_at}
              </p>

              <PipelineGraph
                graph={selectedTrace.graph}
                selectedNodeId={selectedNode.node_id}
                primaryStage={verdict.primary_stage}
                onSelectNode={setSelectedNodeId}
              />
            </section>

            <div className="grid gap-5 lg:grid-cols-2">
              <section className="min-w-0 rounded-xl border border-slate-800 bg-slate-900 p-5">
                <h2 className="font-semibold">{selectedNode.label}</h2>

                <p className="mt-2 text-sm text-slate-400">
                  Runtime: {selectedNode.status} · {selectedNode.duration_ms} ms
                </p>

                {selectedNode.artifact_ids.length === 0 && (
                  <p className="mt-4 text-sm text-slate-400">
                    No artifacts were supplied for this stage.
                  </p>
                )}

                {selectedNode.artifact_ids.map((id) => {
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

                      <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap break-words text-xs leading-6 text-slate-300">
                        {artifact?.availability === "available"
                          ? typeof artifact.content === "string"
                            ? artifact.content
                            : JSON.stringify(artifact.content, null, 2)
                          : "Artifact content is unavailable."}
                      </pre>
                    </div>
                  );
                })}
              </section>

              <section className="rounded-xl border border-amber-400/30 bg-slate-900 p-5">
                <h2 className="font-semibold">Diagnostic verdict</h2>

                <p className="mt-3 text-sm text-amber-300">
                  {verdict.category} ·{" "}
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

                <h3 className="mt-5 text-sm font-semibold">Evidence</h3>

                {verdict.evidence_trail.map((evidence, index) => (
                  <div key={index} className="mt-3">
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
                        <p className="mt-1 break-all font-mono text-xs text-slate-500">
                          {citation.artifact_id}
                        </p>
                      </blockquote>
                    ))}
                  </div>
                ))}

                <h3 className="mt-5 text-sm font-semibold">
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
              </section>
            </div>
          </>
        )}
      </div>
    </main>
  );
}