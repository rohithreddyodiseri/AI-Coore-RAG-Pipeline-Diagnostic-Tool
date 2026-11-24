"use client";
import Link from "next/link";
import { useState } from "react";
import PipelineGraph from "@/components/PipelineGraph";
import examples from "@/data/ui-api.examples.json";

const trace = examples.TraceDetail;
const verdict = trace.latest_live_diagnosis.verdict;
const totals = examples.Dashboard.totals;
const benchmark = examples.BenchmarkReport;

const artifacts = examples.artifacts_by_id as Record<
  string,
  {
    artifact_id: string;
    kind: string;
    availability: string;
    content: unknown;
  }
>;

function formatContent(content: unknown) {
  return typeof content === "string"
    ? content
    : JSON.stringify(content, null, 2);
}

export default function Home() {
  const [selectedNodeId, setSelectedNodeId] = useState(
    trace.graph.nodes[0].node_id
  );

  const selectedNode =
    trace.graph.nodes.find((node) => node.node_id === selectedNodeId) ??
    trace.graph.nodes[0];

  const metrics = [
    { label: "Observed traces", value: totals.observed_traces },
    { label: "Flagged traces", value: totals.flagged_traces },
    { label: "Runtime failures", value: totals.runtime_failed_traces },
    {
      label: "Abstention rate",
      value: `${(totals.abstention_rate * 100).toFixed(1)}%`,
    },
  ];

  const benchmarkMetrics = [
    { label: "Evaluated", value: benchmark.evaluated },
    { label: "Abstained", value: benchmark.abstained },
    {
      label: "Overall accuracy",
      value: `${(benchmark.accuracy_overall * 100).toFixed(1)}%`,
    },
    {
      label: "Accuracy when answered",
      value: `${(benchmark.accuracy_when_answered * 100).toFixed(1)}%`,
    },
  ];

  return (
    <main className="min-h-screen bg-slate-950 p-5 text-slate-100 md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-widest text-cyan-400">
              RAG Diagnostic Lab
            </p>

            <h1 className="mt-2 text-3xl font-semibold">
              Diagnostic Control Center
            </h1>

            <p className="mt-2 text-sm text-slate-400">
              Inspect executions, stage outputs, and diagnostic evidence.
            </p>
          </div>

          <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-4 py-2 text-sm text-amber-300">
            Demo data
          </span>

          <Link
            href="/apps/new"
            className="rounded-lg bg-cyan-400 px-4 py-2 text-sm font-medium text-slate-950 hover:bg-cyan-300"
          >
            Connect application
          </Link>

          <Link
            href="/apps"
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-cyan-400 hover:border-cyan-400"
          >
            View applications
          </Link>
        </header>

        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {metrics.map((metric) => (
            <div
              key={metric.label}
              className="rounded-xl border border-slate-800 bg-slate-900 p-5"
            >
              <p className="text-sm text-slate-400">{metric.label}</p>
              <p className="mt-3 text-3xl font-semibold">{metric.value}</p>
            </div>
          ))}
        </section>

        <p className="text-xs text-slate-400">
          Metrics describe the sample reporting window. The supplied file
          includes one detailed execution to inspect.
        </p>

        <div className="grid gap-5 lg:grid-cols-12">
          <aside className="rounded-xl border border-slate-800 bg-slate-900 p-5 lg:col-span-3">
            <h2 className="font-semibold">Recorded traces</h2>

            <div className="mt-4 rounded-lg border border-cyan-500/50 bg-cyan-500/10 p-4">
              <p className="font-mono text-sm text-cyan-300">
                {trace.execution.execution_id}
              </p>

              <p className="mt-2 text-sm">
                {trace.execution.environment}
              </p>

              <p className="mt-1 text-xs text-slate-400">
                {trace.execution.duration_ms} ms · {trace.execution.status}
              </p>

              <span className="mt-3 inline-block rounded bg-amber-400/10 px-2 py-1 text-xs text-amber-300">
                Flagged for diagnosis
              </span>
            </div>
          </aside>

          <section className="min-w-0 space-y-5 lg:col-span-9">
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <h2 className="font-semibold">Pipeline stages</h2>

              <p className="mt-1 text-sm text-slate-400">
                Select a stage to inspect its recorded artifacts.
              </p>

              <PipelineGraph
                graph={trace.graph}
                selectedNodeId={selectedNodeId}
                primaryStage={verdict.primary_stage}
                onSelectNode={setSelectedNodeId}
              />

              <p className="mt-4 text-xs text-slate-400">
                A stage can finish successfully and still contribute to a
                wrong answer.
              </p>
            </div>

            <div className="grid gap-5 xl:grid-cols-2">
              <section className="min-w-0 rounded-xl border border-slate-800 bg-slate-900 p-5">
                <h2 className="font-semibold">{selectedNode.label}</h2>

                <p className="mt-1 text-sm text-slate-400">
                  Recorded stage artifacts
                </p>

                <div className="mt-4 space-y-4">
                  {selectedNode.artifact_ids.length === 0 && (
                    <p className="text-sm text-slate-400">
                      No artifacts were supplied for this stage.
                    </p>
                  )}

                  {selectedNode.artifact_ids.map((id) => {
                    const artifact = artifacts[id];

                    return (
                      <div
                        key={id}
                        className="rounded-lg border border-slate-700 bg-slate-950 p-4"
                      >
                        <p className="break-all font-mono text-xs text-cyan-300">
                          {id}
                        </p>

                        <p className="mt-2 text-xs text-slate-400">
                          Availability: {artifact?.availability ?? "unknown"}
                        </p>

                        <pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap break-words text-xs leading-6 text-slate-300">
                          {artifact?.availability === "available"
                            ? formatContent(artifact.content)
                            : "Artifact content is unavailable."}
                        </pre>
                      </div>
                    );
                  })}
                </div>
              </section>

              <section className="min-w-0 rounded-xl border border-amber-500/30 bg-slate-900 p-5">
                <div className="flex items-center justify-between gap-3">
                  <h2 className="font-semibold">Diagnostic verdict</h2>

                  <span className="rounded bg-amber-400/10 px-2 py-1 text-xs text-amber-300">
                    {verdict.category}
                  </span>
                </div>

                <p className="mt-4 text-lg font-medium">
                  {verdict.primary_stage.replaceAll("_", " ")}
                </p>

                <p className="mt-3 text-sm leading-6 text-slate-300">
                  {verdict.summary}
                </p>

                <p className="mt-4 text-sm text-cyan-300">
                  Evidence score: {(verdict.confidence * 100).toFixed(0)}%
                </p>

                <p className="mt-1 text-xs text-slate-400">
                  Uncalibrated score; not a probability of correctness.
                </p>

                <h3 className="mt-6 text-sm font-semibold">Evidence</h3>

                {verdict.evidence_trail.map((evidence, index) => (
                  <div key={index} className="mt-3 text-sm">
                    <p className="leading-6 text-slate-300">
                      {evidence.finding}
                    </p>

                    {evidence.citations.map((citation, citationIndex) => (
                      <blockquote
                        key={`${citation.artifact_id}-${citationIndex}`}
                        className="mt-3 border-l-2 border-cyan-500 pl-3"
                      >
                        <p className="text-slate-300">
                          “{citation.quote}”
                        </p>

                        <p className="mt-1 break-all font-mono text-xs text-slate-500">
                          {citation.artifact_id}
                        </p>
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
              </section>
            </div>
          </section>
        </div>

        <section className="rounded-xl border border-slate-800 bg-slate-900 p-5">
          <h2 className="font-semibold">Example benchmark results</h2>

          <p className="mt-1 text-sm text-slate-400">
            {benchmark.dataset.name} · {benchmark.dataset.split}
          </p>

          <div className="mt-5 grid gap-5 sm:grid-cols-4">
            {benchmarkMetrics.map((metric) => (
              <div key={metric.label}>
                <p className="text-sm text-slate-400">{metric.label}</p>
                <p className="mt-2 text-2xl font-semibold">{metric.value}</p>
              </div>
            ))}
          </div>

          <p className="mt-5 text-xs text-slate-400">
            These are supplied example values, not newly measured results.
          </p>
        </section>
      </div>
    </main>
  );
}