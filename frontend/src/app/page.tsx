"use client";

import Link from "next/link";
import { useState } from "react";
import PipelineGraph from "@/components/PipelineGraph";
import examples from "@/data/ui-api.examples.json";

const trace = examples.TraceDetail;
const verdict = trace.latest_live_diagnosis.verdict;
const dashboard = examples.Dashboard;
const benchmark = examples.BenchmarkReport;

const snapshotFrom = dashboard.window.from.slice(0, 16);
const snapshotTo = dashboard.window.to.slice(0, 16);

type Artifact = {
  artifact_id: string;
  kind: string;
  availability: string;
  content: unknown;
};

const artifacts = examples.artifacts_by_id as Record<string, Artifact>;

function formatContent(content: unknown) {
  return typeof content === "string"
    ? content
    : JSON.stringify(content, null, 2);
}

function utcTimestamp(value: string) {
  return value ? Date.parse(`${value}:00Z`) : NaN;
}

export default function Home() {
  const [selectedNodeId, setSelectedNodeId] = useState(
    trace.graph.nodes[0].node_id
  );
  const [appId, setAppId] = useState("all");
  const [from, setFrom] = useState(snapshotFrom);
  const [to, setTo] = useState(snapshotTo);

  const selectedNode =
    trace.graph.nodes.find((node) => node.node_id === selectedNodeId) ??
    trace.graph.nodes[0];

  const fromTime = utcTimestamp(from);
  const toTime = utcTimestamp(to);

  const validRange =
    Number.isFinite(fromTime) &&
    Number.isFinite(toTime) &&
    fromTime < toTime;

  const snapshotSelected =
    validRange &&
    fromTime === Date.parse(dashboard.window.from) &&
    toTime === Date.parse(dashboard.window.to);

  const selectedApp = dashboard.apps.find((app) => app.app_id === appId);

  const counts = snapshotSelected
    ? appId === "all"
      ? dashboard.totals
      : selectedApp?.counts
    : undefined;

  const traceTime = Date.parse(trace.execution.started_at);

  const traceVisible =
    validRange &&
    (appId === "all" || appId === trace.execution.app_id) &&
    traceTime >= fromTime &&
    traceTime < toTime;

  const metrics = [
    {
      label: "Observed traces",
      value: counts?.observed_traces ?? "Unavailable",
    },
    {
      label: "Quality flags",
      value: counts?.flagged_traces ?? "Unavailable",
    },
    {
      label: "Runtime failures",
      value: counts?.runtime_failed_traces ?? "Unavailable",
    },
    {
      label: "Abstention rate",
      value: counts
        ? `${(counts.abstention_rate * 100).toFixed(1)}%`
        : "Unavailable",
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

  function resetFilters() {
    setAppId("all");
    setFrom(snapshotFrom);
    setTo(snapshotTo);
  }

  const fieldClass =
    "mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-cyan-400";

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

          <Link
            href="/traces"
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-cyan-400 hover:border-cyan-400"
          >
            Trace explorer
          </Link>

          <Link
            href="/spans"
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-cyan-400 hover:border-cyan-400"
          >
            Span inspector
          </Link>
        </header>

        <section className="rounded-xl border border-slate-800 bg-slate-900 p-5">
          <div className="grid gap-4 md:grid-cols-3">
            <label className="text-sm">
              Application
              <select
                value={appId}
                onChange={(event) => setAppId(event.target.value)}
                className={fieldClass}
              >
                <option value="all">All sample applications</option>
                {dashboard.apps.map((app) => (
                  <option key={app.app_id} value={app.app_id}>
                    {app.service_name}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-sm">
              From (UTC)
              <input
                type="datetime-local"
                value={from}
                onChange={(event) => setFrom(event.target.value)}
                className={fieldClass}
              />
            </label>

            <label className="text-sm">
              To (UTC, exclusive)
              <input
                type="datetime-local"
                value={to}
                onChange={(event) => setTo(event.target.value)}
                className={fieldClass}
              />
            </label>
          </div>

          <button
            type="button"
            onClick={resetFilters}
            className="mt-4 text-sm text-cyan-400 hover:underline"
          >
            Reset to supplied reporting window
          </button>

          {!validRange && (
            <p role="alert" className="mt-3 text-sm text-red-400">
              Enter both dates, with the end after the start.
            </p>
          )}

          <p className="mt-4 text-xs leading-6 text-slate-400">
            These filters use sample data only. Registered applications are
            listed separately under View applications. Aggregate metrics
            are available only for the supplied reporting window:
            {" "}
            {dashboard.window.from} to {dashboard.window.to}.
          </p>
        </section>

        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {metrics.map((metric) => (
            <div
              key={metric.label}
              className="rounded-xl border border-slate-800 bg-slate-900 p-5"
            >
              <p className="text-sm text-slate-400">{metric.label}</p>
              <p className="mt-3 break-words text-2xl font-semibold">
                {metric.value}
              </p>
            </div>
          ))}
        </section>

        <p role="status" className="text-xs leading-6 text-slate-400">
          {counts
            ? "Metrics describe the supplied reporting window. Only one detailed execution is included, so the trace preview is not a complete list of those totals."
            : "Aggregate metrics are unavailable for this range. The recorded trace below is filtered independently by its start time."}
          {" "}
          Runtime failures describe execution errors; quality flags identify
          runs marked for diagnostic review. These groups may overlap.
        </p>

        {!traceVisible ? (
          <section className="rounded-xl border border-slate-800 bg-slate-900 p-6">
            <h2 className="font-semibold">No sample trace in this selection</h2>
            <p className="mt-3 text-sm text-slate-400">
              The only supplied detailed execution started at
              {" "}
              {trace.execution.started_at}. This does not establish that
              the backend has no traces in your selected range.
            </p>
            <button
              type="button"
              onClick={resetFilters}
              className="mt-4 text-sm text-cyan-400 hover:underline"
            >
              Restore sample selection
            </button>
          </section>
        ) : (
          <div className="grid gap-5 lg:grid-cols-12">
            <aside className="rounded-xl border border-slate-800 bg-slate-900 p-5 lg:col-span-3">
              <h2 className="font-semibold">Sample trace</h2>

              <div className="mt-4 rounded-lg border border-cyan-500/50 bg-cyan-500/10 p-4">
                <Link
                  href={`/traces/${trace.execution.execution_id}`}
                  className="font-mono text-sm text-cyan-300 hover:underline"
                >
                  {trace.execution.execution_id}
                </Link>

                <p className="mt-2 text-sm">
                  {trace.execution.environment}
                </p>

                <p className="mt-1 text-xs text-slate-400">
                  {trace.execution.duration_ms} ms · {trace.execution.status}
                </p>

                <span className="mt-3 inline-block rounded bg-amber-400/10 px-2 py-1 text-xs text-amber-300">
                  Flagged for diagnosis
                </span>

                <Link
                  href={`/traces/${trace.execution.execution_id}`}
                  className="mt-4 block text-sm text-cyan-300 hover:underline"
                >
                  Open trace details →
                </Link>
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
        )}

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
            This separate benchmark report is unaffected by the dashboard
            app and time filters.
          </p>
        </section>
      </div>
    </main>
  );
}