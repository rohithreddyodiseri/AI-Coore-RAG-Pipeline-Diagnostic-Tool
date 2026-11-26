"use client";

import Link from "next/link";
import { useState } from "react";
import ApiStatus from "@/components/ApiStatus";
import { useApi } from "@/hooks/use-api";
import { HistoryChart, OutcomeChart, PrecisionChart } from "./Charts";
import {
  type AccuracyHistory, type AccuracyRun, type Outcome, type Summary,
  dateLabel, isActive, outcomes, percent, pollHistory, readable,
} from "./types";

const panel = "min-w-0 rounded-2xl border border-slate-800 bg-slate-900/70 p-5";
const select = "mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm";
const measures = [
  { key: "suite_accuracy", label: "Overall accuracy", note: "Exact root, fault set and expected statuses" },
  { key: "primary_accuracy", label: "Primary accuracy", note: "Exact primary mechanism and stage on fault cases" },
  { key: "stage_accuracy", label: "Stage accuracy", note: "Correct primary stage on fault cases" },
  { key: "mechanism_precision", label: "Mechanism precision", note: "Correct fault labels among all predictions" },
] as const;

function comparisonDelta(summary: Summary, baseline: Summary | undefined, metric: keyof Summary) {
  const current = summary[metric];
  const previous = baseline?.[metric];
  if (typeof current !== "number" || typeof previous !== "number") return null;
  const points = (current - previous) * 100;
  return `${points >= 0 ? "+" : ""}${points.toFixed(1)} pp vs comparison`;
}

function RunStatus({ run }: { run: AccuracyRun }) {
  if (run.error_message) return <ApiStatus error={run.error_message} />;
  if (isActive(run)) {
    return (
      <section className={panel} aria-live="polite">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="font-semibold">{run.state === "queued" ? "Suite queued" : "Running the fault suite"}</p>
          <p className="text-sm tabular-nums text-slate-400">{run.finished_cases} / {run.total_cases} cases</p>
        </div>
        <progress aria-label="Completed test cases" className="mt-4 h-2 w-full accent-cyan-400"
          value={run.finished_cases} max={run.total_cases} />
      </section>
    );
  }
  if (run.summary?.errors) {
    return <p role="status" className="rounded-xl border border-amber-800 bg-amber-950/30 p-4 text-sm text-amber-200">
      {run.summary.errors} of {run.summary.attempted} cases could not complete diagnosis and count as incorrect.
    </p>;
  }
  return null;
}

export default function AccuracyPage() {
  const history = useApi<AccuracyHistory>("accuracy-runs", { pollIntervalMs: 2000, pollWhile: pollHistory });
  const [selectedId, setSelectedId] = useState("");
  const [comparisonId, setComparisonId] = useState("");
  const [outcome, setOutcome] = useState<Outcome | "">("");
  const [mechanism, setMechanism] = useState("");
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState("");

  const runs = history.data?.items ?? [];
  const runNumbers = new Map(runs.map((run, index) => [run.run_id, runs.length - index]));
  const active = runs.find(isActive);
  const current = runs.find((run) => run.run_id === selectedId) ?? runs[0];
  const completed = runs.filter((run) => run.summary);
  const comparison = completed.find((run) => run.run_id === comparisonId && run.run_id !== current?.run_id);
  const detail = useApi<AccuracyRun>(current?.summary ? `accuracy-runs/${current.run_id}` : null);
  const baseline = useApi<AccuracyRun>(comparison ? `accuracy-runs/${comparison.run_id}` : null);
  const report = detail.data?.report;
  const visibleCases = report?.cases.filter((row) =>
    (!outcome || row.outcome === outcome) &&
    (!mechanism || row.expected_faults.includes(mechanism) || row.predicted_faults.includes(mechanism)),
  );

  function selectRun(id: string) {
    setSelectedId(id);
    if (comparisonId === id) setComparisonId("");
    setOutcome("");
    setMechanism("");
  }

  async function start() {
    setStarting(true);
    setStartError("");
    try {
      const response = await fetch("/api/accuracy-runs", { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.message ?? "Could not start the suite.");
      selectRun(data.run_id);
      history.refresh();
    } catch (error) {
      setStartError(error instanceof Error ? error.message : "Could not start the suite.");
    } finally {
      setStarting(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-7 text-slate-100 sm:px-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <Link href="/" className="text-sm font-medium text-cyan-300 hover:text-cyan-100">← Dashboard</Link>

        <header className="flex flex-wrap items-end justify-between gap-5 border-b border-slate-800 pb-6">
          <div>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">Diagnostic Accuracy</h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
              Compare the judge’s diagnoses with known faults in the controlled test suite.
              {history.data && ` Each run evaluates ${history.data.suite_cases} cases and uses Gemini API quota.`}
            </p>
          </div>
          {history.data?.can_run && (
            <button type="button" onClick={() => void start()} disabled={starting || Boolean(active)}
              className="rounded-lg bg-cyan-300 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-50">
              {starting ? "Starting…" : active ? "Suite in progress" : "Run fault suite"}
            </button>
          )}
        </header>

        <ApiStatus loading={history.loading} error={startError || history.error} />
        {history.data && !history.data.can_run && <p className="text-sm text-slate-400">An administrator can launch a new run.</p>}
        {history.data && !runs.length && (
          <section className={panel}>
            <h2 className="text-lg font-semibold">No test runs yet</h2>
            <p className="mt-2 text-sm text-slate-400">Completed runs will appear here with accuracy, precision, and individual case results.</p>
          </section>
        )}

        {current && (
          <section className={`${panel} grid gap-4 sm:grid-cols-2`}>
            <label className="text-sm text-slate-400">
              Test run
              <select className={select} value={current.run_id} onChange={(event) => selectRun(event.target.value)}>
                {runs.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    #{runNumbers.get(run.run_id)} · {dateLabel(run.created_at)} · {readable(run.state)}{run.summary?.model ? ` · ${run.summary.model}` : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-sm text-slate-400">
              Compare with
              <select className={select} value={comparison?.run_id ?? ""} disabled={!current.summary}
                onChange={(event) => setComparisonId(event.target.value)}>
                <option value="">Choose a completed run</option>
                {completed.filter((run) => run.run_id !== current.run_id).map((run) => (
                  <option key={run.run_id} value={run.run_id}>#{runNumbers.get(run.run_id)} · {dateLabel(run.created_at)} · {run.summary?.model}</option>
                ))}
              </select>
            </label>
          </section>
        )}

        {active && active.run_id !== current?.run_id && (
          <button type="button" onClick={() => selectRun(active.run_id)} className="text-sm text-cyan-300 underline">
            View active run ({active.finished_cases} / {active.total_cases} cases)
          </button>
        )}
        {current && <RunStatus run={current} />}

        {current?.summary && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {measures.map(({ key, label, note }) => (
              <section key={key} className={panel}>
                <h2 className="text-sm font-medium text-slate-300">{label}</h2>
                <p className="mt-3 text-3xl font-semibold tabular-nums text-cyan-200">{percent(current.summary?.[key])}</p>
                <p className="mt-2 text-xs leading-5 text-slate-400">{note}</p>
                <p className="mt-1 text-xs text-violet-300">{current.summary && comparisonDelta(current.summary, comparison?.summary ?? undefined, key)}</p>
              </section>
            ))}
          </div>
        )}

        {completed.length > 1 && (
          <section className={panel}>
            <h2 className="text-lg font-semibold">Accuracy across runs</h2>
            <p className="mb-4 mt-1 text-sm text-slate-400">Hover over a run to see its scores and judge model.</p>
            <HistoryChart runs={runs} />
          </section>
        )}

        <ApiStatus loading={detail.loading || baseline.loading} error={detail.error ?? baseline.error} />
        {report && (
          <>
            <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,2.1fr)_minmax(0,1fr)]">
              <section className={panel}>
                <h2 className="text-lg font-semibold">Precision by failure mechanism</h2>
                <p className="mt-1 text-sm text-slate-400">Select a bar to filter cases. A dash in the metrics table means the judge made no predictions for that mechanism.</p>
                <div className="mt-4 overflow-x-auto"><div className="min-w-[420px]">
                  <PrecisionChart metrics={report.mechanism_metrics} comparison={baseline.data?.report?.mechanism_metrics}
                    selected={mechanism} onSelect={(value) => setMechanism(mechanism === value ? "" : value)} />
                </div></div>
                <details className="mt-3 text-sm">
                  <summary className="cursor-pointer text-cyan-300">View precision, recall, and counts</summary>
                  <div className="mt-3 overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="text-slate-400">
                        <tr>{["Mechanism", "Expected", "Predicted", "Correct labels", "Precision", "Recall",
                          ...(comparison ? ["Comparison precision"] : [])].map((label) =>
                          <th key={label} scope="col" className="p-2 font-medium">{label}</th>)}</tr>
                      </thead>
                      <tbody>
                        {Object.entries(report.mechanism_metrics).map(([name, row]) => (
                          <tr key={name} className="border-t border-slate-800">
                            <th scope="row" className="p-2 font-medium">{readable(name)}</th>
                            {[row.support, row.predicted, row.true_positives, percent(row.precision), percent(row.recall),
                              ...(comparison ? [percent(baseline.data?.report?.mechanism_metrics[name]?.precision)] : [])].map((value, i) =>
                              <td key={i} className="p-2 tabular-nums">{value}</td>)}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              </section>
              <section className={panel}>
                <h2 className="text-lg font-semibold">Test outcomes</h2>
                <OutcomeChart cases={report.cases} selected={outcome}
                  onSelect={(value) => setOutcome(outcome === value ? "" : value)} />
                <p className="mt-4 text-xs leading-5 text-slate-400">An expected abstention on a control counts as correct when the case passes its checks.</p>
              </section>
            </div>

            <section className={panel}>
              <div className="flex flex-wrap items-end justify-between gap-4">
                <div>
                  <h2 className="text-lg font-semibold">Individual test results</h2>
                  <p className="mt-1 text-sm text-slate-400" aria-live="polite">{visibleCases?.length} of {report.cases.length} cases shown</p>
                </div>
                <div className="flex flex-wrap items-end gap-3">
                  <label className="text-sm text-slate-400">
                    Mechanism
                    <select className={select} value={mechanism} onChange={(event) => setMechanism(event.target.value)}>
                      <option value="">All mechanisms</option>
                      {Object.keys(report.mechanism_metrics).map((name) => <option key={name} value={name}>{readable(name)}</option>)}
                    </select>
                  </label>
                  <label className="text-sm text-slate-400">
                    Outcome
                    <select className={select} value={outcome} onChange={(event) => setOutcome(event.target.value as Outcome | "")}>
                      <option value="">All outcomes</option>
                      {Object.entries(outcomes).map(([value, item]) => <option key={value} value={value}>{item.label}</option>)}
                    </select>
                  </label>
                </div>
              </div>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[800px] text-left text-sm">
                  <thead className="bg-slate-950/60 text-slate-400">
                    <tr>{["Case", "Expected failure", "Detected failure", "Predicted stage", "Outcome"].map((label) =>
                      <th key={label} scope="col" className="px-3 py-3 font-medium">{label}</th>)}</tr>
                  </thead>
                  <tbody>
                    {visibleCases?.map((row) => (
                      <tr key={row.run_id} className="border-b border-slate-800 align-top">
                        <td className="px-3 py-3 font-medium">{row.kind === "fault" ? readable(row.description) : row.description}</td>
                        <td className="px-3 py-3">{row.expected_faults.map(readable).join(", ") || "None"}
                          {row.expected_root && <span className="mt-1 block text-xs text-slate-400">Primary · {readable(row.expected_root.mechanism)} / {readable(row.expected_root.technical_stage)}</span>}
                        </td>
                        <td className="px-3 py-3">{row.predicted_faults.map(readable).join(", ") || "None"}
                          {row.predicted_root && <span className="mt-1 block text-xs text-slate-400">Primary · {readable(row.predicted_root.mechanism)}</span>}
                        </td>
                        <td className="px-3 py-3">{row.predicted_root ? readable(row.predicted_root.technical_stage) : "—"}</td>
                        <td className="px-3 py-3">
                          <span className="inline-block rounded-full border px-2 py-0.5 text-xs" style={{ color: outcomes[row.outcome].color }}>
                            {outcomes[row.outcome].label}
                          </span>
                          {row.correct && row.status === "abstained" && <span className="mt-1 block text-xs text-slate-400">Expected abstention</span>}
                        </td>
                      </tr>
                    ))}
                    {!visibleCases?.length && <tr><td colSpan={5} className="p-6 text-center text-slate-400">No cases match the selected filters.</td></tr>}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
