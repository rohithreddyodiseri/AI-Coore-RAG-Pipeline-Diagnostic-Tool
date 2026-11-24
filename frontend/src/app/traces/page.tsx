"use client";

import Link from "next/link";
import { useState } from "react";
import examples from "@/data/ui-api.examples.json";

const traces = [examples.TraceDetail];

export default function TraceExplorerPage() {
  const [search, setSearch] = useState("");
  const [environment, setEnvironment] = useState("all");
  const [status, setStatus] = useState("all");
  const [flagged, setFlagged] = useState("all");

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
            <h1 className="mt-4 text-3xl font-semibold">
              Trace explorer
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Search executions and open their stages and evidence.
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
                <th scope="col" className="p-4">
                  Execution
                </th>
                <th scope="col" className="p-4">
                  App
                </th>
                <th scope="col" className="p-4">
                  Environment
                </th>
                <th scope="col" className="p-4">
                  Duration
                </th>
                <th scope="col" className="p-4">
                  Runtime
                </th>
                <th scope="col" className="p-4">
                  Diagnosis
                </th>
              </tr>
            </thead>

            <tbody>
              {filteredTraces.map(({ execution }) => (
                <tr
                  key={execution.execution_id}
                  className="border-b border-slate-800 last:border-0 hover:bg-slate-800/50"
                >
                  <td className="p-4">
                    <Link
                      href={`/traces/${encodeURIComponent(
                        execution.execution_id
                      )}`}
                      className="font-mono text-cyan-400 hover:underline"
                    >
                      {execution.execution_id}
                    </Link>
                  </td>

                  <td className="p-4">{execution.app_id}</td>

                  <td className="p-4">
                    {execution.environment}
                  </td>

                  <td className="p-4">
                    {execution.duration_ms} ms
                  </td>

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
                  <td
                    colSpan={6}
                    className="p-8 text-center text-slate-400"
                  >
                    No executions match these filters. Try clearing them.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </section>

        {filteredTraces.length > 0 && (
          <p className="rounded-xl border border-slate-800 p-6 text-sm text-slate-400">
            Click an execution ID to open its pipeline graph, verdict,
            and supporting evidence.
          </p>
        )}
      </div>
    </main>
  );
}