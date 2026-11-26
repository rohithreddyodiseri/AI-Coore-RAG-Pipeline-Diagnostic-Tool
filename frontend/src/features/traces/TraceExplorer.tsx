"use client";

import { ArrowUpRight, RefreshCw, Search } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import {
  CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis,
} from "recharts";
import ApiStatus from "@/components/ApiStatus";
import {
  Badge, ConsolePage, Empty, Section, button, duration, field, link, panel, readable,
} from "@/features/console/ui";
import { useApi } from "@/hooks/use-api";
import type { App, Execution, Page } from "@/lib/api";

const blankFilters = {
  app_id: "", environment: "", status: "", flagged: "", diagnosis_state: "", from: "", to: "",
};
type Filters = typeof blankFilters;
const filterKeys = Object.keys(blankFilters) as (keyof Filters)[];
const diagnosisStates = ["not_requested", "queued", "running", "completed", "failed", "insufficient_evidence"];
const shortTime = new Intl.DateTimeFormat(undefined, {
  month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
});

function TracePlot({ traces, onSelect, active }: {
  traces: Execution[]; onSelect: (id: string) => void; active?: string;
}) {
  const points = traces.flatMap((trace) => {
    const time = Date.parse(trace.started_at);
    return trace.duration_ms == null || !Number.isFinite(time) ? [] : [{
      id: trace.execution_id, time, elapsed: trace.duration_ms, status: trace.status, flagged: trace.flagged,
    }];
  });
  if (!points.length) return <Empty title="No timed executions">Executions with a recorded start and duration appear here.</Empty>;

  return (
    <div className="h-64 min-w-0" aria-label="Scatter plot of execution start times and durations">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 16, bottom: 10, left: 4 }}>
          <CartesianGrid stroke="#273449" vertical={false} />
          <XAxis type="number" dataKey="time" name="Started" domain={["dataMin - 1000", "dataMax + 1000"]}
            tickFormatter={(value) => shortTime.format(value)} stroke="#94a3b8" minTickGap={65} fontSize={11} />
          <YAxis type="number" dataKey="elapsed" name="Duration" tickFormatter={duration} stroke="#94a3b8" fontSize={11} />
          <Tooltip
            contentStyle={{ background: "#0f172a", borderColor: "#475569", borderRadius: 8 }}
            itemStyle={{ color: "#e2e8f0" }}
            formatter={(value, name) => name === "Started" ? shortTime.format(Number(value)) : duration(Number(value))}
            cursor={{ stroke: "#64748b", strokeDasharray: "3 3" }}
          />
          <Scatter name="Executions" data={points} fill="#22d3ee" isAnimationActive={false}
            cursor="pointer" onClick={(_, index) => onSelect(points[index].id)}>
            {points.map((point) => (
              <Cell key={point.id} fill={point.status === "failed" ? "#fb7185" : point.flagged ? "#fbbf24" : "#22d3ee"}
                stroke={point.id === active ? "#f8fafc" : "#0f172a"} strokeWidth={point.id === active ? 3 : 1} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function TraceExplorer({ initialFilters }: { initialFilters: Record<string, string> }) {
  const [search, setSearch] = useState("");
  const [filters, setFilters] = useState<Filters>(() => Object.fromEntries(
    filterKeys.map((key) => [key, initialFilters[key] ?? blankFilters[key]]),
  ) as Filters);
  const [pages, setPages] = useState<string[]>([""]);
  const [sort, setSort] = useState("recent");
  const [selectedId, setSelectedId] = useState("");
  const query = new URLSearchParams({ limit: "50" });
  for (const key of filterKeys) if (filters[key].trim()) query.set(key, filters[key].trim());
  const cursor = pages.at(-1);
  if (cursor) query.set("cursor", cursor);
  const result = useApi<Page<Execution>>(`traces?${query}`);
  const apps = useApi<Page<App>>("apps?limit=100");
  const names = new Map(apps.data?.items.map((app) => [app.app_id, app.service_name]));
  const knownEnvironments = [...new Set([
    ...(apps.data?.items.flatMap((app) => app.environments) ?? []),
    ...(result.data?.items.flatMap((trace) => trace.environment ? [trace.environment] : []) ?? []),
  ])].sort();
  const needle = search.trim().toLowerCase();
  const traces = (result.data?.items ?? []).filter((item) =>
    [item.execution_id, item.trace_id, item.app_id, names.get(item.app_id)].join(" ").toLowerCase().includes(needle),
  ).sort((a, b) => sort === "slowest"
    ? (b.duration_ms ?? -1) - (a.duration_ms ?? -1)
    : Date.parse(b.started_at) - Date.parse(a.started_at));
  const selected = traces.find((item) => item.execution_id === selectedId);
  const period = [filters.from, filters.to].map(Date.parse).filter(Number.isFinite).map((value) => shortTime.format(value));
  const statusCounts = [
    { label: "Completed", value: "completed", count: traces.filter((item) => item.status === "completed").length, key: "status" },
    { label: "Runtime failures", value: "failed", count: traces.filter((item) => item.status === "failed").length, key: "status" },
    { label: "Flagged", value: "true", count: traces.filter((item) => item.flagged).length, key: "flagged" },
  ] as const;

  function filter(key: keyof Filters, value: string) {
    setFilters((current) => ({ ...current, [key]: value }));
    setPages([""]);
  }

  function reset() {
    setFilters(blankFilters);
    setSearch("");
    setPages([""]);
  }

  return (
    <ConsolePage active="/traces" title="Trace explorer"
      description="Find received executions and compare their timing, runtime status, and diagnostic verdicts."
      actions={<button type="button" className={button} onClick={result.refresh}><RefreshCw size={15} aria-hidden="true" /> Refresh</button>}>
      <Section title="Filters" actions={<button type="button" className={link} onClick={reset}>Clear all filters</button>}>
        <div className="grid gap-4 text-sm text-slate-300 sm:grid-cols-2 lg:grid-cols-3">
          <label>Search this page
            <span className="relative block">
              <Search size={15} aria-hidden="true" className="absolute left-3 top-5 text-slate-500" />
              <input className={`${field} pl-9`} type="search" placeholder="Execution, trace, or service" value={search} onChange={(event) => setSearch(event.target.value)} />
            </span>
          </label>
          <label>Application
            <select className={field} value={filters.app_id} onChange={(event) => filter("app_id", event.target.value)}>
              <option value="">All applications</option>
              {filters.app_id && !names.has(filters.app_id) && <option value={filters.app_id}>{filters.app_id}</option>}
              {apps.data?.items.map((app) => <option key={app.app_id} value={app.app_id}>{app.service_name}</option>)}
            </select>
          </label>
          <label>Environment
            <input className={field} list="trace-environments" value={filters.environment}
              onChange={(event) => filter("environment", event.target.value)} placeholder="All environments" />
            <datalist id="trace-environments">{knownEnvironments.map((name) => <option key={name} value={name} />)}</datalist>
          </label>
          <label>Runtime status
            <select className={field} value={filters.status} onChange={(event) => filter("status", event.target.value)}>
              <option value="">All statuses</option><option value="completed">Completed</option><option value="failed">Failed</option>
            </select>
          </label>
          <label>Diagnostic flag
            <select className={field} value={filters.flagged} onChange={(event) => filter("flagged", event.target.value)}>
              <option value="">All executions</option><option value="true">Flagged</option><option value="false">Unflagged</option>
            </select>
          </label>
          <label>Diagnosis
            <select className={field} value={filters.diagnosis_state} onChange={(event) => filter("diagnosis_state", event.target.value)}>
              <option value="">All states</option>
              {diagnosisStates.map((state) => <option key={state} value={state}>{readable(state)}</option>)}
            </select>
          </label>
        </div>
        {period.length > 0 && (
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <Badge tone="cyan">Time filter · {period.join(" – ")}</Badge>
            <button type="button" className={link} onClick={() => {
              setFilters((current) => ({ ...current, from: "", to: "" }));
              setPages([""]);
            }}>Remove time filter</button>
          </div>
        )}
      </Section>

      <ApiStatus loading={result.loading} error={result.error} />
      {result.data && (
        <>
          <Section title="Execution timing" description="Each dot is a timed execution on this page. Select a dot to preview the trace.">
            <div className="mb-2 flex flex-wrap gap-2">
              {statusCounts.map(({ label, value, count, key }) => (
                <button type="button" key={label} className={`${button} aria-pressed:border-cyan-400/50`}
                  aria-pressed={filters[key] === value} onClick={() => filter(key, filters[key] === value ? "" : value)}>
                  <span className={`size-2 rounded-full ${value === "failed" ? "bg-rose-400" : value === "true" ? "bg-amber-400" : "bg-cyan-400"}`} />
                  {label} <span className="font-mono text-xs">{count}</span>
                </button>
              ))}
            </div>
            <TracePlot traces={traces} active={selected?.execution_id} onSelect={setSelectedId} />
            {selected && <div className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-cyan-400/30 bg-cyan-400/5 p-3 text-sm" aria-live="polite">
              <span className="min-w-0 break-all">{names.get(selected.app_id) ?? selected.app_id} · {shortTime.format(new Date(selected.started_at))} · {duration(selected.duration_ms)}</span>
              <div className="flex flex-wrap gap-3">
                <Link className={link} href={`/traces/${selected.execution_id}`}>Open trace</Link>
                <Link className={link} href={`/spans?execution_id=${selected.execution_id}`}>Inspect spans</Link>
              </div>
            </div>}
          </Section>
          <section className={`${panel} overflow-hidden`}>
            <div className="flex flex-wrap items-center justify-between gap-3 p-5">
              <div><h2 className="font-semibold">Received executions</h2><p className="mt-1 text-sm text-slate-400">{traces.length} shown · page {pages.length}</p></div>
              <select aria-label="Sort current page" className={`${field} mt-0 w-auto`} value={sort} onChange={(event) => setSort(event.target.value)}>
                <option value="recent">Latest on this page</option>
                <option value="slowest">Slowest on this page</option>
              </select>
            </div>
            {traces.length ? (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[750px] text-left text-sm">
                  <thead className="bg-slate-950/70 text-slate-400">
                    <tr>{["Trace", "Environment", "Started", "Duration", "Runtime", "Diagnosis", "Spans"].map((heading) => (
                      <th key={heading} scope="col" className="px-4 py-3 font-medium">{heading}</th>
                    ))}</tr>
                  </thead>
                  <tbody>
                    {traces.map((trace) => (
                      <tr key={trace.execution_id} className="border-t border-slate-800 hover:bg-slate-800/35">
                        <td className="max-w-64 px-4 py-3">
                          <Link className={link} href={`/traces/${trace.execution_id}`} title={trace.execution_id}>{names.get(trace.app_id) ?? "Open trace"}</Link>
                          <p className="mt-1 break-all font-mono text-xs text-slate-400">{trace.trace_id ?? trace.execution_id}</p>
                        </td>
                        <td className="px-4 py-3">{trace.environment ?? "Unknown"}</td>
                        <td className="whitespace-nowrap px-4 py-3">{shortTime.format(new Date(trace.started_at))}</td>
                        <td className="whitespace-nowrap px-4 py-3 font-mono">{duration(trace.duration_ms)}</td>
                        <td className="px-4 py-3"><Badge tone={trace.status === "failed" ? "rose" : "green"}>{trace.status}</Badge></td>
                        <td className="px-4 py-3">
                          <Badge tone={trace.diagnosis_state === "completed" ? "violet" : "slate"}>{readable(trace.diagnosis_state)}</Badge>
                          {trace.flagged && <p className="mt-1 text-xs text-amber-300">Flagged</p>}
                        </td>
                        <td className="px-4 py-3"><Link className={`inline-flex items-center gap-1 ${link}`}
                          href={`/spans?execution_id=${trace.execution_id}`} aria-label={`Inspect spans for ${trace.trace_id ?? trace.execution_id}`}>
                          Inspect <ArrowUpRight size={13} aria-hidden="true" />
                        </Link></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="p-5 pt-0"><Empty title="No matching executions">Adjust the filters or send telemetry from a registered application.</Empty></div>}
            {(pages.length > 1 || result.data.next_cursor) && <div className="flex gap-2 border-t border-slate-800 p-4">
              <button type="button" className={button} disabled={pages.length === 1}
                onClick={() => setPages((current) => current.slice(0, -1))}>Previous page</button>
              <button type="button" className={button} disabled={!result.data.next_cursor} onClick={() => {
                const next = result.data?.next_cursor;
                if (next) setPages((current) => [...current, next]);
              }}>Next page</button>
            </div>}
          </section>
        </>
      )}
    </ConsolePage>
  );
}
