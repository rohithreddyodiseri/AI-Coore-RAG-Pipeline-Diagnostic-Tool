"use client";

import { ArrowUpRight, RefreshCw, Search, ZoomIn, ZoomOut } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";
import ApiStatus from "@/components/ApiStatus";
import {
  Badge, ConsolePage, Empty, Metric, Section, button, duration, field, link, panel, readable,
} from "@/features/console/ui";
import { useApi } from "@/hooks/use-api";
import type { App, Execution, Page } from "@/lib/api";

type Attribute = { key: string; value: { stringValue?: string; intValue?: string | number } };
type Span = {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  attributes?: Attribute[];
  status?: { code?: number };
};
type ExportRequest = { resourceSpans?: { scopeSpans?: { spans?: Span[] }[] }[] };
type TimedSpan = { span: Span; depth: number; offset: number; elapsed: number };

function attribute(span: Span, key: string) {
  const value = span.attributes?.find((item) => item.key === key)?.value;
  return value?.stringValue ?? String(value?.intValue ?? "");
}

function recordedStatus(span: Span) {
  const stage = attribute(span, "rag.stage.status");
  const execution = attribute(span, "rag.execution.status");
  if (span.status?.code === 2 || stage === "failed" || execution === "failed") return "failed";
  return stage || execution || (span.status?.code === 1 ? "ok" : "unknown");
}

function milliseconds(start: string, end: string) {
  return Number(BigInt(end) - BigInt(start)) / 1_000_000;
}

function timestamp(value: string) {
  return new Date(Number(BigInt(value) / BigInt(1_000_000))).toISOString();
}

function pretty(value: string) {
  try { return JSON.stringify(JSON.parse(value), null, 2); }
  catch { return value; }
}

function timedSpans(request: ExportRequest) {
  const unique = new Map<string, Span>();
  for (const resource of request.resourceSpans ?? []) {
    for (const scope of resource.scopeSpans ?? []) {
      for (const span of scope.spans ?? []) {
        if (!span.traceId || !span.spanId || !span.name) throw new Error("A span is missing its ID or name.");
        if (milliseconds(span.startTimeUnixNano, span.endTimeUnixNano) < 0) throw new Error("A span ends before it starts.");
        timestamp(span.startTimeUnixNano);
        timestamp(span.endTimeUnixNano);
        unique.set(`${span.traceId}:${span.spanId}`, span);
      }
    }
  }
  const spans = [...unique.values()].sort((a, b) => milliseconds(b.startTimeUnixNano, a.startTimeUnixNano));
  if (!spans.length) return [];
  if (new Set(spans.map((span) => span.traceId)).size !== 1) throw new Error("The response contains multiple traces.");

  const byId = new Set(spans.map((span) => span.spanId));
  const children = new Map<string, Span[]>();
  for (const span of spans) {
    if (span.parentSpanId) {
      const group = children.get(span.parentSpanId) ?? [];
      group.push(span);
      children.set(span.parentSpanId, group);
    }
  }
  const rows: TimedSpan[] = [];
  const visited = new Set<string>();
  const start = spans[0].startTimeUnixNano;

  function visit(root: Span) {
    const stack = [{ span: root, depth: 0 }];
    while (stack.length) {
      const { span, depth } = stack.pop()!;
      if (visited.has(span.spanId)) continue;
      visited.add(span.spanId);
      rows.push({
        span, depth, offset: milliseconds(start, span.startTimeUnixNano),
        elapsed: milliseconds(span.startTimeUnixNano, span.endTimeUnixNano),
      });
      for (const child of [...(children.get(span.spanId) ?? [])].reverse()) {
        stack.push({ span: child, depth: depth + 1 });
      }
    }
  }

  spans.filter((span) => !span.parentSpanId || !byId.has(span.parentSpanId)).forEach(visit);
  spans.forEach(visit);
  return rows;
}

function SpanDetail({ row, onSelectParent, parentAvailable }: {
  row: TimedSpan; onSelectParent: () => void; parentAvailable: boolean;
}) {
  const [view, setView] = useState<"input" | "output">("input");
  const { span, elapsed, offset } = row;
  const status = recordedStatus(span);
  const errorType = attribute(span, "error.type");
  const payload = attribute(span, `rag.stage.${view}_json`);

  return (
    <Section title={span.name} description="Selected span · recorded telemetry"
      actions={<Badge tone={status === "failed" ? "rose" : "cyan"}>{readable(status)}</Badge>}>
      <dl className="grid gap-3 text-sm sm:grid-cols-2">
        <div><dt className="text-xs text-slate-400">Span ID</dt><dd className="mt-1 break-all font-mono text-xs">{span.spanId}</dd></div>
        <div><dt className="text-xs text-slate-400">Parent span</dt><dd className="mt-1 break-all font-mono text-xs">
          {parentAvailable ? <button type="button" className={`${link} font-mono text-xs`} onClick={onSelectParent}>{span.parentSpanId}</button> : span.parentSpanId ?? "Root"}
        </dd></div>
        <div><dt className="text-xs text-slate-400">Duration</dt><dd className="mt-1">{duration(elapsed)}</dd></div>
        <div><dt className="text-xs text-slate-400">Offset from trace start</dt><dd className="mt-1">{duration(offset)}</dd></div>
        <div className="sm:col-span-2"><dt className="text-xs text-slate-400">Started (UTC)</dt><dd className="mt-1 break-all font-mono text-xs">{timestamp(span.startTimeUnixNano)}</dd></div>
        <div className="sm:col-span-2"><dt className="text-xs text-slate-400">Ended (UTC)</dt><dd className="mt-1 break-all font-mono text-xs">{timestamp(span.endTimeUnixNano)}</dd></div>
      </dl>
      {(status === "failed" || errorType) && <div className="mt-4 rounded-lg border border-rose-400/25 bg-rose-400/10 p-3 text-sm text-rose-200">
        <p className="font-medium">Recorded runtime error</p>
        <p className="mt-1 break-all font-mono text-xs">{errorType || "Error type unavailable"}</p>
      </div>}
      <div className="mt-5 border-t border-slate-800 pt-4">
        <div className="flex gap-2" aria-label="Recorded span payload">
          {(["input", "output"] as const).map((value) => (
            <button key={value} type="button" aria-pressed={view === value}
              className={`${button} capitalize aria-pressed:border-cyan-400/50 aria-pressed:text-cyan-200`}
              onClick={() => setView(value)}>{value}</button>
          ))}
        </div>
        {payload
          ? <pre tabIndex={0} aria-label={`Recorded ${view}`} className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-slate-950 p-3 text-xs leading-5 text-slate-200">{pretty(payload)}</pre>
          : <p className="mt-3 text-sm text-slate-400">No {view} was captured for this span.</p>}
      </div>
    </Section>
  );
}

function Waterfall({ rows }: { rows: TimedSpan[] }) {
  const [selectedId, setSelectedId] = useState("");
  const [search, setSearch] = useState("");
  const [onlyFailures, setOnlyFailures] = useState(false);
  const [sort, setSort] = useState("tree");
  const [zoom, setZoom] = useState(1);
  const root = rows[0];
  const query = attribute(root.span, "rag.query.text");
  const selected = rows.find(({ span }) => span.spanId === selectedId) ?? root;
  const total = rows.reduce((maximum, row) => Math.max(maximum, row.offset + row.elapsed), 0);
  const failures = rows.filter(({ span }) => recordedStatus(span) === "failed").length;
  const needle = search.trim().toLowerCase();
  const visible = rows.filter(({ span }) => (!onlyFailures || recordedStatus(span) === "failed") &&
    `${span.name} ${span.spanId}`.toLowerCase().includes(needle));
  if (sort === "slowest") visible.sort((a, b) => b.elapsed - a.elapsed);

  const fullScale = Math.max(total, 0.001);
  const scale = fullScale / zoom;
  const center = selected.offset + selected.elapsed / 2;
  const start = Math.max(0, Math.min(center - scale / 2, fullScale - scale));

  return (
    <>
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <div className="col-span-2 sm:col-span-1">
          <Metric label="Trace duration" value={duration(total)} detail={`Root status · ${readable(recordedStatus(root.span))}`} />
        </div>
        <Metric label="Captured spans" value={rows.length} color="text-violet-200" />
        <Metric label="Failed spans" value={failures} color="text-rose-200" />
      </div>
      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <Section title="Span waterfall" description="Select a span to inspect its inputs, outputs, and timing. Zoom centers on the selected span.">
          <div className="mb-4 flex flex-wrap items-center gap-2">
            <label className="relative">
              <Search size={15} aria-hidden="true" className="absolute left-3 top-3.5 text-slate-500" />
              <input className={`${field} mt-0 pl-9`} value={search} onChange={(event) => setSearch(event.target.value)}
                placeholder="Find a span" aria-label="Search spans" type="search" />
            </label>
            <select className={`${field} mt-0 w-auto`} value={sort} onChange={(event) => setSort(event.target.value)} aria-label="Sort spans">
              <option value="tree">Hierarchy</option><option value="slowest">Slowest first</option>
            </select>
            <button type="button" className={`${button} aria-pressed:border-rose-400/60`} aria-pressed={onlyFailures}
              onClick={() => setOnlyFailures(!onlyFailures)}>Failures only</button>
          </div>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
            <span>{visible.length} of {rows.length} spans</span>
            <div className="flex items-center gap-2">
              <button type="button" className={button} disabled={zoom === 1} aria-label="Zoom out"
                onClick={() => setZoom(zoom / 2)}><ZoomOut size={15} aria-hidden="true" /></button>
              <button type="button" className={button} onClick={() => setZoom(1)} aria-label="Reset zoom">{zoom}×</button>
              <button type="button" className={button} disabled={zoom === 16 || !total} aria-label="Zoom in"
                onClick={() => setZoom(zoom * 2)}><ZoomIn size={15} aria-hidden="true" /></button>
            </div>
          </div>
          {visible.length ? (
            <div className="max-h-[35rem] overflow-auto rounded-xl border border-slate-800">
              <table className="w-full min-w-[575px] text-left text-sm">
                <thead className="sticky top-0 z-10 bg-slate-950 text-xs text-slate-400">
                  <tr><th scope="col" className="w-[38%] p-3 font-medium">Span / status</th>
                    <th scope="col" className="p-3 font-medium">Duration</th>
                    <th scope="col" className="w-[45%] p-3 font-medium">
                      <span>Time offset</span>
                      <span className="mt-1 flex justify-between font-mono text-[10px] font-normal">
                        {[0, 0.5, 1].map((tick) => <span key={tick}>{duration(start + scale * tick)}</span>)}
                      </span>
                    </th></tr>
                </thead>
                <tbody>
                  {visible.map(({ span, depth, offset, elapsed }) => {
                    const status = recordedStatus(span);
                    const active = span.spanId === selected.span.spanId;
                    const barStart = Math.max(start, offset);
                    const barEnd = Math.min(start + scale, offset + elapsed);
                    const inView = barEnd >= barStart;
                    return (
                      <tr key={span.spanId} className={`border-t border-slate-800 ${active ? "bg-cyan-400/10" : "hover:bg-slate-800/50"}`}>
                        <td className="p-3">
                          <div style={{ paddingLeft: sort === "tree" ? Math.min(depth, 6) * 12 : 0 }}>
                            <button type="button" className="break-all text-left font-medium text-cyan-200 hover:underline"
                              aria-pressed={active} onClick={() => setSelectedId(span.spanId)}>{span.name}</button>
                            <p className={`mt-1 text-xs ${status === "failed" ? "text-rose-300" : "text-slate-400"}`}>{readable(status)}</p>
                          </div>
                        </td>
                        <td className="whitespace-nowrap p-3 font-mono text-xs text-slate-300">{duration(elapsed)}</td>
                        <td className="p-3">
                          <button type="button" tabIndex={-1} onClick={() => setSelectedId(span.spanId)}
                            aria-label={`Select ${span.name}, offset ${duration(offset)}, duration ${duration(elapsed)}`}
                            title={`${span.name} · starts at ${duration(offset)} · lasts ${duration(elapsed)}`}
                            className="relative block h-7 w-full overflow-hidden rounded-md border border-slate-800 bg-slate-950/80 hover:border-slate-500">
                            <span className="absolute inset-y-0 left-1/2 border-l border-slate-800" />
                            {inView && <span className={`absolute top-1.5 h-4 min-w-0.5 rounded-sm ${status === "failed" ? "bg-rose-400" : active ? "bg-cyan-300" : "bg-violet-400/80"}`}
                              style={{ left: `${Math.min(99.5, (barStart - start) / scale * 100)}%`, width: `${(barEnd - barStart) / scale * 100}%` }} />}
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : <Empty title="No matching spans">Clear the search or turn off the failures filter.</Empty>}
        </Section>
        <SpanDetail key={selected.span.spanId} row={selected}
          parentAvailable={rows.some(({ span }) => span.spanId === selected.span.parentSpanId)}
          onSelectParent={() => setSelectedId(selected.span.parentSpanId ?? "")} />
      </div>
      <details className={`${panel} px-4 py-3 text-sm`}>
        <summary className="cursor-pointer font-medium text-cyan-200">Execution context</summary>
        <p className="mt-2 break-all font-mono text-xs text-slate-400">Trace {root.span.traceId}</p>
        {query && <p className="mt-2 whitespace-pre-wrap break-words text-slate-200">{query}</p>}
      </details>
    </>
  );
}

export default function SpanInspector({ initialExecutionId = "" }: { initialExecutionId?: string }) {
  const [executionId, setExecutionId] = useState(initialExecutionId);
  const executions = useApi<Page<Execution>>("traces?limit=100");
  const apps = useApi<Page<App>>("apps?limit=100");
  const items = [...(executions.data?.items ?? [])].sort((a, b) => Date.parse(b.started_at) - Date.parse(a.started_at));
  const requested = executionId || items[0]?.execution_id || "";
  const names = new Map(apps.data?.items.map((app) => [app.app_id, app.service_name]));
  const result = useApi<ExportRequest>(requested ? `traces/${encodeURIComponent(requested)}/spans` : null);
  const { rows, parseError } = useMemo(() => {
    try { return { rows: result.data ? timedSpans(result.data) : [], parseError: undefined }; }
    catch (error) { return { rows: [], parseError: error instanceof Error ? error.message : "Invalid span data." }; }
  }, [result.data]);

  return (
    <ConsolePage active="/spans" title="Span inspector"
      description="Explore the timing and parent-child relationships of recorded spans."
      actions={<button type="button" className={button} onClick={() => { executions.refresh(); result.refresh(); }}>
        <RefreshCw size={15} aria-hidden="true" /> Refresh
      </button>}>
      <div className={`${panel} flex flex-wrap items-end gap-4 p-5`}>
        <label className="min-w-0 grow text-sm text-slate-300">Execution
          <select className={field} value={requested} disabled={!requested && !items.length} onChange={(event) => setExecutionId(event.target.value)}>
            {!requested && <option value="">No executions available</option>}
            {requested && !items.some((item) => item.execution_id === requested) && <option value={requested}>{requested}</option>}
            {items.map((item) => (
              <option key={item.execution_id} value={item.execution_id}>
                {names.get(item.app_id) ?? item.app_id} · {new Date(item.started_at).toLocaleString()} · {item.trace_id ?? item.execution_id}
              </option>
            ))}
          </select>
        </label>
        {requested && <Link href={`/traces/${encodeURIComponent(requested)}`} className={button}>Open trace <ArrowUpRight size={15} aria-hidden="true" /></Link>}
        {executions.data?.next_cursor && <p className="basis-full text-xs text-slate-400">Open other executions from the <Link href="/traces" className={link}>trace explorer</Link>.</p>}
      </div>
      <ApiStatus loading={result.loading || (!requested && executions.loading)} error={result.error ?? parseError ?? executions.error} />
      {rows.length > 0 ? <Waterfall key={requested} rows={rows} /> : (result.data || (executions.data && !requested)) && !parseError && (
        <Empty title={requested ? "No spans captured" : "No received executions"}>
          <Link href="/apps/new" className={link}>Connect an application</Link> and send telemetry to inspect spans here.
        </Empty>
      )}
    </ConsolePage>
  );
}
