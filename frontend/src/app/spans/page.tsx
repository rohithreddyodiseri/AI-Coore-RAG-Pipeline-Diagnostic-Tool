"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Attribute = {
  key: string;
  value: {
    stringValue?: string;
    intValue?: string | number;
  };
};

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

type ExportRequest = {
  resourceSpans?: {
    scopeSpans?: {
      spans?: Span[];
    }[];
  }[];
};

function attribute(span: Span, key: string) {
  const value = span.attributes?.find((item) => item.key === key)?.value;
  return value?.stringValue ?? String(value?.intValue ?? "");
}

function failed(span: Span) {
  return (
    span.status?.code === 2 ||
    attribute(span, "rag.stage.status") === "failed" ||
    attribute(span, "rag.execution.status") === "failed"
  );
}

function recordedStatus(span: Span) {
  if (failed(span)) return "failed";

  return (
    attribute(span, "rag.stage.status") ||
    attribute(span, "rag.execution.status") ||
    (span.status?.code === 1 ? "ok" : "unknown")
  );
}

function milliseconds(start: string, end: string) {
  return Number(BigInt(end) - BigInt(start)) / 1_000_000;
}

function timestamp(value: string) {
  return new Date(Number(BigInt(value) / BigInt(1_000_000))).toISOString();
}

function pretty(value: string) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

function buildTree(spans: Span[]) {
  const ids = new Set(spans.map((span) => span.spanId));
  const rows: { span: Span; depth: number }[] = [];
  const visited = new Set<string>();

  function visit(span: Span, depth: number) {
    if (visited.has(span.spanId)) return;

    visited.add(span.spanId);
    rows.push({ span, depth });

    spans
      .filter((child) => child.parentSpanId === span.spanId)
      .forEach((child) => visit(child, depth + 1));
  }

  spans
    .filter((span) => !span.parentSpanId || !ids.has(span.parentSpanId))
    .forEach((span) => visit(span, 0));

  spans.forEach((span) => visit(span, 0));
  return rows;
}

export default function SpanInspectorPage() {
  const [spans, setSpans] = useState<Span[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadFixture() {
      try {
        const response = await fetch("/fixtures/failing.otlp.jsonl", {
          signal: controller.signal,
          cache: "no-store",
        });

        if (!response.ok) {
          throw new Error(
            "Could not load the fixture. Check public/fixtures/failing.otlp.jsonl."
          );
        }

        const text = await response.text();
        const requests = text
          .split(/\r?\n/)
          .filter((line) => line.trim())
          .map((line) => JSON.parse(line) as ExportRequest);

        const unique = new Map<string, Span>();

        for (const request of requests) {
          for (const resource of request.resourceSpans ?? []) {
            for (const scope of resource.scopeSpans ?? []) {
              for (const span of scope.spans ?? []) {
                if (!span.traceId || !span.spanId || !span.name) {
                  throw new Error("A span is missing its ID or name.");
                }

                const start = BigInt(span.startTimeUnixNano);
                const end = BigInt(span.endTimeUnixNano);

                if (end < start) {
                  throw new Error("A span ends before its start time.");
                }

                timestamp(span.startTimeUnixNano);
                timestamp(span.endTimeUnixNano);
                unique.set(`${span.traceId}:${span.spanId}`, span);
              }
            }
          }
        }

        const loaded = Array.from(unique.values()).sort((a, b) => {
          const first = BigInt(a.startTimeUnixNano);
          const second = BigInt(b.startTimeUnixNano);
          return first < second ? -1 : first > second ? 1 : 0;
        });

        if (loaded.length === 0) {
          throw new Error("The fixture contains no spans.");
        }

        if (new Set(loaded.map((span) => span.traceId)).size !== 1) {
          throw new Error(
            "This page expects one trace. The file contains multiple traces."
          );
        }

        if (!controller.signal.aborted) {
          setSpans(loaded);
          setSelectedId(
            loaded.find((span) => !span.parentSpanId)?.spanId ??
              loaded[0].spanId
          );
        }
      } catch (caught) {
        if (!controller.signal.aborted) {
          setError(
            caught instanceof Error
              ? caught.message
              : "Could not load fixture."
          );
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }

    void loadFixture();
    return () => controller.abort();
  }, []);

  const selected = spans.find((span) => span.spanId === selectedId);
  const root = spans.find((span) => !span.parentSpanId) ?? spans[0];
  const rows = buildTree(spans);

  const start = spans.reduce(
    (minimum, span) =>
      BigInt(span.startTimeUnixNano) < minimum
        ? BigInt(span.startTimeUnixNano)
        : minimum,
    spans.length ? BigInt(spans[0].startTimeUnixNano) : BigInt(0)
  );

  const end = spans.reduce(
    (maximum, span) =>
      BigInt(span.endTimeUnixNano) > maximum
        ? BigInt(span.endTimeUnixNano)
        : maximum,
    start
  );

  const totalDuration = Number(end - start) / 1_000_000;
  const timingScale = Math.max(totalDuration, 0.001);

  const errorType = selected ? attribute(selected, "error.type") : "";
  const input = selected ? attribute(selected, "rag.stage.input_json") : "";
  const output = selected ? attribute(selected, "rag.stage.output_json") : "";

  return (
    <main className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <div className="mx-auto max-w-7xl space-y-6">
        <Link href="/traces" className="text-sm text-cyan-400">
          ← Back to trace explorer
        </Link>

        <header>
          <div className="flex flex-wrap items-center justify-between gap-4">
            <h1 className="mt-4 text-3xl font-semibold">
              Span inspector
            </h1>
            <span className="rounded-full border border-amber-400/30 px-4 py-2 text-sm text-amber-300">
              Person 3 fixture · not live data
            </span>
          </div>
          <p className="mt-3 text-sm text-slate-400">
            Recorded parent-child relationships, relative timings, and
            runtime errors. This fixture has no diagnostic verdict attached.
          </p>
        </header>

        {loading && (
          <p role="status" className="text-slate-400">
            Loading trace fixture…
          </p>
        )}

        {error && (
          <p
            role="alert"
            className="rounded-xl border border-red-400/30 bg-red-400/10 p-5 text-red-300"
          >
            {error}
          </p>
        )}

        {!loading && !error && root && (
          <>
            <section className="rounded-xl border border-slate-800 bg-slate-900 p-5">
              <p className="break-all font-mono text-xs text-cyan-400">
                Trace: {root.traceId}
              </p>
              <p className="mt-3 text-sm">
                {attribute(root, "rag.query.text") || root.name}
              </p>
              <p className="mt-3 text-sm text-slate-400">
                {spans.length} spans · {totalDuration.toFixed(3)} ms ·{" "}
                Root status: {recordedStatus(root)}
              </p>
            </section>

            <section className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-900">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-slate-800 text-slate-400">
                  <tr>
                    <th scope="col" className="p-4">Span tree</th>
                    <th scope="col" className="p-4">Recorded status</th>
                    <th scope="col" className="p-4">Duration</th>
                    <th scope="col" className="min-w-64 p-4">
                      Relative timing
                    </th>
                  </tr>
                </thead>

                <tbody>
                  {rows.map(({ span, depth }) => {
                    const duration = milliseconds(
                      span.startTimeUnixNano,
                      span.endTimeUnixNano
                    );
                    const offset =
                      Number(BigInt(span.startTimeUnixNano) - start) /
                      1_000_000;

                    return (
                      <tr
                        key={span.spanId}
                        className={`border-b border-slate-800 last:border-0 ${
                          selectedId === span.spanId
                            ? "bg-cyan-400/10"
                            : ""
                        }`}
                      >
                        <td
                          className="p-4"
                          style={{ paddingLeft: 16 + depth * 24 }}
                        >
                          <button
                            type="button"
                            aria-pressed={selectedId === span.spanId}
                            onClick={() => setSelectedId(span.spanId)}
                            className="text-left font-mono text-cyan-400 hover:underline"
                          >
                            {span.name}
                          </button>
                          <p className="mt-1 text-xs text-slate-500">
                            {span.parentSpanId
                              ? `Parent: ${span.parentSpanId}`
                              : "Root span"}
                          </p>
                        </td>

                        <td
                          className={`p-4 ${
                            failed(span) ? "text-red-400" : "text-slate-300"
                          }`}
                        >
                          {recordedStatus(span)}
                        </td>

                        <td className="p-4">
                          {duration.toFixed(3)} ms
                        </td>

                        <td className="p-4">
                          <div
                            role="img"
                            aria-label={`Starts ${offset.toFixed(3)} milliseconds after trace start; lasts ${duration.toFixed(3)} milliseconds`}
                            className="relative h-5 overflow-hidden rounded bg-slate-800"
                          >
                            <div
                              className={`absolute h-full rounded ${
                                failed(span) ? "bg-red-400" : "bg-cyan-400"
                              }`}
                              style={{
                                left: `${(offset / timingScale) * 100}%`,
                                width: `${(duration / timingScale) * 100}%`,
                                minWidth: 2,
                              }}
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>

            {selected && (
              <section
                aria-live="polite"
                className="rounded-xl border border-slate-800 bg-slate-900 p-5"
              >
                <h2 className="text-lg font-semibold">{selected.name}</h2>

                <dl className="mt-4 grid gap-4 text-sm sm:grid-cols-2">
                  <div>
                    <dt className="text-slate-400">Span ID</dt>
                    <dd className="mt-1 break-all font-mono">
                      {selected.spanId}
                    </dd>
                  </div>

                  <div>
                    <dt className="text-slate-400">Parent span ID</dt>
                    <dd className="mt-1 break-all font-mono">
                      {selected.parentSpanId || "None — root span"}
                    </dd>
                  </div>

                  <div>
                    <dt className="text-slate-400">Start (UTC)</dt>
                    <dd className="mt-1 break-all font-mono">
                      {timestamp(selected.startTimeUnixNano)}
                    </dd>
                  </div>

                  <div>
                    <dt className="text-slate-400">End (UTC)</dt>
                    <dd className="mt-1 break-all font-mono">
                      {timestamp(selected.endTimeUnixNano)}
                    </dd>
                  </div>

                  <div>
                    <dt className="text-slate-400">Duration</dt>
                    <dd className="mt-1">
                      {milliseconds(
                        selected.startTimeUnixNano,
                        selected.endTimeUnixNano
                      ).toFixed(3)} ms
                    </dd>
                  </div>

                  <div>
                    <dt className="text-slate-400">Recorded status</dt>
                    <dd className="mt-1">{recordedStatus(selected)}</dd>
                  </div>
                </dl>

                <p className="mt-4 text-xs text-slate-500">
                  Start and end labels show millisecond precision.
                  Durations use the original nanosecond timestamps.
                </p>

                {(failed(selected) || errorType) && (
                  <div className="mt-5 rounded-lg border border-red-400/30 bg-red-400/10 p-4">
                    <h3 className="font-medium text-red-300">
                      Recorded runtime error
                    </h3>
                    <p className="mt-2 font-mono text-sm text-red-300">
                      {errorType || "Error type not supplied"}
                    </p>
                    <p className="mt-3 text-xs text-slate-400">
                      This fixture supplies no exception message or stack
                      trace. A runtime error is not a diagnostic root-cause
                      verdict.
                    </p>
                  </div>
                )}

                <div className="mt-6 grid gap-5 lg:grid-cols-2">
                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold">Recorded input</h3>
                    <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-950 p-4 text-xs leading-6 text-slate-300">
                      {input
                        ? pretty(input)
                        : "No stage input was captured for this span."}
                    </pre>
                  </div>

                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold">Recorded output</h3>
                    <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-950 p-4 text-xs leading-6 text-slate-300">
                      {output
                        ? pretty(output)
                        : "No stage output was captured for this span."}
                    </pre>
                  </div>
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </main>
  );
}