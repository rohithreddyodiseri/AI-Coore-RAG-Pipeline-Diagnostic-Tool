"use client";

import { Cable, RefreshCw } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import ApiStatus from "@/components/ApiStatus";
import {
  Badge, ConsolePage, Empty, MeterList, Metric, Section,
  button, duration, field, link, panel, primary, readable, tones,
} from "@/features/console/ui";
import { useApi } from "@/hooks/use-api";
import type { App, Counts, Dashboard, Page } from "@/lib/api";

const metrics = [
  { key: "observed_traces", label: "Executions", tone: "cyan" },
  { key: "flagged_traces", label: "Flagged", tone: "amber" },
  { key: "runtime_failed_traces", label: "Runtime failures", tone: "rose" },
] as const;

function recentWindow(hours: number) {
  const now = Date.now();
  const end = Math.ceil(now / 60_000) * 60_000;
  return {
    from: new Date(end - hours * 3_600_000).toISOString().slice(0, 16),
    to: new Date(end).toISOString().slice(0, 16),
  };
}

export default function Home() {
  const [time, setTime] = useState<{ from: string; to: string }>();
  const [preset, setPreset] = useState(24);
  const [appId, setAppId] = useState("");
  const [environment, setEnvironment] = useState("");
  const [metricKey, setMetricKey] = useState<(typeof metrics)[number]["key"]>("observed_traces");
  const [selectedStage, setSelectedStage] = useState("");
  const [initialWindow] = useState(() => recentWindow(24));
  const window = time ?? initialWindow;
  const valid = Boolean(window.from && window.to && window.from < window.to);
  const query = new URLSearchParams({ from: `${window.from}:00Z`, to: `${window.to}:00Z` });
  if (appId) query.set("app_id", appId);
  if (environment.trim()) query.set("environment", environment.trim());
  const health = useApi<Dashboard>(valid ? `analytics/health?${query}` : null);
  const apps = useApi<Page<App>>("apps?limit=100");
  const counts = health.data?.totals;
  const environmentOptions = [...new Set(
    apps.data?.items.filter((app) => !appId || app.app_id === appId).flatMap((app) => app.environments),
  )].sort();
  const reportApps = health.data?.apps ?? [];
  const stages = health.data?.latency_bottlenecks ?? [];
  const stage = stages.find((row) => row.stage === selectedStage) ?? stages[0];
  const metric = metrics.find(({ key }) => key === metricKey) ?? metrics[0];
  const maxAppCount = Math.max(1, ...reportApps.map((app) => app.counts[metric.key]));

  function traces(application?: string, count?: keyof Counts) {
    const filters = new URLSearchParams(query);
    if (application) filters.set("app_id", application);
    if (count === "flagged_traces") filters.set("flagged", "true");
    if (count === "runtime_failed_traces") filters.set("status", "failed");
    return `/traces?${filters}`;
  }

  return (
    <ConsolePage active="/" title="Diagnostic Control Center"
      description="Monitor execution health, compare applications, and locate slow pipeline stages."
      actions={<Link href="/apps/new" className={primary}><Cable size={16} aria-hidden="true" /> Connect application</Link>}>
      <section className={`${panel} p-5`} aria-label="Dashboard filters">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            {[[1, "Last hour"], [24, "Last 24 hours"], [168, "Last 7 days"]].map(([hours, label]) => (
              <button type="button" key={hours} aria-pressed={preset === hours}
                className={`${button} aria-pressed:border-cyan-400/50 aria-pressed:text-cyan-200`}
                onClick={() => { setPreset(Number(hours)); setTime(recentWindow(Number(hours))); }}>
                {label}
              </button>
            ))}
          </div>
          <button type="button" className={button} onClick={() => {
            if (preset) setTime(recentWindow(preset));
            health.refresh();
            apps.refresh();
          }}><RefreshCw size={15} aria-hidden="true" /> Refresh</button>
        </div>
        <div className="mt-4 grid gap-4 text-sm text-slate-300 sm:grid-cols-2 xl:grid-cols-4">
          <label>Application
            <select className={field} value={appId} onChange={(event) => {
              setAppId(event.target.value);
              setEnvironment("");
            }}>
              <option value="">All applications</option>
              {apps.data?.items.map((app) => <option key={app.app_id} value={app.app_id}>{app.service_name}</option>)}
            </select>
          </label>
          <label>Environment
            <input className={field} list="dashboard-environments" value={environment}
              onChange={(event) => setEnvironment(event.target.value)} placeholder="All environments" />
            <datalist id="dashboard-environments">
              {environmentOptions.map((item) => <option key={item} value={item} />)}
            </datalist>
          </label>
          <label>From (UTC)
            <input className={field} type="datetime-local" suppressHydrationWarning value={window.from}
              onChange={(event) => { setPreset(0); setTime({ ...window, from: event.target.value }); }} />
          </label>
          <label>To (UTC, exclusive)
            <input className={field} type="datetime-local" suppressHydrationWarning value={window.to}
              onChange={(event) => { setPreset(0); setTime({ ...window, to: event.target.value }); }} />
          </label>
        </div>
      </section>

      {!valid && <ApiStatus error="Choose a start and end time with the end after the start." />}
      <ApiStatus loading={health.loading} error={health.error ?? apps.error} />

      {health.data && counts && (
        <>
          <section aria-label="Health metrics" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Link href={traces()} className="rounded-2xl hover:ring-1 hover:ring-cyan-400/50">
              <Metric label="Observed executions" value={counts.observed_traces.toLocaleString()} detail="Open trace explorer" />
            </Link>
            <Link href={traces(undefined, "flagged_traces")} className="rounded-2xl hover:ring-1 hover:ring-amber-400/50">
              <Metric label="Quality flags" value={counts.flagged_traces.toLocaleString()} detail="Inspect flagged executions" color="text-amber-200" />
            </Link>
            <Link href={traces(undefined, "runtime_failed_traces")} className="rounded-2xl hover:ring-1 hover:ring-rose-400/50">
              <Metric label="Runtime failures" value={counts.runtime_failed_traces.toLocaleString()} detail="Inspect failed executions" color="text-rose-200" />
            </Link>
            <Metric label="Abstention rate" value={counts.abstention_rate == null ? "—" : `${(counts.abstention_rate * 100).toFixed(1)}%`}
              detail={`${counts.abstained_traces} abstained · ${counts.analyzed_flagged_traces} analyzed`} color="text-violet-200" />
          </section>
          <div className="grid items-start gap-5 lg:grid-cols-2">
            <Section title="Application activity" description="Select an application to inspect its executions."
              actions={<select aria-label="Application activity metric" className={`${field} mt-0 w-auto`}
                value={metricKey} onChange={(event) => setMetricKey(event.target.value as typeof metricKey)}>
                {metrics.map(({ key, label }) => <option key={key} value={key}>{label}</option>)}
              </select>}>
              {reportApps.length ? (
                <div className="max-h-96 space-y-1 overflow-y-auto">
                  {[...reportApps].sort((a, b) => b.counts[metric.key] - a.counts[metric.key]).map((app) => (
                    <Link key={app.app_id} href={traces(app.app_id, metric.key)}
                      className="block rounded-lg p-2.5 text-sm hover:bg-slate-800/80">
                      <span className="mb-2 flex items-center justify-between gap-3">
                        <span className="break-all font-medium">{app.service_name}</span>
                        <span className="font-mono text-xs text-slate-300">{app.counts[metric.key].toLocaleString()}</span>
                      </span>
                      <span className="block h-1.5 overflow-hidden rounded-full bg-slate-800">
                        <span className={`block h-full rounded-full ${tones[metric.tone].bar}`}
                          style={{ width: `${100 * app.counts[metric.key] / maxAppCount}%` }} />
                      </span>
                    </Link>
                  ))}
                </div>
              ) : <Empty title="No application activity"><Link className={link} href="/apps/new">Connect an application</Link> to start collecting telemetry.</Empty>}
            </Section>
            <Section title="Stage latency" description="Select a stage to see its average and sample count." actions={<Badge tone="cyan">Average</Badge>}>
              {stage ? <>
                <MeterList items={stages.map((row) => ({
                  id: row.stage, label: readable(row.stage), value: row.avg_ms, formatted: duration(row.avg_ms),
                  tone: "violet", title: `${row.sample_count} samples`,
                }))} active={stage.stage} onSelect={setSelectedStage} />
                <div className="mt-4 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-sm" aria-live="polite">
                  <span className="font-medium capitalize">{readable(stage.stage)}</span>
                  <span className="text-slate-400">{duration(stage.avg_ms)} average · {stage.sample_count.toLocaleString()} samples</span>
                </div>
              </> : <Empty title="No stage timings">Stage timings appear when executions with recorded spans match this window.</Empty>}
            </Section>
          </div>
          <p className="text-xs text-slate-400">Telemetry sampling coverage is {health.data.coverage}.</p>
        </>
      )}
    </ConsolePage>
  );
}
