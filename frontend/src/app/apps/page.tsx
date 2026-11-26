"use client";

import { ArrowUpRight, Cable, Layers, RefreshCw, Search } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import ApiStatus from "@/components/ApiStatus";
import {
  Badge, ConsolePage, Empty, MeterList, Section, button, field, link, panel, primary, tones,
} from "@/features/console/ui";
import { useApi } from "@/hooks/use-api";
import { type App, type Page, api } from "@/lib/api";

export default function ApplicationsPage() {
  const registry = useApi<Page<App>>("apps?limit=50");
  const [extra, setExtra] = useState<Page<App>>();
  const [loadingMore, setLoadingMore] = useState(false);
  const [moreError, setMoreError] = useState("");
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [environment, setEnvironment] = useState("");
  const [sort, setSort] = useState("recent");

  const cursor = extra ? extra.next_cursor : registry.data?.next_cursor;
  const apps = [...(registry.data?.items ?? []), ...(extra?.items ?? [])];
  const reporting = apps.filter((app) => app.last_trace_at).length;
  const environments = new Map<string, number>();
  for (const app of apps) {
    for (const name of app.environments) environments.set(name, (environments.get(name) ?? 0) + 1);
  }
  const needle = search.trim().toLowerCase();
  const visible = apps.filter((app) =>
    (!needle || `${app.service_name} ${app.app_id} ${app.framework}`.toLowerCase().includes(needle)) &&
    (!status || Boolean(app.last_trace_at) === (status === "reporting")) &&
    (!environment || app.environments.includes(environment)),
  ).sort((a, b) => {
    if (sort === "recent" && a.last_trace_at !== b.last_trace_at) {
      return (b.last_trace_at ?? "").localeCompare(a.last_trace_at ?? "");
    }
    return a.service_name.localeCompare(b.service_name);
  });

  async function loadMore() {
    if (!cursor || loadingMore) return;
    setLoadingMore(true);
    setMoreError("");
    try {
      const page = await api<Page<App>>(`apps?limit=50&cursor=${encodeURIComponent(cursor)}`);
      setExtra((previous) => ({ items: [...(previous?.items ?? []), ...page.items], next_cursor: page.next_cursor }));
    } catch (error) {
      setMoreError(error instanceof Error ? error.message : "Could not load more applications.");
    } finally {
      setLoadingMore(false);
    }
  }

  function refresh() {
    setExtra(undefined);
    setMoreError("");
    registry.refresh();
  }

  return (
    <ConsolePage active="/apps" title="Applications"
      description="Find registered services and see which ones have sent telemetry."
      actions={<><button type="button" className={button} disabled={loadingMore} onClick={refresh}>
        <RefreshCw size={15} aria-hidden="true" /> Refresh
      </button><Link href="/apps/new" className={primary}><Cable size={16} aria-hidden="true" /> Connect application</Link></>}>
      <ApiStatus loading={registry.loading} error={registry.error} />
      {registry.data && apps.length === 0 && (
        <Empty title="No applications registered"><Link className={link} href="/apps/new">Connect your first application</Link> to receive traces.</Empty>
      )}
      {apps.length > 0 && (
        <>
          <div className="grid items-start gap-5 lg:grid-cols-2">
            <Section title="Telemetry status" description={cursor ? `Showing ${apps.length} loaded applications.` : `${apps.length} registered applications.`}
              actions={(status || environment) && <button type="button" className={link} onClick={() => { setStatus(""); setEnvironment(""); }}>Clear chart filters</button>}>
              <div className="flex h-3 overflow-hidden rounded-full bg-slate-800" aria-hidden="true">
                <span className={tones.green.bar} style={{ width: `${reporting / apps.length * 100}%` }} />
                <span className={tones.amber.bar} style={{ width: `${(apps.length - reporting) / apps.length * 100}%` }} />
              </div>
              <div className="mt-4 grid gap-2 sm:grid-cols-2">
                {[
                  { id: "reporting", title: "Received telemetry", count: reporting, tone: "green" },
                  { id: "waiting", title: "Awaiting first trace", count: apps.length - reporting, tone: "amber" },
                ].map(({ id, title, count, tone }) => (
                  <button key={id} type="button" aria-pressed={status === id}
                    className={`${button} justify-between aria-pressed:border-cyan-400/50`}
                    onClick={() => setStatus(status === id ? "" : id)}>
                    <span className="flex items-center gap-2"><span className={`size-2 rounded-full ${tones[tone as "green" | "amber"].bar}`} />{title}</span>
                    <span className="font-mono">{count}</span>
                  </button>
                ))}
              </div>
            </Section>
            <Section title="Registered environments" description="Select an environment to filter the list.">
              <MeterList items={[...environments].sort(([a], [b]) => a.localeCompare(b)).map(([name, value]) => ({
                id: name, label: name, value, tone: "violet",
              }))} active={environment} onSelect={(id) => setEnvironment(environment === id ? "" : id)} />
            </Section>
          </div>

          <Section title="Service registry" description={`${visible.length} of ${apps.length} loaded applications shown`}
            actions={<div className="flex flex-wrap items-center gap-2">
              <label className="relative">
                <Search size={15} aria-hidden="true" className="absolute left-3 top-3.5 text-slate-500" />
                <input className={`${field} mt-0 pl-9`} type="search" value={search}
                  onChange={(event) => setSearch(event.target.value)} aria-label="Search applications" placeholder="Find an application" />
              </label>
              <select className={`${field} mt-0 w-auto`} value={sort} onChange={(event) => setSort(event.target.value)} aria-label="Sort applications">
                <option value="recent">Recent telemetry</option>
                <option value="name">Service name</option>
              </select>
            </div>}>
            {visible.length === 0 && <Empty title="No matching applications">Adjust the search or chart filters.</Empty>}
            <div className="grid gap-3 lg:grid-cols-2">
              {visible.map((app) => (
                <article key={app.app_id} className={`${panel} bg-slate-950/45 p-4`}>
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex min-w-0 items-start gap-3">
                      <div className="grid size-9 shrink-0 place-items-center rounded-lg bg-cyan-400/10 text-cyan-300"><Layers size={17} aria-hidden="true" /></div>
                      <div className="min-w-0">
                        <h3 className="break-all font-semibold">{app.service_name}</h3>
                        <p className="mt-1 break-all font-mono text-xs text-slate-400">{app.app_id}</p>
                      </div>
                    </div>
                    <Badge tone={app.last_trace_at ? "green" : "amber"}>{app.last_trace_at ? "Received telemetry" : "Awaiting first trace"}</Badge>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Badge>{app.framework}</Badge>
                    {app.environments.map((name) => <Badge key={name} tone="violet">{name}</Badge>)}
                  </div>
                  <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-slate-800 pt-3">
                    <p className="text-xs text-slate-400">{app.last_trace_at
                      ? `Last trace ${new Date(app.last_trace_at).toLocaleString()}`
                      : "No traces received yet"}</p>
                    <Link href={`/traces?app_id=${encodeURIComponent(app.app_id)}`} className={`flex items-center gap-1 ${link}`}>
                      View traces <ArrowUpRight size={14} aria-hidden="true" />
                    </Link>
                  </div>
                </article>
              ))}
            </div>
            <ApiStatus loading={loadingMore} error={moreError} />
            {cursor && <button type="button" className={`${button} mt-4`} disabled={loadingMore} onClick={() => void loadMore()}>Load more applications</button>}
          </Section>
        </>
      )}
    </ConsolePage>
  );
}
