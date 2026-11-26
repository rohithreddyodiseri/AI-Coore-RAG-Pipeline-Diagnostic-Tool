import { Activity, BarChart3, Cable, Layers, ListTree, Network } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

export const panel = "min-w-0 rounded-2xl border border-slate-800 bg-slate-900/70";
export const field = "mt-2 w-full min-w-0 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none";
export const button = "inline-flex items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3.5 py-2.5 text-sm font-medium text-slate-200 hover:border-slate-500 hover:bg-slate-800 disabled:opacity-50";
export const primary = "inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-300 px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-200 disabled:opacity-50";
export const link = "text-sm font-medium text-cyan-300 hover:underline";

const navigation = [
  { href: "/", title: "Control center", icon: Activity },
  { href: "/apps", title: "Applications", icon: Layers },
  { href: "/apps/new", title: "Connect application", icon: Cable },
  { href: "/traces", title: "Trace explorer", icon: Network },
  { href: "/spans", title: "Span inspector", icon: ListTree },
  { href: "/accuracy", title: "Diagnostic Accuracy", icon: BarChart3 },
] as const;

export function ConsolePage({ active, title, description, actions, children }: {
  active: typeof navigation[number]["href"];
  title: string;
  description: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 [color-scheme:dark]">
      <div className="border-b border-slate-800 bg-slate-900/60">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-3 sm:px-8 xl:flex-row xl:items-center xl:gap-7">
          <Link href="/" className="flex w-fit shrink-0 items-center gap-2.5 text-sm font-semibold tracking-tight">
            <span className="grid size-8 place-items-center rounded-lg border border-cyan-500/30 bg-cyan-400/10 text-cyan-300">
              <Activity size={17} aria-hidden="true" />
            </span>
            RAG Diagnostic Lab
          </Link>
          <nav aria-label="Diagnostic tools" className="flex flex-wrap gap-1">
            {navigation.map(({ href, title, icon: Icon }) => (
              <Link key={href} href={href} aria-current={href === active ? "page" : undefined}
                className="flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-slate-400 hover:bg-slate-800 hover:text-slate-100 aria-[current=page]:bg-cyan-400/10 aria-[current=page]:text-cyan-200">
                <Icon size={14} aria-hidden="true" />{title}
              </Link>
            ))}
          </nav>
        </div>
      </div>
      <div className="mx-auto max-w-7xl space-y-6 px-4 py-7 sm:px-8">
        <header className="flex flex-wrap items-end justify-between gap-4 border-b border-slate-800 pb-6">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">{description}</p>
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
        {children}
      </div>
    </main>
  );
}

export function Section({ title, description, actions, children }: {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className={`${panel} p-5`}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-semibold tracking-tight">{title}</h2>
          {description && <p className="mt-1 text-sm leading-5 text-slate-400">{description}</p>}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Metric({ label, value, detail, color = "text-cyan-200" }: {
  label: string; value: ReactNode; detail?: string; color?: string;
}) {
  return (
    <div className={`${panel} p-5`}>
      <p className="text-sm text-slate-300">{label}</p>
      <p className={`mt-3 text-3xl font-semibold tracking-tight tabular-nums ${color}`}>{value}</p>
      {detail && <p className="mt-2 text-xs leading-5 text-slate-400">{detail}</p>}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950/45 px-5 py-8 text-center">
      <p className="font-medium text-slate-200">{title}</p>
      <div className="mt-2 text-sm leading-6 text-slate-400">{children}</div>
    </div>
  );
}

export const tones = {
  cyan: { bar: "bg-cyan-400", badge: "border-cyan-400/25 bg-cyan-400/10 text-cyan-200" },
  green: { bar: "bg-emerald-400", badge: "border-emerald-400/25 bg-emerald-400/10 text-emerald-200" },
  amber: { bar: "bg-amber-400", badge: "border-amber-400/25 bg-amber-400/10 text-amber-200" },
  rose: { bar: "bg-rose-400", badge: "border-rose-400/25 bg-rose-400/10 text-rose-200" },
  violet: { bar: "bg-violet-400", badge: "border-violet-400/25 bg-violet-400/10 text-violet-200" },
  slate: { bar: "bg-slate-400", badge: "border-slate-600 bg-slate-800 text-slate-300" },
};
export type Tone = keyof typeof tones;

export function Badge({ tone = "slate", children }: { tone?: Tone; children: ReactNode }) {
  return <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${tones[tone].badge}`}>{children}</span>;
}

type Meter = {
  id: string; label: string; value: number; formatted?: string;
  tone?: Tone; title?: string;
};

export function MeterList({ items, active, onSelect }: {
  items: Meter[]; active?: string; onSelect: (id: string) => void;
}) {
  const max = Math.max(1, ...items.map((item) => item.value));
  return (
    <div className="max-h-[24rem] space-y-1 overflow-y-auto pr-1">
      {items.map((item) => (
        <button key={item.id} type="button" onClick={() => onSelect(item.id)}
          aria-pressed={item.id === active} title={item.title}
          className="block w-full rounded-lg border border-transparent px-2.5 py-2 text-left text-sm hover:bg-slate-800/85 aria-pressed:border-cyan-400/40 aria-pressed:bg-cyan-400/10">
          <span className="mb-2 flex items-center justify-between gap-3">
            <span className="min-w-0 break-all font-medium text-slate-200">{item.label}</span>
            <span className="shrink-0 font-mono text-xs text-slate-300">{item.formatted ?? item.value.toLocaleString()}</span>
          </span>
          <span className="block h-1.5 overflow-hidden rounded-full bg-slate-800">
            <span className={`block h-full rounded-full ${tones[item.tone ?? "cyan"].bar}`} style={{ width: `${100 * item.value / max}%` }} />
          </span>
        </button>
      ))}
    </div>
  );
}

export function readable(value: string) {
  return value.replaceAll("_", " ");
}

export function duration(value: number | null | undefined) {
  if (value == null) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(value < 1 ? 3 : 2)} ms`;
}
