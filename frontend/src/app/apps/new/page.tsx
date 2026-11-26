"use client";

import { Activity, ArrowRight, Cable, Check, Copy, Server, Terminal } from "lucide-react";
import Link from "next/link";
import { useState, type FormEvent } from "react";
import ApiStatus from "@/components/ApiStatus";
import { Badge, ConsolePage, Section, button, field, link, panel, primary } from "@/features/console/ui";
import { useApi } from "@/hooks/use-api";
import type { App } from "@/lib/api";

const SERVICE_PATTERN = /^[a-z0-9][a-z0-9._-]{0,127}$/;
const SERVICE_HELP = "Use 1–128 lowercase letters, numbers, dots, underscores, or hyphens. Start with a letter or number.";
const steps = [
  { id: "register", label: "Register service", icon: Server },
  { id: "configure", label: "Configure exporter", icon: Terminal },
  { id: "verify", label: "Verify telemetry", icon: Activity },
] as const;
type Step = (typeof steps)[number]["id"];
type Registration = { app: App; env: Record<string, string>; span_contract_url: string };
const waiting = (app: App) => !app.last_trace_at;

export default function RegisterAppPage() {
  const [service, setService] = useState("");
  const [environment, setEnvironment] = useState("staging");
  const [step, setStep] = useState<Step>("register");
  const [registering, setRegistering] = useState(false);
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [result, setResult] = useState<Registration>();
  const [copy, setCopy] = useState<"copied" | "error">();
  const app = useApi<App>(result ? `apps/${encodeURIComponent(result.app.app_id)}` : null, {
    pollIntervalMs: 3000, pollWhile: waiting,
  });
  const lastTrace = app.data?.last_trace_at;
  const settings = result && Object.entries(result.env).map(([key, value]) => `${key}=${value}`).join("\n");

  async function register(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (registering) return;
    const name = service.trim();
    setError("");
    setFieldErrors({});
    if (!SERVICE_PATTERN.test(name)) {
      setFieldErrors({ service_name: SERVICE_HELP });
      return;
    }
    setRegistering(true);
    try {
      const response = await fetch("/api/apps", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ service_name: name, framework: "custom", environments: [environment] }),
      });
      const data = await response.json();
      if (!response.ok) {
        setFieldErrors(data.field_errors ?? {});
        setError(data.message ?? "Registration failed.");
        return;
      }
      setResult(data);
      setStep("configure");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Registration failed.");
    } finally {
      setRegistering(false);
    }
  }

  async function copySettings() {
    if (!settings) return;
    try {
      await navigator.clipboard.writeText(settings);
      setCopy("copied");
    } catch {
      setCopy("error");
    }
  }

  return (
    <ConsolePage active="/apps/new" title="Connect application"
      description="Register a RAG service, configure its OpenTelemetry exporter, and verify its first trace."
      actions={<Link href="/apps" className={button}>View applications</Link>}>
      <ol className="grid gap-3 sm:grid-cols-3" aria-label="Connection setup">
        {steps.map(({ id, label, icon: Icon }) => {
          const complete = id === "register" ? Boolean(result) : Boolean(lastTrace);
          return (
            <li key={id}>
              <button type="button" aria-current={step === id ? "step" : undefined} disabled={!result && id !== "register"}
                onClick={() => setStep(id)}
                className={`${panel} flex w-full items-center gap-3 p-4 text-left hover:border-slate-600 aria-[current=step]:border-cyan-400/60 aria-[current=step]:bg-cyan-400/10 disabled:opacity-55`}>
                <span className={`grid size-9 shrink-0 place-items-center rounded-lg ${complete ? "bg-emerald-400/10 text-emerald-300" : "bg-slate-800 text-cyan-200"}`}>
                  {complete ? <Check size={17} aria-hidden="true" /> : <Icon size={17} aria-hidden="true" />}
                </span>
                <span className="text-sm font-semibold">{label}</span>
              </button>
            </li>
          );
        })}
      </ol>

      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1.7fr)_minmax(0,1fr)]">
        {step === "register" && !result && (
          <Section title="Service details" description="Use the same service name in your OpenTelemetry exporter.">
            <form noValidate className="space-y-5" onSubmit={register}>
              <label className="block text-sm text-slate-300" htmlFor="service">Service name
                <input className={field} id="service" required maxLength={128} autoCapitalize="none" autoCorrect="off"
                  spellCheck={false} aria-describedby="service-help service-error" aria-invalid={Boolean(fieldErrors.service_name)}
                  value={service} placeholder="my-rag-app" disabled={registering} onChange={(event) => {
                    setService(event.target.value);
                    setFieldErrors({});
                    setError("");
                  }} />
              </label>
              <div className="-mt-3">
                <p id="service-help" className="text-xs text-slate-400">{SERVICE_HELP}</p>
                <p id="service-error" role={fieldErrors.service_name ? "alert" : undefined} className="mt-1 text-sm text-rose-300">{fieldErrors.service_name}</p>
              </div>
              <fieldset disabled={registering}>
                <legend className="mb-2 text-sm text-slate-300">Environment</legend>
                <div className="grid gap-2 sm:grid-cols-2">
                  {["staging", "production"].map((value) => (
                    <label key={value} className="flex cursor-pointer items-center gap-3 rounded-lg border border-slate-700 bg-slate-950 p-3 capitalize has-checked:border-cyan-400/55 has-checked:bg-cyan-400/10">
                      <input className="accent-cyan-300" type="radio" name="environment" value={value}
                        checked={environment === value} onChange={() => setEnvironment(value)} />
                      {value}
                    </label>
                  ))}
                </div>
                {Object.entries(fieldErrors).filter(([key]) => key.startsWith("environments")).map(([key, message]) =>
                  <p key={key} role="alert" className="mt-2 text-sm text-rose-300">{message}</p>)}
              </fieldset>
              <ApiStatus error={error} />
              <button type="submit" className={primary} disabled={registering || !service.trim()}>
                {registering ? "Registering…" : "Register application"} <ArrowRight size={16} aria-hidden="true" />
              </button>
            </form>
          </Section>
        )}

        {step === "register" && result && (
          <Section title="Application registered" description={result.app.service_name}>
            <p className="break-all font-mono text-xs text-slate-400">{result.app.app_id}</p>
            <button type="button" className={`${primary} mt-4`} onClick={() => setStep("configure")}>Configure exporter <ArrowRight size={16} aria-hidden="true" /></button>
          </Section>
        )}

        {step === "configure" && result && (
          <Section title="Exporter settings" description="Add these environment variables on your RAG service, then run a query with OpenTelemetry enabled.">
            <p className="mb-3 text-sm text-amber-200">Save these settings before leaving. The ingestion key is shown once.</p>
            <pre tabIndex={0} aria-label="Exporter environment settings" className="overflow-auto rounded-lg border border-slate-800 bg-slate-950 p-4 font-mono text-xs leading-6 text-cyan-200">{settings}</pre>
            <p className="mt-3 text-sm leading-6 text-slate-400">
              Set the OpenTelemetry resource attribute <code className="text-slate-200">deployment.environment.name</code> to <strong className="text-slate-200">{environment}</strong> so traces appear under that environment.
            </p>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => void copySettings()} className={button}><Copy size={15} aria-hidden="true" /> Copy exporter settings</button>
              <button type="button" onClick={() => setStep("verify")} className={primary}>Verify telemetry <ArrowRight size={16} aria-hidden="true" /></button>
            </div>
            {copy && <p role="status" className="mt-3 text-sm text-slate-300">{copy === "copied" ? "Exporter settings copied." : "Clipboard unavailable. Select and copy the settings from the box."}</p>}
            {result.span_contract_url && <a className={`mt-4 inline-block ${link}`} href={result.span_contract_url} target="_blank" rel="noreferrer">Open OpenTelemetry span contract ↗</a>}
          </Section>
        )}

        {step === "verify" && result && (
          <Section title="Telemetry delivery" description="The connection status updates when the diagnostic backend receives telemetry.">
            <ApiStatus loading={app.loading} error={app.error} />
            {app.data && <div role="status" className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
              <Badge tone={lastTrace ? "green" : "amber"}>{lastTrace ? "Telemetry received" : "Awaiting first trace"}</Badge>
              <p className="mt-2 text-sm text-slate-300">{lastTrace
                ? `Last trace received ${new Date(lastTrace).toLocaleString()}.`
                : "Start or restart the instrumented service and submit a query."}</p>
            </div>}
            <div className="mt-4 flex flex-wrap gap-2">
              <button type="button" onClick={app.refresh} className={button}>Check again</button>
              <Link href={`/traces?app_id=${encodeURIComponent(result.app.app_id)}`} className={primary}>Open trace explorer <ArrowRight size={16} aria-hidden="true" /></Link>
            </div>
          </Section>
        )}

        <Section title="Connection preview" description="Telemetry flows from your application’s exporter to the diagnostic backend.">
          <div className="space-y-3">
            {[
              { icon: Server, name: result?.app.service_name ?? (service.trim() || "Your RAG service"), label: environment, tone: "violet" },
              { icon: Cable, name: "OTLP exporter", label: "HTTP / protobuf", tone: "cyan" },
              { icon: Activity, name: "Diagnostic backend", label: lastTrace ? "Telemetry received" : result ? "Awaiting first trace" : "Register to connect", tone: lastTrace ? "green" : "amber" },
            ].map(({ icon: Icon, name, label, tone }, i) => (
              <div key={i} className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-950/55 p-3">
                <Icon size={17} className="shrink-0 text-cyan-300" aria-hidden="true" />
                <div className="min-w-0 grow"><p className="break-all text-sm font-medium">{name}</p><div className="mt-1"><Badge tone={tone as "violet" | "cyan" | "amber" | "green"}>{label}</Badge></div></div>
              </div>
            ))}
          </div>
        </Section>
      </div>
    </ConsolePage>
  );
}
