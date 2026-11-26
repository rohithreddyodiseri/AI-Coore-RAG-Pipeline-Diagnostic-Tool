"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";

type RegistrationResult = {
  app: {
    app_id: string;
    service_name: string;
  };
  env: Record<string, string>;
};

export default function RegisterAppPage() {
  const [serviceName, setServiceName] = useState("");
  const [environment, setEnvironment] = useState("staging");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<RegistrationResult | null>(null);
  const [copyMessage, setCopyMessage] = useState("");
  const [copyError, setCopyError] = useState("");

  const exporterSettings = result
    ? Object.entries(result.env)
        .map(([key, value]) => `${key}=${value}`)
        .join("\n")
    : "";

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (loading || !serviceName.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    setCopyMessage("");
    setCopyError("");

    try {
      const response = await fetch("/api/apps", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          service_name: serviceName.trim(),
          framework: "custom",
          environments: [environment],
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Registration failed.");
      }

      setResult(data);
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Registration failed."
      );
    } finally {
      setLoading(false);
    }
  }

  async function copyExporterSettings() {
    setCopyMessage("");
    setCopyError("");

    try {
      await navigator.clipboard.writeText(exporterSettings);
      setCopyMessage("Exporter settings copied.");
    } catch {
      setCopyError(
        "Could not copy automatically. Select the settings below and press Cmd + C."
      );
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <div className="mx-auto max-w-2xl space-y-6">
        <Link href="/" className="text-sm text-cyan-400">
          ← Back to dashboard
        </Link>

        <header>
          <h1 className="mt-6 text-3xl font-semibold">
            Connect a RAG application
          </h1>
          <p className="mt-3 text-sm text-slate-400">
            Register your app to receive its telemetry connection settings.
          </p>
        </header>

        {!result ? (
          <form
            onSubmit={handleSubmit}
            className="space-y-5 rounded-xl border border-slate-800 bg-slate-900 p-6"
          >
            <div>
              <label htmlFor="service" className="text-sm font-medium">
                Service name
              </label>
              <input
                id="service"
                required
                disabled={loading}
                value={serviceName}
                onChange={(event) => setServiceName(event.target.value)}
                placeholder="my-rag-app"
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 outline-none focus:border-cyan-400 disabled:opacity-50"
              />
            </div>

            <div>
              <label htmlFor="environment" className="text-sm font-medium">
                Environment
              </label>
              <select
                id="environment"
                disabled={loading}
                value={environment}
                onChange={(event) => setEnvironment(event.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 disabled:opacity-50"
              >
                <option value="staging">Staging</option>
                <option value="production">Production</option>
              </select>
            </div>

            {error && (
              <p role="alert" className="text-sm text-red-400">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading || !serviceName.trim()}
              className="rounded-lg bg-cyan-400 px-5 py-3 font-medium text-slate-950 disabled:opacity-50"
            >
              {loading ? "Registering…" : "Register application"}
            </button>
          </form>
        ) : (
          <section className="space-y-5 rounded-xl border border-emerald-500/30 bg-slate-900 p-6">
            <div>
              <h2 className="text-xl font-semibold text-emerald-400">
                Application registered
              </h2>
              <p className="mt-2">{result.app.service_name}</p>
              <p className="mt-1 break-all font-mono text-xs text-slate-400">
                {result.app.app_id}
              </p>
            </div>

            <p className="text-sm text-amber-300">
              Save these settings privately before leaving. The ingestion
              key is returned only once.
            </p>

            <button
              type="button"
              onClick={copyExporterSettings}
              className="rounded-lg bg-cyan-400 px-4 py-3 text-sm font-medium text-slate-950 hover:bg-cyan-300"
            >
              Copy exporter settings
            </button>

            {copyMessage && (
              <p role="status" className="text-sm text-emerald-400">
                {copyMessage}
              </p>
            )}

            {copyError && (
              <p role="alert" className="text-sm text-amber-300">
                {copyError}
              </p>
            )}

            <pre
              tabIndex={0}
              aria-label="Exporter environment settings"
              className="select-text overflow-x-auto rounded-lg bg-slate-950 p-4 text-xs leading-7 text-cyan-300"
            >
              {exporterSettings}
            </pre>

            <p className="text-sm leading-6 text-slate-400">
              Configure these environment variables on your RAG application
              server and enable its OpenTelemetry instrumentation. Then run
              a query and check the applications list for the first received
              trace.
            </p>

            <p className="text-sm text-amber-300">
              Registration alone does not confirm telemetry delivery.
            </p>

            <Link
              href="/apps"
              className="inline-block rounded-lg border border-slate-700 px-4 py-3 text-sm text-cyan-400 hover:border-cyan-400"
            >
              View applications
            </Link>
          </section>
        )}
      </div>
    </main>
  );
}