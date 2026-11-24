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

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");

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
                value={serviceName}
                onChange={(event) => setServiceName(event.target.value)}
                placeholder="my-rag-app"
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 outline-none focus:border-cyan-400"
              />
            </div>

            <div>
              <label htmlFor="environment" className="text-sm font-medium">
                Environment
              </label>
              <select
                id="environment"
                value={environment}
                onChange={(event) => setEnvironment(event.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3"
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

            <pre className="overflow-x-auto rounded-lg bg-slate-950 p-4 text-xs leading-7 text-cyan-300">
              {Object.entries(result.env)
                .map(([key, value]) => `${key}=${value}`)
                .join("\n")}
            </pre>

            <p className="text-sm text-slate-400">
              Registration is complete. Your RAG application still needs
              these settings and OpenTelemetry instrumentation to send traces.
            </p>
          </section>
        )}
      </div>
    </main>
  );
}