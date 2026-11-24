"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type RegisteredApp = {
  app_id: string;
  service_name: string;
  framework: string;
  environments: string[];
  last_trace_at: string | null;
};

type AppList = {
  items: RegisteredApp[];
  next_cursor: string | null;
};

export default function ApplicationsPage() {
  const [apps, setApps] = useState<RegisteredApp[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function loadApps(nextCursor?: string) {
    setLoading(true);
    setError("");

    try {
      const url = nextCursor
        ? `/api/apps?cursor=${encodeURIComponent(nextCursor)}`
        : "/api/apps";

      const response = await fetch(url, { cache: "no-store" });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Could not load applications.");
      }

      const result = data as AppList;

      setApps((previous) =>
        nextCursor ? [...previous, ...result.items] : result.items
      );
      setCursor(result.next_cursor);
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "Could not load applications."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadApps();
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <div className="mx-auto max-w-5xl space-y-6">
        <Link href="/" className="text-sm text-cyan-400">
          ← Back to dashboard
        </Link>

        <header className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="mt-4 text-3xl font-semibold">
              Registered applications
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Applications registered with your local diagnostic backend.
            </p>
          </div>

          <Link
            href="/apps/new"
            className="rounded-lg bg-cyan-400 px-4 py-3 text-sm font-medium text-slate-950"
          >
            Connect application
          </Link>
        </header>

        <button
          type="button"
          onClick={() => void loadApps()}
          disabled={loading}
          className="rounded-lg border border-slate-700 px-4 py-2 text-sm disabled:opacity-50"
        >
          Refresh applications
        </button>

        {error && (
          <p
            role="alert"
            className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300"
          >
            {error}
          </p>
        )}

        {loading && (
          <p role="status" className="text-sm text-slate-400">
            Loading applications…
          </p>
        )}

        {!loading && !error && apps.length === 0 && (
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
            <h2 className="font-semibold">No applications registered yet</h2>
            <p className="mt-2 text-sm text-slate-400">
              Select Connect application to register your first RAG service.
            </p>
          </div>
        )}

        <section className="grid gap-4 md:grid-cols-2">
          {apps.map((app) => (
            <article
              key={app.app_id}
              className="rounded-xl border border-slate-800 bg-slate-900 p-6"
            >
              <h2 className="text-lg font-semibold">{app.service_name}</h2>

              <p className="mt-2 break-all font-mono text-xs text-slate-500">
                {app.app_id}
              </p>

              <p className="mt-4 text-sm text-slate-400">
                Framework: {app.framework}
              </p>

              <p className="mt-2 text-sm text-slate-400">
                Environments: {app.environments.join(", ")}
              </p>

              <p
                className={`mt-5 text-sm ${
                  app.last_trace_at
                    ? "text-emerald-400"
                    : "text-amber-300"
                }`}
              >
                {app.last_trace_at
                  ? `Last trace received: ${new Date(
                      app.last_trace_at
                    ).toLocaleString()}`
                  : "No traces received yet"}
              </p>
            </article>
          ))}
        </section>

        {cursor && (
          <button
            type="button"
            disabled={loading}
            onClick={() => void loadApps(cursor)}
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm disabled:opacity-50"
          >
            Load more
          </button>
        )}
      </div>
    </main>
  );
}