export default function ApiStatus({ loading, error }: { loading?: boolean; error?: string }) {
  if (error) return <p role="alert" className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-300">{error}</p>;
  if (loading) return <p role="status" className="text-slate-400">Loading diagnostic data…</p>;
  return null;
}
