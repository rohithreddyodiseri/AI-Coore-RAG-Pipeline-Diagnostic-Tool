import SpanInspector from "@/features/traces/SpanInspector";

export default async function SpansPage({ searchParams }: {
  searchParams: Promise<{ execution_id?: string | string[] }>;
}) {
  const { execution_id } = await searchParams;
  const initialId = typeof execution_id === "string" ? execution_id : "";
  return <SpanInspector key={initialId} initialExecutionId={initialId} />;
}
