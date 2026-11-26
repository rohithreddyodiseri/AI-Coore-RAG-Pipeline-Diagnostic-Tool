export async function api<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`/api/${path}`, { cache: "no-store", signal });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message ?? "Could not load diagnostic data.");
  return data as T;
}

export type Page<T> = { items: T[]; next_cursor: string | null };
export type App = {
  app_id: string;
  service_name: string;
  framework: string;
  environments: string[];
  last_trace_at: string | null;
};
export type Execution = {
  execution_id: string;
  trace_id: string | null;
  app_id: string;
  environment: string | null;
  started_at: string;
  duration_ms: number | null;
  status: "completed" | "failed";
  flagged: boolean;
  diagnosis_state: string;
};
export type Artifact = {
  artifact_id: string;
  kind: string;
  availability: string;
  content?: unknown;
};
export type PipelineNode = {
  node_id: string;
  stage: string;
  label: string;
  status: string;
  duration_ms: number | null;
  artifact_ids: string[];
};
export type Verdict = {
  status: string;
  primary_stage: string | null;
  category: string | null;
  mechanism?: string | null;
  evaluations?: { mechanism: string; status: string; rationale: string; confidence: number }[];
  unresolved_mechanisms?: { mechanism: string; questions: string[] }[];
  confidence: number | null;
  summary: string;
  evidence_trail: { finding: string; citations: { artifact_id: string; quote: string | null }[] }[];
  recommendations: { action: string }[];
};
export type TraceDetail = {
  execution: Execution;
  graph: { nodes: PipelineNode[]; edges: { from: string; to: string; kind: string }[] };
  execution_artifact_ids?: string[];
  latest_live_diagnosis: { state: string; verdict?: Verdict; error?: { message: string } } | null;
};
export type Counts = {
  observed_traces: number;
  flagged_traces: number;
  runtime_failed_traces: number;
  analyzed_flagged_traces: number;
  abstained_traces: number;
  abstention_rate: number | null;
};
export type Dashboard = {
  window: { from: string; to: string };
  coverage: string;
  totals: Counts;
  apps: { app_id: string; service_name: string; counts: Counts }[];
  latency_bottlenecks: { stage: string; avg_ms: number; sample_count: number }[];
};
