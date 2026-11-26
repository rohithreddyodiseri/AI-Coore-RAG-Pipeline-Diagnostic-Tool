export type Outcome = "correct" | "incorrect" | "abstained" | "error";

export type CaseResult = {
  run_id: string;
  kind: "control" | "fault" | "multi_fault";
  description: string;
  expected_root: Attribution | null;
  predicted_root: Attribution | null;
  expected_faults: string[];
  predicted_faults: string[];
  outcome: Outcome;
  status: string;
  correct: boolean;
};

type Attribution = { mechanism: string; technical_stage: string };

export type MechanismMetrics = {
  support: number;
  predicted: number;
  true_positives: number;
  precision: number | null;
  recall: number | null;
};

export type Summary = {
  suite_accuracy: number | null;
  primary_accuracy: number | null;
  stage_accuracy: number | null;
  mechanism_precision: number | null;
  mechanism_recall: number | null;
  attempted: number;
  errors: number;
  model: string | null;
  taxonomy_version: number;
};

export type AccuracyRun = {
  run_id: string;
  state: "queued" | "running" | "completed" | "completed_with_errors" | "failed";
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error_message: string | null;
  total_cases: number;
  finished_cases: number;
  summary: Summary | null;
  report?: { cases: CaseResult[]; mechanism_metrics: Record<string, MechanismMetrics> };
};

export type AccuracyHistory = { items: AccuracyRun[]; suite_cases: number; can_run: boolean };

export const outcomes: Record<Outcome, { label: string; color: string }> = {
  correct: { label: "Correct", color: "#34d399" },
  incorrect: { label: "Incorrect", color: "#fb7185" },
  abstained: { label: "Abstained", color: "#fbbf24" },
  error: { label: "Execution error", color: "#a78bfa" },
};

export const isActive = (run: AccuracyRun) => run.state === "queued" || run.state === "running";
export const pollHistory = (history: AccuracyHistory) => history.items.some(isActive);

export function percent(value: number | null | undefined) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

export function readable(value: string) {
  return value.toLowerCase().replaceAll("_", " ");
}

export function dateLabel(date: string) {
  return new Date(date).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
