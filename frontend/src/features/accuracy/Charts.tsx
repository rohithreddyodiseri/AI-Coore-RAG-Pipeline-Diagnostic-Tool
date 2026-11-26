import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  type AccuracyRun, type CaseResult, type MechanismMetrics, type Outcome,
  dateLabel, outcomes, percent, readable,
} from "./types";

const cyan = "#22d3ee";
const violet = "#a78bfa";
const slate = "#94a3b8";
const grid = "#273449";
const tooltip = { background: "#0f172a", borderColor: "#475569", color: "#e2e8f0", borderRadius: 8 };

export function HistoryChart({ runs }: { runs: AccuracyRun[] }) {
  const data = [...runs].reverse().map((run, index) => ({
    run: index + 1, date: dateLabel(run.created_at), model: run.summary?.model,
    overall: run.summary?.suite_accuracy, primary: run.summary?.primary_accuracy,
    stage: run.summary?.stage_accuracy,
    completed: Boolean(run.summary),
  })).filter((point) => point.completed);

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid vertical={false} stroke={grid} />
        <XAxis dataKey="run" tickFormatter={(run) => `#${run}`} stroke={slate} fontSize={12} />
        <YAxis domain={[0, 1]} tickFormatter={percent} stroke={slate} fontSize={12} />
        <Tooltip
          contentStyle={tooltip}
          labelStyle={{ color: "#f1f5f9" }}
          labelFormatter={(_, payload) => {
            const point = payload[0]?.payload;
            return point ? `${point.date}${point.model ? ` · ${point.model}` : ""}` : "";
          }}
          formatter={(value) => percent(Number(value))}
        />
        <Legend />
        <Line name="Overall accuracy" type="linear" dataKey="overall" stroke={cyan} strokeWidth={3} isAnimationActive={false} />
        <Line name="Primary accuracy" type="linear" dataKey="primary" stroke={violet} strokeWidth={2} isAnimationActive={false} />
        <Line name="Stage accuracy" type="linear" dataKey="stage" stroke="#34d399" strokeWidth={2} strokeDasharray="5 4" isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PrecisionChart({ metrics, comparison, selected, onSelect }: {
  metrics: Record<string, MechanismMetrics>;
  comparison?: Record<string, MechanismMetrics>;
  selected: string;
  onSelect: (mechanism: string) => void;
}) {
  const data = Object.entries(metrics).map(([mechanism, row]) => ({
    mechanism, precision: row.precision, comparison: comparison?.[mechanism]?.precision,
  }));

  return (
    <ResponsiveContainer width="100%" height={data.length * (comparison ? 35 : 28) + 55}>
      <BarChart data={data} layout="vertical" margin={{ top: 6, right: 16, bottom: 0, left: 4 }}>
        <CartesianGrid horizontal={false} stroke={grid} />
        <XAxis type="number" domain={[0, 1]} tickFormatter={percent} stroke={slate} fontSize={12} />
        <YAxis type="category" dataKey="mechanism" width={198} interval={0}
          tickFormatter={readable} stroke={slate} fontSize={11} />
        <Tooltip contentStyle={tooltip} labelStyle={{ color: "#f1f5f9" }}
          labelFormatter={(label) => readable(String(label))} formatter={(value) => percent(Number(value))}
          cursor={{ fill: "#1e293b" }} />
        {comparison && <Legend />}
        <Bar dataKey="precision" name="Selected run" isAnimationActive={false} fill={cyan} radius={[0, 4, 4, 0]} cursor="pointer"
          minPointSize={(value) => value === 0 ? 3 : 0}
          onClick={(_, index) => onSelect(data[index].mechanism)}>
          {data.map(({ mechanism }) => (
            <Cell key={mechanism} fillOpacity={!selected || selected === mechanism ? 1 : 0.4} />
          ))}
        </Bar>
        {comparison && <Bar dataKey="comparison" name="Comparison run" isAnimationActive={false} fill={violet}
          radius={[0, 4, 4, 0]} minPointSize={(value) => value === 0 ? 3 : 0} />}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function OutcomeChart({ cases, selected, onSelect }: {
  cases: CaseResult[];
  selected: Outcome | "";
  onSelect: (outcome: Outcome) => void;
}) {
  const data = (Object.keys(outcomes) as Outcome[]).map((outcome) => ({
    outcome, name: outcomes[outcome].label, value: cases.filter((row) => row.outcome === outcome).length,
  }));

  return (
    <>
      <div className="relative">
        <ResponsiveContainer width="100%" height={218}>
          <PieChart>
            <Tooltip contentStyle={tooltip} itemStyle={{ color: "#f1f5f9" }} />
            <Pie data={data} dataKey="value" nameKey="name" innerRadius={59} outerRadius={89}
              paddingAngle={3} stroke="none" isAnimationActive={false} onClick={(_, index) => onSelect(data[index].outcome)}>
              {data.map(({ outcome }) => (
                <Cell key={outcome} fill={outcomes[outcome].color} cursor="pointer"
                  fillOpacity={!selected || selected === outcome ? 1 : 0.35} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 grid place-content-center text-center">
          <strong className="text-3xl">{cases.length}</strong>
          <span className="text-xs text-slate-400">cases</span>
        </div>
      </div>
      <div className="space-y-2">
        {data.map(({ outcome, name, value }) => (
          <button type="button" key={outcome} onClick={() => onSelect(outcome)}
            aria-pressed={selected === outcome}
            className="flex w-full items-center justify-between gap-3 rounded-lg border border-slate-800 px-3 py-2 text-sm hover:bg-slate-800 aria-pressed:border-cyan-500">
            <span className="flex items-center gap-2">
              <span className="size-2 rounded-full" style={{ backgroundColor: outcomes[outcome].color }} />
              {name}
            </span>
            <span className="font-mono tabular-nums">{value}</span>
          </button>
        ))}
      </div>
    </>
  );
}
