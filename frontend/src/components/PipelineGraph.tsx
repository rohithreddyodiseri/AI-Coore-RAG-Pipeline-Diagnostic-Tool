"use client";

import { useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MarkerType,
  Position,
  type Node,
  type Edge,
} from "@xyflow/react";

type PipelineNode = {
  node_id: string;
  stage: string;
  label: string;
  status: string;
  duration_ms: number;
};

type PipelineEdge = {
  from: string;
  to: string;
  kind: string;
};

type Props = {
  graph: {
    nodes: PipelineNode[];
    edges: PipelineEdge[];
  };
  selectedNodeId: string;
  primaryStage: string;
  onSelectNode: (nodeId: string) => void;
};

export default function PipelineGraph({
  graph,
  selectedNodeId,
  primaryStage,
  onSelectNode,
}: Props) {
  const nodes = useMemo<Node[]>(
    () =>
      graph.nodes.map((node, index) => {
        const selected = node.node_id === selectedNodeId;
        const implicated = node.stage === primaryStage;

        return {
          id: node.node_id,
          position: { x: index * 290, y: 50 },
          sourcePosition: Position.Right,
          targetPosition: Position.Left,
          selected,
          data: {
            label: (
              <div className="text-left">
                <p className="text-xs text-slate-400">
                  Stage {index + 1}
                </p>

                <p className="mt-2 text-sm font-semibold text-slate-100">
                  {node.label}
                </p>

                <p className="mt-2 text-xs text-slate-400">
                  Runtime: {node.status} · {node.duration_ms} ms
                </p>

                {implicated && (
                  <p className="mt-3 text-xs text-amber-300">
                    Primary diagnosed fault
                  </p>
                )}
              </div>
            ),
          },
          style: {
            width: 235,
            minHeight: 130,
            padding: 18,
            borderRadius: 12,
            background: selected ? "#0c293b" : "#0f172a",
            border: `2px solid ${
              selected ? "#22d3ee" : implicated ? "#fbbf24" : "#334155"
            }`,
            boxShadow: selected
              ? "0 0 18px rgba(34, 211, 238, 0.12)"
              : "none",
          },
        };
      }),
    [graph.nodes, selectedNodeId, primaryStage]
  );

  const edges = useMemo<Edge[]>(
    () =>
      graph.edges.map((edge, index) => ({
        id: `${edge.from}-${edge.to}-${index}`,
        source: edge.from,
        target: edge.to,
        type: "smoothstep",
        style: {
          stroke: "#64748b",
          strokeWidth: 2,
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: "#64748b",
        },
      })),
    [graph.edges]
  );

  return (
    <div className="mt-5 h-[320px] overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodeClick={(_, node) => onSelectNode(node.id)}
        nodesDraggable={false}
        nodesConnectable={false}
        edgesReconnectable={false}
        deleteKeyCode={null}
        fitView
        fitViewOptions={{ padding: "20%" }}
        forceColorMode="dark"
        minZoom={0.25}
        maxZoom={1.5}
      >
        <Background color="#1e293b" gap={22} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}