/**
 * PipelineDAG.tsx
 * Live pipeline DAG using ReactFlow — shows each stage with color-coded status.
 * Polls state via WebSocket for live updates.
 */

import React, { useMemo } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  MarkerType,
} from 'reactflow'
import 'reactflow/dist/style.css'
import type { StageStatuses } from '../api/client'

interface Props {
  stageStatuses: StageStatuses
  currentStage: string
  isPaused: boolean
  governanceIterations: number
}

const STATUS_COLORS: Record<string, { bg: string; border: string; text: string; glow: string }> = {
  pending:           { bg: '#1e293b', border: '#475569', text: '#94a3b8', glow: 'none' },
  running:           { bg: '#1e3a5f', border: '#3b82f6', text: '#93c5fd', glow: '0 0 12px rgba(59,130,246,0.4)' },
  awaiting_approval: { bg: '#451a03', border: '#f59e0b', text: '#fbbf24', glow: '0 0 16px rgba(245,158,11,0.5)' },
  approved:          { bg: '#052e16', border: '#16a34a', text: '#4ade80', glow: '0 0 10px rgba(22,163,74,0.3)' },
  complete:          { bg: '#052e16', border: '#22c55e', text: '#4ade80', glow: 'none' },
  rejected:          { bg: '#450a0a', border: '#dc2626', text: '#f87171', glow: '0 0 12px rgba(220,38,38,0.4)' },
  looped_back:       { bg: '#450a0a', border: '#f97316', text: '#fb923c', glow: '0 0 12px rgba(249,115,22,0.4)' },
  failed:            { bg: '#450a0a', border: '#dc2626', text: '#f87171', glow: 'none' },
  skipped:           { bg: '#1e293b', border: '#475569', text: '#64748b', glow: 'none' },
}

const STAGE_LABELS: Record<string, string> = {
  objective_intake:   '📋 Objective Intake',
  compliance:         '⚖️ Compliance',
  data_profiling:     '🔍 Data Profiling',
  feature_engineering:'⚙️ Feature Engineering',
  model_selection:    '🤖 Model Selection',
  governance:         '🛡️ Governance Audit',
  explainability:     '💡 Explainability',
  reporting:          '📄 Reporting',
}

const STAGE_ORDER = [
  'objective_intake',
  'compliance',
  'data_profiling',
  'feature_engineering',
  'model_selection',
  'governance',
  'explainability',
  'reporting',
]

function StatusBadge({ status }: { status: string }) {
  const labels: Record<string, string> = {
    pending: 'Pending',
    running: 'Running...',
    awaiting_approval: 'Awaiting Approval',
    approved: 'Approved',
    complete: 'Complete',
    rejected: 'Rejected',
    looped_back: 'Loop-back',
    failed: 'Failed',
  }
  return <span style={{ fontSize: 10, opacity: 0.85 }}>{labels[status] || status}</span>
}

export default function PipelineDAG({ stageStatuses, currentStage, isPaused, governanceIterations }: Props) {
  const nodes: Node[] = useMemo(() => {
    return STAGE_ORDER.map((stage, i) => {
      const status = (stageStatuses as any)[stage] || 'pending'
      const colors = STATUS_COLORS[status] || STATUS_COLORS.pending
      const isActive = stage === currentStage

      return {
        id: stage,
        type: 'default',
        position: { x: 300, y: i * 110 },
        data: {
          label: (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: colors.text }}>
                {STAGE_LABELS[stage] || stage}
              </div>
              <StatusBadge status={status} />
              {stage === 'governance' && governanceIterations > 0 && (
                <div style={{ fontSize: 10, marginTop: 4, color: '#f97316' }}>
                  Loop #{governanceIterations}
                </div>
              )}
            </div>
          ),
        },
        style: {
          background: colors.bg,
          border: `2px solid ${colors.border}`,
          borderRadius: 12,
          padding: '12px 20px',
          minWidth: 200,
          boxShadow: isActive ? colors.glow : 'none',
          animation: status === 'awaiting_approval' ? 'pulse 2s infinite' : undefined,
          transition: 'all 0.3s ease',
        },
      }
    })
  }, [stageStatuses, currentStage, isPaused, governanceIterations])

  const edges: Edge[] = useMemo(() => {
    const baseEdges: Edge[] = STAGE_ORDER.slice(0, -1).map((stage, i) => ({
      id: `e-${stage}-${STAGE_ORDER[i + 1]}`,
      source: stage,
      target: STAGE_ORDER[i + 1],
      markerEnd: { type: MarkerType.ArrowClosed, color: '#475569' },
      style: { stroke: '#475569', strokeWidth: 2 },
      animated: STAGE_ORDER[i + 1] === currentStage,
    }))

    // Governance loopback edges
    if (governanceIterations > 0) {
      baseEdges.push({
        id: 'e-gov-loopback-fe',
        source: 'governance',
        target: 'feature_engineering',
        markerEnd: { type: MarkerType.ArrowClosed, color: '#f97316' },
        style: { stroke: '#f97316', strokeWidth: 2, strokeDasharray: '5,5' },
        label: 'FAIL → retry',
        labelStyle: { fill: '#fb923c', fontSize: 10 },
      })
    }

    return baseEdges
  }, [currentStage, governanceIterations])

  return (
    <div style={{ width: '100%', height: 900, borderRadius: 16, overflow: 'hidden' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#334155" gap={20} />
        <Controls style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
        <MiniMap
          nodeColor={(n) => {
            const status = n.style?.border?.toString().replace('2px solid ', '') || '#475569'
            return status
          }}
          style={{ background: '#1e293b', border: '1px solid #334155' }}
        />
      </ReactFlow>
    </div>
  )
}
