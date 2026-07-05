/**
 * PipelineDAG.tsx
 * Live pipeline DAG using ReactFlow — shows each stage with color-coded status.
 * Beautiful, interactive, horizontal layout with custom nodes and animated edges.
 */

import React, { useMemo } from 'react'
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  MarkerType,
  Handle,
  Position
} from 'reactflow'
import 'reactflow/dist/style.css'
import {
  CheckCircle2, AlertCircle, Clock, Loader2, PlayCircle, ShieldCheck, 
  Search, Settings, Cpu, Shield, Lightbulb, FileText, FastForward
} from 'lucide-react'
import type { StageStatuses } from '../api/client'
import clsx from 'clsx'

interface Props {
  stageStatuses: StageStatuses
  currentStage: string
  isPaused: boolean
  governanceIterations: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom Node Design
// ─────────────────────────────────────────────────────────────────────────────

const STATUS_THEMES: Record<string, { bg: string, border: string, text: string, iconColor: string, glow: string }> = {
  pending:           { bg: 'bg-surface-800/80', border: 'border-surface-600', text: 'text-slate-400', iconColor: 'text-slate-500', glow: '' },
  running:           { bg: 'bg-blue-900/30', border: 'border-blue-500/50', text: 'text-blue-100', iconColor: 'text-blue-400', glow: 'shadow-[0_0_15px_rgba(59,130,246,0.3)]' },
  awaiting_approval: { bg: 'bg-amber-900/40', border: 'border-amber-500/60', text: 'text-amber-100', iconColor: 'text-amber-400', glow: 'shadow-[0_0_20px_rgba(245,158,11,0.4)]' },
  approved:          { bg: 'bg-emerald-900/20', border: 'border-emerald-600/50', text: 'text-emerald-300', iconColor: 'text-emerald-400', glow: '' },
  complete:          { bg: 'bg-emerald-900/20', border: 'border-emerald-600/50', text: 'text-emerald-300', iconColor: 'text-emerald-400', glow: '' },
  rejected:          { bg: 'bg-red-900/30', border: 'border-red-500/50', text: 'text-red-100', iconColor: 'text-red-400', glow: 'shadow-[0_0_15px_rgba(239,68,68,0.4)]' },
  looped_back:       { bg: 'bg-orange-900/30', border: 'border-orange-500/50', text: 'text-orange-100', iconColor: 'text-orange-400', glow: 'shadow-[0_0_15px_rgba(249,115,22,0.4)]' },
  failed:            { bg: 'bg-red-900/30', border: 'border-red-500/50', text: 'text-red-100', iconColor: 'text-red-400', glow: '' },
  skipped:           { bg: 'bg-surface-800/50', border: 'border-surface-700', text: 'text-slate-500', iconColor: 'text-slate-600', glow: '' },
}

const STAGE_CONFIG: Record<string, { label: string, icon: React.ReactNode }> = {
  objective_intake:   { label: 'Objective', icon: <PlayCircle size={18} /> },
  compliance:         { label: 'Compliance', icon: <ShieldCheck size={18} /> },
  data_profiling:     { label: 'Data Profiling', icon: <Search size={18} /> },
  feature_engineering:{ label: 'Engineering', icon: <Settings size={18} /> },
  model_selection:    { label: 'Modeling', icon: <Cpu size={18} /> },
  governance:         { label: 'Governance', icon: <Shield size={18} /> },
  explainability:     { label: 'Explainability', icon: <Lightbulb size={18} /> },
  reporting:          { label: 'Reporting', icon: <FileText size={18} /> },
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

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'running': return <Loader2 size={12} className="animate-spin text-blue-400" />
    case 'awaiting_approval': return <Clock size={12} className="text-amber-400" />
    case 'complete':
    case 'approved': return <CheckCircle2 size={12} className="text-emerald-400" />
    case 'rejected':
    case 'failed': return <AlertCircle size={12} className="text-red-400" />
    case 'looped_back': return <FastForward size={12} className="text-orange-400" />
    default: return <Clock size={12} className="text-slate-500" />
  }
}

function StatusLabel({ status }: { status: string }) {
  const labels: Record<string, string> = {
    pending: 'Pending',
    running: 'Running...',
    awaiting_approval: 'Awaiting Decision',
    approved: 'Approved',
    complete: 'Complete',
    rejected: 'Rejected',
    looped_back: 'Loop-back',
    failed: 'Failed',
    skipped: 'Skipped'
  }
  return <span>{labels[status] || status}</span>
}

function CustomNode({ data }: { data: any }) {
  const { stage, status, isActive, iterations } = data
  const theme = STATUS_THEMES[status] || STATUS_THEMES.pending
  const config = STAGE_CONFIG[stage] || { label: stage, icon: <PlayCircle size={18} /> }

  return (
    <div className={clsx(
      "relative min-w-[200px] rounded-xl border p-4 backdrop-blur-md transition-all duration-300",
      theme.bg, theme.border, theme.glow,
      isActive && "scale-105",
      !isActive && "hover:scale-105 hover:bg-surface-700/80 cursor-default"
    )}>
      {/* Handles for ReactFlow edges */}
      <Handle type="target" position={Position.Left} className="w-2 h-2 !bg-surface-500 border-none" />
      <Handle type="source" position={Position.Right} className="w-2 h-2 !bg-surface-500 border-none" />
      
      {/* Invisible Top/Bottom handles for custom edge routing (like governance loopbacks) */}
      <Handle type="source" position={Position.Top} id="top" className="opacity-0 pointer-events-none" />
      <Handle type="target" position={Position.Top} id="top-target" className="opacity-0 pointer-events-none" />
      <Handle type="source" position={Position.Bottom} id="bottom" className="opacity-0 pointer-events-none" />
      <Handle type="target" position={Position.Bottom} id="bottom-target" className="opacity-0 pointer-events-none" />

      <div className="flex items-center gap-3 mb-3">
        <div className={clsx("p-2 rounded-lg bg-surface-900/50", theme.iconColor)}>
          {config.icon}
        </div>
        <div>
          <h3 className={clsx("font-semibold text-sm tracking-wide", theme.text)}>
            {config.label}
          </h3>
          <div className="flex items-center gap-1.5 mt-0.5 text-[11px] font-medium uppercase tracking-wider text-slate-400">
            <StatusIcon status={status} />
            <StatusLabel status={status} />
          </div>
        </div>
      </div>
      
      {/* Pulse effect for awaiting approval */}
      {status === 'awaiting_approval' && (
        <div className="absolute inset-0 rounded-xl ring-2 ring-amber-500/50 animate-ping opacity-20 pointer-events-none" />
      )}

      {/* Governance Loop Badge */}
      {stage === 'governance' && iterations > 0 && (
        <div className="absolute -top-3 -right-3 bg-orange-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-lg border border-orange-400">
          Loop #{iterations}
        </div>
      )}
    </div>
  )
}

const nodeTypes = {
  custom: CustomNode,
}

// ─────────────────────────────────────────────────────────────────────────────
// Graph Component
// ─────────────────────────────────────────────────────────────────────────────

export default function PipelineDAG({ stageStatuses, currentStage, isPaused, governanceIterations }: Props) {
  
  const nodes: Node[] = useMemo(() => {
    return STAGE_ORDER.map((stage, i) => {
      const statuses = stageStatuses || {}
      const status = (statuses as any)[stage] || 'pending'
      
      // Wrap into a 2-row grid (4 nodes per row)
      const row = Math.floor(i / 4)
      const col = i % 4

      return {
        id: stage,
        type: 'custom',
        position: { x: col * 280, y: row * 220 + 100 },
        data: {
          stage,
          status,
          isActive: stage === currentStage,
          iterations: governanceIterations
        },
      }
    })
  }, [stageStatuses, currentStage, governanceIterations])

  const edges: Edge[] = useMemo(() => {
    const baseEdges: Edge[] = STAGE_ORDER.slice(0, -1).map((stage, i) => {
      const nextStage = STAGE_ORDER[i + 1]
      const statuses = stageStatuses || {}
      const isActive = nextStage === currentStage || stage === currentStage
      const isComplete = (statuses as any)[stage] === 'complete' || (statuses as any)[stage] === 'approved'

      return {
        id: `e-${stage}-${nextStage}`,
        source: stage,
        target: nextStage,
        type: 'smoothstep',
        markerEnd: { 
          type: MarkerType.ArrowClosed, 
          color: isActive ? '#3b82f6' : isComplete ? '#10b981' : '#475569' 
        },
        style: { 
          stroke: isActive ? '#3b82f6' : isComplete ? '#10b981' : '#475569', 
          strokeWidth: isActive ? 3 : 2 
        },
        animated: isActive,
      }
    })

    // Governance loopback edge (routes back visually above the pipeline)
    if (governanceIterations > 0) {
      baseEdges.push({
        id: 'e-gov-loopback-fe',
        source: 'governance',
        target: 'feature_engineering',
        type: 'smoothstep',
        sourceHandle: 'top',
        targetHandle: 'bottom-target',
        markerEnd: { type: MarkerType.ArrowClosed, color: '#f97316' },
        style: { stroke: '#f97316', strokeWidth: 2, strokeDasharray: '5,5' },
        label: 'FAIL → retry',
        labelStyle: { fill: '#fb923c', fontSize: 10, fontWeight: 'bold' },
        labelBgStyle: { fill: '#1e293b', stroke: '#334155', strokeWidth: 1 },
        labelBgPadding: [4, 2],
        labelBgBorderRadius: 4,
        animated: true
      })
    }

    return baseEdges
  }, [currentStage, stageStatuses, governanceIterations])

  return (
    <div className="w-full h-[650px] rounded-2xl overflow-hidden border border-surface-700 bg-surface-900 relative shadow-inner">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        panOnDrag={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#334155" gap={24} size={2} className="opacity-40" />
      </ReactFlow>
    </div>
  )
}
