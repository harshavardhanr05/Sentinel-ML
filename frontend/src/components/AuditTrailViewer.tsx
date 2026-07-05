/**
 * AuditTrailViewer.tsx
 * Two-tab audit panel:
 *  Tab 1: "Decision Checkpoints" — existing decisions_log entries
 *  Tab 2: "Agent Activity"       — verbose agent_step_log timeline
 */

import React, { useState, useMemo } from 'react'
import {
  Clock, Bot, User, CheckCircle, XCircle, MessageSquare, AlertTriangle,
  Zap, Activity, ChevronDown, ChevronRight, Filter, Database, ShieldX, Link, 
  FlaskConical, Layers,
} from 'lucide-react'
import type { DecisionLogEntry } from '../api/client'
import clsx from 'clsx'

interface AgentStep {
  entry_id: string
  stage: string
  step_name: string
  details: string
  timestamp: string
}

interface Props {
  decisions: DecisionLogEntry[]
  runId: string
  featureLog?: {
    accepted: Array<{
      feature: string
      transformation?: string | null
      status: string
      reason: string
      metric_delta?: number | null
      governance_flagged?: boolean
      imputation_strategy?: string | null
    }>
    rejected: Array<{
      feature: string
      transformation?: string | null
      status: string
      reason: string
      governance_flagged?: boolean
    }>
    final_feature_set: string[]
  }
  agentStepLog?: AgentStep[]
}

// ── Checkpoint tab config ────────────────────────────────────────────

const ACTION_ICON: Record<string, React.ReactNode> = {
  approve:         <CheckCircle size={14} className="text-emerald-400" />,
  reject:          <XCircle size={14} className="text-red-400" />,
  counter_propose: <MessageSquare size={14} className="text-orange-400" />,
  pending:         <Clock size={14} className="text-amber-400" />,
}

const ACTION_COLORS: Record<string, string> = {
  approve:         'from-emerald-900/30 to-emerald-900/10 border-emerald-800/50',
  reject:          'from-red-900/30 to-red-900/10 border-red-800/50',
  counter_propose: 'from-orange-900/30 to-orange-900/10 border-orange-800/50',
  pending:         'from-amber-900/20 to-amber-900/5 border-amber-800/40',
}

const STAGE_COLORS: Record<string, string> = {
  objective_intake:     '#6366f1',
  data_profiling:       '#0ea5e9',
  feature_engineering:  '#f59e0b',
  model_selection:      '#10b981',
  governance:           '#ec4899',
  explainability:       '#8b5cf6',
  reporting:            '#84cc16',
}

function stageDot(stage: string) {
  return <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: STAGE_COLORS[stage] || '#64748b' }} />
}

// ── Step icon by step_name keywords ─────────────────────────────────

function stepIcon(stepName: string, stage: string) {
  const lower = stepName.toLowerCase()
  if (lower.includes('fail') || lower.includes('error')) return <XCircle size={13} className="text-red-400" />
  if (lower.includes('complete') || lower.includes('selected') || lower.includes('trained')) return <CheckCircle size={13} className="text-emerald-400" />
  if (lower.includes('smote') || lower.includes('resam')) return <Zap size={13} className="text-emerald-400" />
  if (lower.includes('loop') || lower.includes('governance')) return <Activity size={13} className="text-pink-400" />
  if (lower.includes('ai') || lower.includes('llm') || lower.includes('chart')) return <Bot size={13} className="text-brand-400" />
  return <div className="w-2 h-2 rounded-full" style={{ background: STAGE_COLORS[stage] || '#64748b' }} />
}

// ── Decision Checkpoint Card ─────────────────────────────────────────

function CheckpointCard({ entry }: { entry: DecisionLogEntry }) {
  const [expanded, setExpanded] = useState(false)
  const action = entry.user_action || 'pending'
  const colors = ACTION_COLORS[action] || ACTION_COLORS.pending

  return (
    <div className={clsx('rounded-xl border shadow-lg bg-gradient-to-br p-4 transition-all duration-200', colors)}>
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2.5 min-w-0">
          {stageDot(entry.stage)}
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider truncate">{entry.stage.replace(/_/g, ' ')}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {ACTION_ICON[action]}
          <span className="text-xs text-slate-400 font-medium capitalize">{action.replace(/_/g, ' ')}</span>
          <span className="text-xs text-slate-600">·</span>
          <span className="text-xs text-slate-500">{new Date(entry.timestamp).toLocaleTimeString()}</span>
        </div>
      </div>

      {/* Problem context */}
      {entry.problem_context && (
        <div className="mb-2">
          <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">Problem</span>
          <p className="text-sm text-slate-300 mt-0.5 leading-relaxed">{entry.problem_context}</p>
        </div>
      )}

      {/* Action taken */}
      {entry.action_taken && (
        <div className="mb-2">
          <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">Action Taken</span>
          <p className="text-sm text-slate-200 font-medium mt-0.5">{entry.action_taken}</p>
        </div>
      )}

      {/* Proposed */}
      <div className="bg-surface-700/40 rounded-lg p-2.5 mb-3 border border-surface-600/30">
        <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">Proposed Action</span>
        <p className="text-sm text-slate-200 font-medium mt-1 leading-relaxed">{entry.proposed_action}</p>
      </div>

      {/* Toggle extras */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors"
      >
        {expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        {expanded ? 'Hide details' : 'Show reasoning & alternatives'}
      </button>

      {expanded && (
        <div className="mt-3 space-y-3 border-t border-surface-600/30 pt-3">
          {/* Reasoning */}
          <div>
            <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">Agent Reasoning</span>
            <p className="text-sm text-slate-400 mt-1 leading-relaxed">{entry.reasoning}</p>
          </div>

          {/* Alternatives */}
          {entry.alternatives_considered?.length > 0 && (
            <div>
              <span className="text-[10px] font-bold uppercase text-slate-500 tracking-wider">Alternatives Considered</span>
              <ul className="mt-1.5 space-y-1">
                {entry.alternatives_considered.map((alt, i) => (
                  <li key={i} className="text-sm text-slate-500 pl-3 border-l-2 border-surface-600">{alt}</li>
                ))}
              </ul>
            </div>
          )}

          {/* User note */}
          {entry.user_note && (
            <div className="bg-orange-900/20 border border-orange-800/30 rounded-lg p-2.5">
              <div className="flex items-center gap-2 mb-1">
                <User size={12} className="text-orange-400" />
                <span className="text-[10px] font-bold uppercase text-orange-400">User Alternative Suggestion</span>
              </div>
              <p className="text-sm text-orange-200">{entry.user_note}</p>
            </div>
          )}

          {/* Agent justification of user's suggestion */}
          {entry.agent_justification && (
            <div className="bg-brand-900/20 border border-brand-800/30 rounded-lg p-2.5">
              <div className="flex items-center gap-2 mb-1">
                <Bot size={12} className="text-brand-400" />
                <span className="text-[10px] font-bold uppercase text-brand-400">Agent Response to Suggestion</span>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">
                {typeof entry.agent_justification === 'string'
                  ? entry.agent_justification
                  : JSON.stringify(entry.agent_justification, null, 2)}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Agent Activity Step ──────────────────────────────────────────────

function ActivityStep({ step, isLast }: { step: AgentStep; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="flex gap-3">
      {/* Timeline line */}
      <div className="flex flex-col items-center">
        <div className="w-7 h-7 rounded-full bg-surface-700/80 border border-surface-600/50 flex items-center justify-center flex-shrink-0 shadow-md">
          {stepIcon(step.step_name, step.stage)}
        </div>
        {!isLast && <div className="w-px flex-1 bg-surface-600/40 mt-1" />}
      </div>
      {/* Content */}
      <div className={clsx('pb-5 flex-1 min-w-0', isLast && 'pb-0')}>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded" style={{ background: (STAGE_COLORS[step.stage] || '#64748b') + '25', color: STAGE_COLORS[step.stage] || '#94a3b8' }}>
            {step.stage.replace(/_/g, ' ')}
          </span>
          <span className="text-sm font-semibold text-slate-200">{step.step_name}</span>
          <span className="text-xs text-slate-600 ml-auto">{new Date(step.timestamp).toLocaleTimeString()}</span>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-1 text-xs text-slate-500 hover:text-slate-300 transition-colors text-left flex items-center gap-1"
        >
          {expanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
          {expanded ? 'Hide details' : step.details.slice(0, 80) + (step.details.length > 80 ? '...' : '')}
        </button>
        {expanded && (
          <div className="mt-2 bg-surface-800/60 rounded-lg p-3 border border-surface-600/30 text-sm text-slate-400 leading-relaxed">
            {step.details}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main Component ───────────────────────────────────────────────────

export default function AuditTrailViewer({ decisions, runId, featureLog, agentStepLog = [] }: Props) {
  const [activeTab, setActiveTab] = useState<'checkpoints' | 'features' | 'activity'>('checkpoints')
  const [stageFilter, setStageFilter] = useState<string>('all')
  const [featFilter, setFeatFilter] = useState<'all' | 'accepted' | 'rejected'>('all')

  const accepted = featureLog?.accepted || []
  const rejected = featureLog?.rejected || []
  const allFeatures = [...accepted, ...rejected]
  const displayFeatures = featFilter === 'all' ? allFeatures
    : featFilter === 'accepted' ? accepted
    : rejected

  const stages = useMemo(() => {
    const s = new Set(agentStepLog.map(s => s.stage))
    return ['all', ...Array.from(s)]
  }, [agentStepLog])

  const filteredSteps = useMemo(() =>
    stageFilter === 'all' ? agentStepLog : agentStepLog.filter(s => s.stage === stageFilter),
    [agentStepLog, stageFilter]
  )

  // Rejection category classifier
  function getRejectionCategory(reason: string, governance_flagged?: boolean): { label: string; color: string; Icon: any } {
    if (governance_flagged) return { label: 'Fairness Proxy', color: 'text-pink-400 bg-pink-900/30 border-pink-800/40', Icon: ShieldX }
    if (reason.includes('leakage')) return { label: 'Data Leakage', color: 'text-red-400 bg-red-900/30 border-red-800/40', Icon: Link }
    if (reason.includes('multicollinearity') || reason.includes('corr(')) return { label: 'Multicollinear', color: 'text-orange-400 bg-orange-900/30 border-orange-800/40', Icon: Layers }
    if (reason.includes('missing') || reason.includes('missingness')) return { label: 'High Missingness', color: 'text-amber-400 bg-amber-900/30 border-amber-800/40', Icon: Database }
    if (reason.includes('Zero variance') || reason.includes('constant')) return { label: 'Zero Variance', color: 'text-slate-400 bg-slate-900/30 border-slate-700/40', Icon: FlaskConical }
    if (reason.includes('AI Semantic') || reason.includes('—')) return { label: 'Semantically Irrelevant', color: 'text-violet-400 bg-violet-900/30 border-violet-800/40', Icon: Bot }
    return { label: 'Dropped', color: 'text-slate-400 bg-slate-800/30 border-slate-700/40', Icon: XCircle }
  }

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex gap-1 bg-surface-800/60 p-1 rounded-xl border border-surface-600/30 w-fit flex-wrap">
        <button
          onClick={() => setActiveTab('checkpoints')}
          className={clsx(
            'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2',
            activeTab === 'checkpoints'
              ? 'bg-brand-600 text-white shadow-lg shadow-brand-900/50'
              : 'text-slate-400 hover:text-slate-200'
          )}
        >
          <Clock size={14} />
          Decision Checkpoints
          {decisions.length > 0 && (
            <span className="bg-surface-600/60 text-slate-300 text-xs px-1.5 py-0.5 rounded-full">{decisions.length}</span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('features')}
          className={clsx(
            'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2',
            activeTab === 'features'
              ? 'bg-amber-600 text-white shadow-lg shadow-amber-900/50'
              : 'text-slate-400 hover:text-slate-200'
          )}
        >
          <Database size={14} />
          Feature Selection
          {allFeatures.length > 0 && (
            <span className="bg-surface-600/60 text-slate-300 text-xs px-1.5 py-0.5 rounded-full">{accepted.length}✓ {rejected.length}✗</span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('activity')}
          className={clsx(
            'px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2',
            activeTab === 'activity'
              ? 'bg-brand-600 text-white shadow-lg shadow-brand-900/50'
              : 'text-slate-400 hover:text-slate-200'
          )}
        >
          <Activity size={14} />
          Agent Activity
          {agentStepLog.length > 0 && (
            <span className="bg-surface-600/60 text-slate-300 text-xs px-1.5 py-0.5 rounded-full">{agentStepLog.length}</span>
          )}
        </button>
      </div>

      {/* Decision Checkpoints Tab */}
      {activeTab === 'checkpoints' && (
        decisions.length === 0 ? (
          <div className="text-center text-slate-500 text-sm py-16">
            <Clock size={32} className="mx-auto mb-3 opacity-30" />
            <p>No decision checkpoints logged yet.</p>
            <p className="text-xs mt-1 text-slate-600">Checkpoints appear when the pipeline pauses for human review.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {decisions.map((entry, i) => (
              <CheckpointCard key={entry.entry_id || i} entry={entry} />
            ))}
          </div>
        )
      )}

      {/* Feature Selection Tab */}
      {activeTab === 'features' && (
        <div className="space-y-4">
          {/* Summary bar */}
          <div className="grid grid-cols-3 gap-3">
            <div className="card-sm border-emerald-800/40 bg-emerald-950/20 flex items-center gap-3">
              <CheckCircle size={20} className="text-emerald-400 flex-shrink-0" />
              <div>
                <div className="text-2xl font-bold text-emerald-400">{accepted.length}</div>
                <div className="text-xs text-slate-400">Features Accepted</div>
              </div>
            </div>
            <div className="card-sm border-red-800/40 bg-red-950/20 flex items-center gap-3">
              <XCircle size={20} className="text-red-400 flex-shrink-0" />
              <div>
                <div className="text-2xl font-bold text-red-400">{rejected.length}</div>
                <div className="text-xs text-slate-400">Features Rejected</div>
              </div>
            </div>
            <div className="card-sm border-surface-600/40 flex items-center gap-3">
              <Database size={20} className="text-brand-400 flex-shrink-0" />
              <div>
                <div className="text-2xl font-bold text-brand-400">{featureLog?.final_feature_set?.length ?? 0}</div>
                <div className="text-xs text-slate-400">Final Feature Set</div>
              </div>
            </div>
          </div>

          {/* Filter pills */}
          <div className="flex items-center gap-2">
            {(['all', 'accepted', 'rejected'] as const).map(f => (
              <button key={f} onClick={() => setFeatFilter(f)}
                className={clsx(
                  'text-xs px-3 py-1.5 rounded-full border font-medium transition-all capitalize',
                  featFilter === f
                    ? f === 'accepted' ? 'bg-emerald-900/40 border-emerald-700 text-emerald-300'
                    : f === 'rejected' ? 'bg-red-900/40 border-red-700 text-red-300'
                    : 'bg-brand-900/30 border-brand-700 text-brand-300'
                    : 'border-surface-600/40 text-slate-500 hover:text-slate-300'
                )}
              >
                {f === 'all' ? `All (${allFeatures.length})` : f === 'accepted' ? `✓ Accepted (${accepted.length})` : `✗ Rejected (${rejected.length})`}
              </button>
            ))}
          </div>

          {/* Feature rows */}
          {allFeatures.length === 0 ? (
            <div className="text-center text-slate-500 text-sm py-16">
              <Database size={32} className="mx-auto mb-3 opacity-30" />
              <p>Feature Engineering has not run yet.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {displayFeatures.map((feat: any) => {
                const isAccepted = feat.status === 'accepted'
                const cat = isAccepted ? null : getRejectionCategory(feat.reason || '', feat.governance_flagged)

                // Parse out description vs reason for dropped features
                // Format: "Description — Rejection reason" (set by backend)
                let description = ''
                let shortReason = feat.reason || ''
                const dashIdx = feat.reason?.indexOf(' — ')
                if (!isAccepted && dashIdx && dashIdx > 0) {
                  description = feat.reason.slice(0, dashIdx)
                  shortReason = feat.reason.slice(dashIdx + 3)
                }

                return (
                  <div key={feat.feature} className={clsx(
                    'rounded-xl border p-4 transition-all duration-150',
                    isAccepted
                      ? 'border-emerald-800/30 bg-emerald-950/10 hover:bg-emerald-950/20'
                      : 'border-red-900/30 bg-red-950/10 hover:bg-red-950/20'
                  )}>
                    <div className="flex items-start gap-3">
                      {/* Status dot */}
                      <div className={clsx(
                        'mt-0.5 flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center',
                        isAccepted ? 'bg-emerald-900/60' : 'bg-red-900/60'
                      )}>
                        {isAccepted
                          ? <CheckCircle size={12} className="text-emerald-400" />
                          : <XCircle size={12} className="text-red-400" />}
                      </div>

                      <div className="flex-1 min-w-0">
                        {/* Column name + category badge */}
                        <div className="flex items-center gap-2 flex-wrap mb-1">
                          <span className={clsx(
                            'font-mono font-semibold text-sm',
                            isAccepted ? 'text-emerald-300' : 'text-red-300'
                          )}>{feat.feature}</span>

                          {/* Rejection category badge */}
                          {!isAccepted && cat && (
                            <span className={clsx(
                              'text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border flex items-center gap-1',
                              cat.color
                            )}>
                              <cat.Icon size={9} />
                              {cat.label}
                            </span>
                          )}

                          {/* Transformation badge for accepted */}
                          {isAccepted && feat.transformation && feat.transformation !== 'keep_as_is' && (
                            <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border border-emerald-800/40 bg-emerald-900/20 text-emerald-400">
                              {feat.transformation?.replace(/_/g, ' ')}
                            </span>
                          )}

                          {/* Governance flag */}
                          {feat.governance_flagged && (
                            <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border border-pink-800/50 bg-pink-900/30 text-pink-400 flex items-center gap-1">
                              <ShieldX size={9} /> Fairness Risk
                            </span>
                          )}
                        </div>

                        {/* Column description (what it means) */}
                        {description && (
                          <p className="text-xs text-slate-400 leading-relaxed mb-1.5 italic">
                            {description}
                          </p>
                        )}

                        {/* Reason */}
                        <p className={clsx(
                          'text-xs leading-relaxed',
                          isAccepted ? 'text-emerald-500/80' : 'text-red-400/80'
                        )}>
                          {isAccepted ? '✓ ' : '✗ '}{shortReason}
                        </p>

                        {/* Metric delta for accepted */}
                        {isAccepted && feat.metric_delta != null && (
                          <div className="mt-1.5">
                            <span className={clsx(
                              'text-[10px] font-mono font-bold px-2 py-0.5 rounded',
                              feat.metric_delta >= 0 ? 'text-emerald-400 bg-emerald-900/30' : 'text-amber-400 bg-amber-900/30'
                            )}>
                              Δ metric: {feat.metric_delta >= 0 ? '+' : ''}{feat.metric_delta.toFixed(4)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Agent Activity Tab */}
      {activeTab === 'activity' && (
        agentStepLog.length === 0 ? (
          <div className="text-center text-slate-500 text-sm py-16">
            <Activity size={32} className="mx-auto mb-3 opacity-30" />
            <p>No agent activity logged yet.</p>
            <p className="text-xs mt-1 text-slate-600">Activity appears as the pipeline runs each stage.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Stage filter */}
            <div className="flex items-center gap-2 flex-wrap">
              <Filter size={13} className="text-slate-500" />
              {stages.map(s => (
                <button
                  key={s}
                  onClick={() => setStageFilter(s)}
                  className={clsx(
                    'text-xs px-2.5 py-1 rounded-lg border transition-all',
                    stageFilter === s
                      ? 'border-brand-600 bg-brand-900/30 text-brand-300'
                      : 'border-surface-600/40 text-slate-500 hover:text-slate-300 hover:border-surface-500'
                  )}
                >
                  {s.replace(/_/g, ' ')}
                </button>
              ))}
            </div>

            {/* Timeline */}
            <div className="bg-surface-800/40 rounded-2xl border border-surface-600/30 p-5">
              {filteredSteps.map((step, i) => (
                <ActivityStep key={step.entry_id} step={step} isLast={i === filteredSteps.length - 1} />
              ))}
            </div>
          </div>
        )
      )}
    </div>
  )
}
