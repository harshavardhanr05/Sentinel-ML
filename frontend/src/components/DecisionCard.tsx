/**
 * DecisionCard.tsx
 * Checkpoint UI — shows proposed action, reasoning, alternatives, cost estimate.
 * Approve / Reject / Counter-Propose buttons wired to the decision API.
 */

import React, { useState, useEffect } from 'react'
import { CheckCircle, XCircle, MessageSquare, ChevronDown, ChevronUp, Zap, Bot, Activity } from 'lucide-react'
import { submitDecision } from '../api/client'
import type { DecisionCard as IDecisionCard } from '../api/client'
import { ActivityStep, type AgentStep } from './AuditTrailViewer'
import clsx from 'clsx'

interface Props {
  runId: string
  card: IDecisionCard
  realtimeLogs?: any[]
  onDecisionMade: (justification?: string) => void
}

export default function DecisionCard({ runId, card, realtimeLogs, onDecisionMade }: Props) {
  const [loading, setLoading] = useState<string | null>(null)
  const [showAlternatives, setShowAlternatives] = useState(false)
  const [showCounterForm, setShowCounterForm] = useState(false)
  const [counterNote, setCounterNote] = useState('')
  const [agentResponse, setAgentResponse] = useState<string | null>(null)
  const [aiExecutionLogs, setAiExecutionLogs] = useState<any[] | null>(card.ai_execution_logs || null)
  const [logStartIndex, setLogStartIndex] = useState<number>(-1)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (card.ai_execution_logs) {
      setAiExecutionLogs(card.ai_execution_logs)
    }
  }, [card.ai_execution_logs])

  const handleDecision = async (action: 'approve' | 'reject' | 'counter_propose', note?: string) => {
    setLoading(action)
    setError(null)
    if (action === 'counter_propose' && realtimeLogs) {
      setLogStartIndex(realtimeLogs.length)
      setAiExecutionLogs(null)
    }
    try {
      const result = await submitDecision(runId, action, note)
      if (result.ai_execution_logs) {
        setAiExecutionLogs(result.ai_execution_logs)
      }
      if (action === 'counter_propose') {
        setCounterNote('')
      }
      onDecisionMade(result.agent_justification)
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to submit decision')
    } finally {
      setLoading(null)
    }
  }

  const displayLogs = loading === 'counter_propose' && realtimeLogs && logStartIndex >= 0
    ? realtimeLogs.slice(logStartIndex).filter(l => l.is_ai_code_request)
    : aiExecutionLogs;

  const stageName = card.stage.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

  return (
    <div className="card border-amber-800/50 animate-slide-up relative overflow-hidden">
      {/* Amber accent top bar */}
      <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-amber-500 via-amber-400 to-amber-500" />

      {/* Header */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-10 h-10 rounded-xl bg-amber-900/50 border border-amber-700/50 flex items-center justify-center">
          <Zap size={18} className="text-amber-400" />
        </div>
        <div>
          <div className="text-xs font-semibold text-amber-400 uppercase tracking-wide mb-0.5">
            Awaiting Your Decision
          </div>
          <h2 className="text-lg font-semibold text-slate-100">{stageName}</h2>
        </div>
      </div>

      {/* Proposed action */}
      <div className="bg-surface-700/50 rounded-xl p-4 mb-4 border border-surface-600/50">
        <div className="text-xs text-slate-400 font-medium uppercase tracking-wide mb-2">Proposed Action</div>
        <p className="text-slate-100 font-medium leading-relaxed">{card.proposed_action}</p>
      </div>

      {/* Reasoning */}
      <div className="mb-4">
        <div className="text-xs text-slate-400 font-medium uppercase tracking-wide mb-2">Agent Reasoning</div>
        <p className="text-slate-300 text-sm leading-relaxed">{card.reasoning}</p>
      </div>

      {/* Cost estimate */}
      {card.cost_estimate && (
        <div className="bg-surface-700/30 rounded-xl p-3 mb-4 flex items-center gap-2 border border-surface-600/30">
          <span className="text-xs text-slate-400">⏱️ Estimated compute time:</span>
          <span className="text-sm font-semibold text-brand-300">{card.cost_estimate}</span>
        </div>
      )}

      {/* Metrics summary */}
      {Object.keys(card.metrics_summary).length > 0 && (
        <div className="mb-4">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {Object.entries(card.metrics_summary)
              .filter(([key]) => key !== 'smote_class_distributions')
              .slice(0, 6).map(([key, val]) => (
              <div key={key} className="bg-surface-700/50 rounded-lg p-2.5 border border-surface-600/30">
                <div className="text-xs text-slate-500 mb-1">{key.replace(/_/g, ' ')}</div>
                <div className="text-sm font-semibold text-slate-200">
                  {typeof val === 'number' 
                    ? val.toFixed(3) 
                    : typeof val === 'object' && val !== null 
                      ? Object.entries(val).map(([k, v]) => `${k}: ${v}`).join(', ')
                      : String(val ?? 'N/A')}
                </div>
              </div>
            ))}
          </div>

          {/* SMOTE Distributions Visualization */}
          {Boolean(card.metrics_summary.smote_class_distributions) && typeof card.metrics_summary.smote_class_distributions === 'object' && (
            <div className="bg-surface-800/80 rounded-xl p-4 border border-surface-600/50 mt-4 shadow-inner">
              <h4 className="text-xs font-semibold uppercase text-brand-300 mb-3 flex items-center gap-2">
                Class Balance (SMOTE)
              </h4>
              <div className="flex flex-col md:flex-row gap-4">
                {['before', 'after'].map((stage) => {
                  const dist = (card.metrics_summary.smote_class_distributions as any)[stage]
                  if (!dist) return null
                  const maxCount = Math.max(...(Object.values(dist) as number[]))
                  return (
                    <div key={stage} className="flex-1 bg-surface-900/50 p-3 rounded-lg border border-surface-700/50">
                      <div className="text-xs font-medium text-slate-400 capitalize mb-2">{stage} SMOTE</div>
                      <div className="space-y-2">
                        {Object.entries(dist).map(([cls, count]) => (
                          <div key={cls} className="flex items-center gap-2">
                            <span className="text-xs font-mono w-16 truncate text-slate-300" title={cls}>{cls}</span>
                            <div className="flex-1 h-2 bg-surface-700 rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${stage === 'before' ? 'bg-amber-500' : 'bg-emerald-500'}`} 
                                style={{ width: `${Math.max(((count as number) / maxCount) * 100, 2)}%` }}
                              />
                            </div>
                            <span className="text-xs font-mono text-slate-400 w-10 text-right">{count as number}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Alternatives */}
      {card.alternatives_considered.length > 0 && (
        <div className="mb-4">
          <button
            onClick={() => setShowAlternatives(!showAlternatives)}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
          >
            {showAlternatives ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            {card.alternatives_considered.length} alternative(s) considered
          </button>
          {showAlternatives && (
            <ul className="mt-2 space-y-1.5">
              {card.alternatives_considered.map((alt, i) => (
                <li key={i} className="text-sm text-slate-400 pl-3 border-l-2 border-surface-600">
                  <button
                    onClick={() => {
                      setCounterNote(alt)
                      setShowCounterForm(true)
                      setTimeout(() => document.getElementById('counter-propose-input')?.focus(), 50)
                    }}
                    className="text-left hover:text-brand-300 transition-colors flex items-center gap-2 group w-full"
                  >
                    <span>{alt}</span>
                    <span className="text-[10px] text-brand-500/0 group-hover:text-brand-500/80 uppercase font-bold tracking-wider">Use this</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* AI Execution Logs (shown after counter-propose if any ai logs are present) */}
      {displayLogs && displayLogs.length > 0 && (
        <div className="mb-4 bg-brand-900/20 border border-brand-800/30 rounded-xl p-4">
          <div className="space-y-3">
            <span className="text-xs font-bold uppercase tracking-wider text-brand-300 flex items-center gap-2 mb-2">
              <Activity size={14} /> AI Code Execution History
            </span>
            <div className="pl-2">
              {displayLogs.map((log, i) => (
                <ActivityStep 
                  key={log.entry_id || i} 
                  step={log as unknown as AgentStep} 
                  isLast={i === displayLogs.length - 1} 
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Counter-propose form */}
      {showCounterForm && (
        <div className="mb-4 space-y-3 animate-slide-up">
          <div className="bg-surface-800/50 rounded-lg p-3 border border-surface-700/50">
            <div className="text-xs font-semibold text-slate-300 mb-2 flex items-center gap-1.5">
              <Zap size={12} className="text-amber-400" />
              Instant Native Features Catalog
            </div>
            <div className="grid grid-cols-1 gap-2 text-[11px] text-slate-400">
              <div className="flex gap-2">
                <span className="text-brand-300 font-mono w-24 shrink-0">smote, smotetomek, smoteenn</span>
                <span>Advanced class imbalance handling (oversampling/hybrid).</span>
              </div>
              <div className="flex gap-2">
                <span className="text-brand-300 font-mono w-24 shrink-0">quantile, power, robust scaler</span>
                <span>Advanced numerical scaling and variance stabilization.</span>
              </div>
              <div className="flex gap-2">
                <span className="text-brand-300 font-mono w-24 shrink-0">pca, polynomial</span>
                <span>Dimensionality reduction and feature interactions.</span>
              </div>
              <div className="flex gap-2">
                <span className="text-brand-300 font-mono w-24 shrink-0">isolation forest</span>
                <span>Automatic robust outlier removal on training data.</span>
              </div>
            </div>
          </div>
          <textarea
            value={counterNote}
            onChange={e => setCounterNote(e.target.value)}
            placeholder="Describe your alternative approach (e.g. 'Use Age as the target column')..."
            rows={3}
            className="input-field resize-none text-sm"
            id="counter-propose-input"
          />
          <div className="flex gap-2">
            <button
              id="submit-counter-propose"
              disabled={!counterNote.trim() || loading === 'counter_propose'}
              onClick={() => handleDecision('counter_propose', counterNote)}
              className="btn-primary flex-1 justify-center text-sm"
            >
              {loading === 'counter_propose' ? 'Getting agent response...' : 'Submit Alternative'}
            </button>
            <button
              onClick={() => { setShowCounterForm(false); setCounterNote('') }}
              className="btn-secondary text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mb-4 bg-red-900/30 border border-red-800/50 rounded-lg p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-3 pt-2 border-t border-surface-700">
        <button
          id="approve-btn"
          onClick={() => handleDecision('approve')}
          disabled={!!loading}
          className="btn-primary flex-1 justify-center"
        >
          <CheckCircle size={16} />
          {loading === 'approve' ? 'Approving...' : 'Approve'}
        </button>
        <button
          id="reject-btn"
          onClick={() => handleDecision('reject')}
          disabled={!!loading}
          className="btn-danger flex-1 justify-center"
        >
          <XCircle size={16} />
          {loading === 'reject' ? 'Rejecting...' : 
            card.stage === 'model_selection' ? 'Retrain Models' :
            card.stage === 'feature_engineering' ? 'Retry Engineering' :
            card.stage === 'data_profiling' ? 'Retry Profiling' :
            'Reject'}
        </button>
        <button
          id="counter-propose-btn"
          onClick={() => setShowCounterForm(!showCounterForm)}
          disabled={!!loading}
          className="btn-secondary flex-1 justify-center"
        >
          <MessageSquare size={16} />
          Suggest Alt.
        </button>
      </div>
    </div>
  )
}
