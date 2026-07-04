/**
 * AuditTrailViewer.tsx
 * Chronological timeline of every decisions_log entry.
 */

import React from 'react'
import { Clock, Bot, User, CheckCircle, XCircle, MessageSquare } from 'lucide-react'
import type { DecisionLogEntry } from '../api/client'
import clsx from 'clsx'

interface Props {
  decisions: DecisionLogEntry[]
  runId: string
}

const ACTION_ICON: Record<string, React.ReactNode> = {
  approve:         <CheckCircle size={14} className="text-emerald-400" />,
  reject:          <XCircle size={14} className="text-red-400" />,
  counter_propose: <MessageSquare size={14} className="text-orange-400" />,
  pending:         <Clock size={14} className="text-amber-400" />,
}

const ACTION_COLORS: Record<string, string> = {
  approve:         'bg-emerald-900/30 border-emerald-800/40',
  reject:          'bg-red-900/30 border-red-800/40',
  counter_propose: 'bg-orange-900/30 border-orange-800/40',
  pending:         'bg-amber-900/20 border-amber-800/30',
}

export default function AuditTrailViewer({ decisions, runId }: Props) {
  if (decisions.length === 0) {
    return (
      <div className="card">
        <h3 className="section-title flex items-center gap-2">
          <Clock size={16} className="text-brand-400" />
          Audit Trail
        </h3>
        <div className="text-center text-slate-500 text-sm py-12">
          <Clock size={32} className="mx-auto mb-3 opacity-30" />
          <p>No decisions logged yet.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-6">
        <Clock size={16} className="text-brand-400" />
        <h3 className="section-title mb-0">Audit Trail</h3>
        <div className="ml-auto flex gap-2 items-center">
          <span className="text-xs text-slate-500">Run ID:</span>
          <code className="text-xs bg-surface-700 px-2 py-0.5 rounded text-slate-300 font-mono">
            {runId.slice(0, 8)}...
          </code>
        </div>
      </div>

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-5 top-0 bottom-0 w-px bg-surface-700" />

        <div className="space-y-4">
          {decisions.map((entry, i) => (
            <div key={entry.entry_id || i} className="relative pl-12 animate-fade-in">
              {/* Timeline dot */}
              <div className={clsx(
                'absolute left-3.5 top-3 w-3 h-3 rounded-full border-2 border-surface-900',
                entry.user_action === 'approve' ? 'bg-emerald-400' :
                entry.user_action === 'reject' ? 'bg-red-400' :
                entry.user_action === 'counter_propose' ? 'bg-orange-400' :
                'bg-slate-500'
              )} />

              <div className={clsx(
                'rounded-xl border p-4',
                ACTION_COLORS[entry.user_action] || 'bg-surface-700/30 border-surface-600/30'
              )}>
                {/* Header */}
                <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide bg-surface-700 px-2 py-0.5 rounded">
                      {entry.stage.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                      {ACTION_ICON[entry.user_action]}
                      <span className={clsx(
                        'text-xs font-semibold uppercase tracking-wide',
                        entry.user_action === 'approve' ? 'text-emerald-400' :
                        entry.user_action === 'reject' ? 'text-red-400' :
                        entry.user_action === 'counter_propose' ? 'text-orange-400' :
                        'text-amber-400'
                      )}>
                        {entry.user_action?.replace(/_/g, ' ') || 'pending'}
                      </span>
                    </div>
                    <span className="text-xs text-slate-500 flex items-center gap-1">
                      <Clock size={10} />
                      {new Date(entry.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>

                {/* Agent proposal */}
                <div className="mb-2">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Bot size={12} className="text-brand-400" />
                    <span className="text-xs text-slate-500">Agent proposed:</span>
                  </div>
                  <p className="text-sm text-slate-200 font-medium leading-relaxed">{entry.proposed_action}</p>
                </div>

                {/* Reasoning (collapsed by default) */}
                {entry.reasoning && entry.reasoning !== entry.proposed_action && (
                  <p className="text-xs text-slate-400 leading-relaxed mb-2 border-l-2 border-surface-600 pl-3">
                    {entry.reasoning}
                  </p>
                )}

                {/* User note */}
                {entry.user_note && (
                  <div className="mt-2 bg-surface-800/50 rounded-lg p-2.5 border border-surface-700/50">
                    <div className="flex items-center gap-1.5 mb-1">
                      <User size={11} className="text-slate-400" />
                      <span className="text-xs text-slate-500">User note:</span>
                    </div>
                    <p className="text-xs text-slate-300 italic">"{entry.user_note}"</p>
                  </div>
                )}

                {/* Agent justification (for counter-proposals) */}
                {entry.agent_justification && (
                  <div className="mt-2 bg-brand-900/20 rounded-lg p-2.5 border border-brand-800/30">
                    <div className="flex items-center gap-1.5 mb-1">
                      <Bot size={11} className="text-brand-400" />
                      <span className="text-xs text-brand-400">Agent response:</span>
                    </div>
                    <p className="text-xs text-brand-200 leading-relaxed">{entry.agent_justification}</p>
                  </div>
                )}

                {/* Decision time */}
                {entry.decided_at && (
                  <div className="mt-2 text-xs text-slate-600">
                    Decided at: {new Date(entry.decided_at).toLocaleString()}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Export links */}
      <div className="mt-6 pt-4 border-t border-surface-700 flex gap-3">
        <a
          href={`/api/runs/${runId}/audit-trail`}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-secondary text-sm flex-1 justify-center"
        >
          Export as HTML
        </a>
        <a
          href={`/api/runs/${runId}/model-card`}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-primary text-sm flex-1 justify-center"
        >
          View Model Card
        </a>
      </div>
    </div>
  )
}
