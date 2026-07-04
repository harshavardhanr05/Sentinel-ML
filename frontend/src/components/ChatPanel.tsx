/**
 * ChatPanel.tsx
 * Scrolling log of agent reasoning pulled from decisions_log.
 * Shows who made each decision (agent vs user) with timestamps.
 */

import React, { useEffect, useRef } from 'react'
import { Bot, User, AlertCircle, RefreshCw } from 'lucide-react'
import type { DecisionLogEntry } from '../api/client'
import clsx from 'clsx'

interface Props {
  decisions: DecisionLogEntry[]
  currentStage: string
  errorMessage?: string
}

const ACTION_COLORS: Record<string, string> = {
  approve:         'text-emerald-400',
  reject:          'text-red-400',
  counter_propose: 'text-orange-400',
  pending:         'text-amber-400',
}

const ACTION_LABELS: Record<string, string> = {
  approve:         '✓ Approved',
  reject:          '✗ Rejected',
  counter_propose: '↺ Counter-proposed',
  pending:         '⏳ Awaiting decision',
}

export default function ChatPanel({ decisions, currentStage, errorMessage }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [decisions])

  return (
    <div className="card h-full flex flex-col">
      <div className="flex items-center gap-2 mb-4">
        <Bot size={16} className="text-brand-400" />
        <h3 className="section-title mb-0">Agent Reasoning Log</h3>
        <div className="ml-auto badge badge-info">{decisions.length} entries</div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-1">
        {decisions.length === 0 && (
          <div className="text-center text-slate-500 text-sm py-12">
            <Bot size={32} className="mx-auto mb-3 opacity-30" />
            <p>Agent decisions will appear here as the pipeline runs.</p>
          </div>
        )}

        {decisions.map((entry, i) => (
          <div
            key={entry.entry_id || i}
            className="animate-fade-in"
          >
            {/* Stage label */}
            <div className="flex items-center gap-2 mb-1.5">
              <div className="w-5 h-5 rounded-full bg-brand-900/60 border border-brand-800/50 flex items-center justify-center">
                <Bot size={11} className="text-brand-400" />
              </div>
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                {entry.stage.replace(/_/g, ' ')}
              </span>
              <span className="text-xs text-slate-600">
                {new Date(entry.timestamp).toLocaleTimeString()}
              </span>
            </div>

            {/* Agent proposal */}
            <div className="ml-7 bg-surface-700/40 rounded-xl rounded-tl-sm p-3 border border-surface-600/30 mb-2">
              <p className="text-sm text-slate-300 leading-relaxed">{entry.proposed_action}</p>
              {entry.reasoning && entry.reasoning !== entry.proposed_action && (
                <p className="text-xs text-slate-500 mt-1.5 leading-relaxed">{entry.reasoning}</p>
              )}
            </div>

            {/* User action */}
            {entry.user_action && entry.user_action !== 'pending' && (
              <div className="ml-7">
                <div className="flex items-center gap-2 mb-1">
                  <div className="w-5 h-5 rounded-full bg-surface-700 border border-surface-600 flex items-center justify-center">
                    <User size={11} className="text-slate-400" />
                  </div>
                  <span className={clsx('text-xs font-semibold', ACTION_COLORS[entry.user_action] || 'text-slate-400')}>
                    {ACTION_LABELS[entry.user_action] || entry.user_action}
                  </span>
                </div>

                {entry.user_note && (
                  <div className="ml-7 bg-surface-700/30 rounded-lg p-2.5 border border-surface-600/30 mb-2">
                    <p className="text-xs text-slate-400 italic">"{entry.user_note}"</p>
                  </div>
                )}

                {entry.agent_justification && (
                  <div className="ml-7 bg-brand-900/20 rounded-lg p-2.5 border border-brand-800/30 mb-2">
                    <p className="text-xs text-brand-300 leading-relaxed">{entry.agent_justification}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Current activity indicator */}
        {currentStage && decisions.length > 0 && (
          <div className="flex items-center gap-2 py-2">
            <RefreshCw size={12} className="text-brand-400 animate-spin" />
            <span className="text-xs text-slate-500">
              Running: {currentStage.replace(/_/g, ' ')}...
            </span>
          </div>
        )}

        {/* Error state */}
        {errorMessage && (
          <div className="flex items-start gap-2 bg-red-900/30 border border-red-800/50 rounded-xl p-3">
            <AlertCircle size={16} className="text-red-400 mt-0.5 flex-shrink-0" />
            <p className="text-sm text-red-300">{errorMessage}</p>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}
