/**
 * InsightDashboard.tsx
 * Recharts visualizations: model leaderboard, fairness cards, drift/robustness gauge.
 */

import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend,
  RadialBarChart, RadialBar,
} from 'recharts'
import { ShieldCheck, ShieldAlert, TrendingUp, Activity, ChevronDown, ChevronRight, CheckCircle, XCircle, RefreshCw } from 'lucide-react'
import type { ModelLeaderboardEntry, GovernanceAudit } from '../api/client'
import clsx from 'clsx'

interface GovernanceLoopRecord {
  loop_number: number
  overall_result: string
  auc_roc?: number | null
  f1_score?: number | null
  rmse?: number | null
  mae?: number | null
  disparate_impact?: number | null
  equal_opportunity_difference?: number | null
  auc_degradation_pct?: number | null
  bootstrap_variance?: number | null
  failure_reasons: string[]
  corrective_action?: string | null
  llm_narrative?: string | null
  timestamp: string
}

interface Props {
  leaderboard: ModelLeaderboardEntry[]
  governance: GovernanceAudit
  featureImportance: Record<string, number>
  costEstimates: Record<string, any>
  finalFeatures?: string[]
  taskType?: string
}

function StatusBadge({ status }: { status: string }) {
  if (status === 'PASS') return <span className="badge-pass">PASS</span>
  if (status === 'FAIL') return <span className="badge-fail">FAIL</span>
  if (status === 'NOT_RUN') return <span className="badge-pending">Not Run</span>
  return <span className="badge-info">{status}</span>
}

function MetricCard({ label, value, threshold, status, description }: {
  label: string; value?: number | null; threshold?: number; status: string; description?: string
}) {
  const isPass = status === 'PASS'
  const isFail = status === 'FAIL'

  return (
    <div className={clsx(
      'card-sm border flex flex-col gap-3',
      isFail ? 'border-red-800/50 bg-red-950/20' : isPass ? 'border-emerald-800/30' : 'border-surface-600'
    )}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-300">{label}</span>
        <StatusBadge status={status} />
      </div>
      <div className="flex items-end gap-2">
        <span className={clsx(
          'text-3xl font-bold tabular-nums',
          isFail ? 'text-red-400' : isPass ? 'text-emerald-400' : 'text-slate-400'
        )}>
          {value != null ? value.toFixed(3) : '—'}
        </span>
        {threshold != null && (
          <span className="text-xs text-slate-500 mb-1">threshold: {threshold}</span>
        )}
      </div>
      {description && <p className="text-xs text-slate-500">{description}</p>}
    </div>
  )
}

export default function InsightDashboard({ leaderboard, governance, featureImportance, costEstimates, finalFeatures = [], taskType = 'classification' }: Props) {
  const isReg = taskType === 'regression' || taskType === 'REGRESSION'
  const metric1Label = isReg ? 'R2 Score' : 'AUC-ROC'
  const metric2Label = isReg ? 'RMSE' : 'F1 Score'

  // Model leaderboard chart data
  const leaderboardData = leaderboard.map(m => ({
    name: m.model_name.replace(' (Optuna)', ' ✨').replace('Logistic Regression', 'LR'),
    Metric1: m.auc_roc ?? 0,
    Metric2: isReg ? (m.rmse ?? 0) : (m.f1_score ?? 0),
    isSelected: m.is_selected,
  }))

  // Feature importance data
  const featImportanceData = Object.entries(featureImportance)
    .filter(([feat]) => finalFeatures.length === 0 || finalFeatures.includes(feat))
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([feat, val]) => ({ feature: feat, importance: val }))

  const di = governance.fairness.disparate_impact
  const eod = governance.fairness.equal_opportunity_difference
  const diThreshold = governance.compliance_thresholds?.disparate_impact_min ?? 0.80
  const eodThreshold = governance.compliance_thresholds?.equal_opportunity_diff_max ?? 0.10
  const robThreshold = governance.compliance_thresholds?.auc_degradation_max_pct ?? 10

  return (
    <div className="space-y-6">

      {/* Governance Overview */}
      <div className="card">
        <div className="flex items-center gap-2 mb-5">
          {governance.overall_status === 'PASS'
            ? <ShieldCheck size={18} className="text-emerald-400" />
            : <ShieldAlert size={18} className="text-red-400" />
          }
          <h3 className="section-title mb-0">Governance Audit</h3>
          <div className="ml-auto">
            <StatusBadge status={governance.overall_status} />
          </div>
        </div>

        {governance.failure_reasons.length > 0 && (
          <div className="bg-red-950/30 border border-red-800/40 rounded-xl p-4 mb-5">
            <p className="text-xs font-semibold text-red-400 uppercase tracking-wide mb-2">Failure Reasons</p>
            <ul className="space-y-1">
              {governance.failure_reasons.map((r, i) => (
                <li key={i} className="text-sm text-red-300 flex gap-2">
                  <span className="text-red-500 mt-0.5">•</span>
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <MetricCard
            label="Disparate Impact"
            value={di}
            threshold={diThreshold}
            status={governance.fairness.status}
            description={`Protected: ${governance.fairness.protected_attribute || 'N/A'}`}
          />
          <MetricCard
            label="Equal Opp. Difference"
            value={eod}
            threshold={eodThreshold}
            status={governance.fairness.status}
            description="Lower is better (|TPR_gap|)"
          />
          <MetricCard
            label="AUC Degradation %"
            value={governance.robustness.auc_degradation_pct}
            threshold={robThreshold}
            status={governance.robustness.status}
            description="Under synthetic covariate shift"
          />
        </div>

        {/* Compliance checklist */}
        {governance.compliance_checklist.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {governance.compliance_checklist.map(r => (
              <span key={r} className="badge badge-info">{r}</span>
            ))}
          </div>
        )}
      </div>

      {/* Model Leaderboard */}
      <div className="card">
        <div className="flex items-center gap-2 mb-5">
          <TrendingUp size={18} className="text-brand-400" />
          <h3 className="section-title mb-0">Model Leaderboard</h3>
        </div>
        
        {leaderboardData.length > 0 ? (
          <>
            <ResponsiveContainer width="100%" height={240}>
            <BarChart data={leaderboardData} margin={{ left: 0, right: 8, top: 4, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0' }}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Bar dataKey="Metric1" name={metric1Label} fill="#6366f1" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Metric2" name={metric2Label} fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>

          {/* Leaderboard table */}
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-700">
                  <th className="text-left pb-2 text-slate-400 font-medium">Model</th>
                  <th className="text-right pb-2 text-slate-400 font-medium">{metric1Label} <span className="text-[10px] text-slate-500 block">Val / Train</span></th>
                  <th className="text-right pb-2 text-slate-400 font-medium">{metric2Label} <span className="text-[10px] text-slate-500 block">Val / Train</span></th>
                  <th className="text-right pb-2 text-slate-400 font-medium">Est. Time</th>
                  <th className="text-right pb-2 text-slate-400 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((m, i) => (
                  <React.Fragment key={i}>
                    <tr className={clsx('border-b border-surface-700/50', m.is_selected && 'bg-brand-900/10')}>
                      <td className="py-2.5 font-medium text-slate-200">
                        {m.is_selected && <span className="text-brand-400 mr-1">★</span>}
                        {m.model_name}
                      </td>
                      <td className="py-2.5 text-right tabular-nums text-slate-300">
                        {m.auc_roc?.toFixed(4) ?? '—'}
                        {m.train_auc_roc != null && <span className="text-slate-500 ml-1 text-xs">/ {m.train_auc_roc?.toFixed(4)}</span>}
                      </td>
                      <td className="py-2.5 text-right tabular-nums text-slate-300">
                        {isReg ? (m.rmse?.toFixed(4) ?? '—') : (m.f1_score?.toFixed(4) ?? '—')}
                        {(isReg ? m.train_rmse : m.train_f1_score) != null && (
                          <span className="text-slate-500 ml-1 text-xs">/ {isReg ? m.train_rmse?.toFixed(4) : m.train_f1_score?.toFixed(4)}</span>
                        )}
                      </td>
                      <td className="py-2.5 text-right tabular-nums text-slate-400 text-xs">{m.cost_estimate_note ?? '—'}</td>
                      <td className="py-2.5 text-right">{m.is_selected && <span className="badge-info badge">Selected</span>}</td>
                    </tr>
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </>
        ) : (
          <div className="text-center py-12 border border-surface-700/50 border-dashed rounded-xl bg-surface-800/30">
            <Activity size={32} className="mx-auto mb-3 text-slate-600" />
            <p className="text-slate-400 font-medium">No models trained yet</p>
            <p className="text-slate-500 text-sm mt-1">Models will appear here once the Model Selection stage completes.</p>
          </div>
        )}
      </div>

      {/* Governance Loop History */}
      {(governance as any).governance_loop_history?.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-2 mb-5">
            <RefreshCw size={18} className="text-pink-400" />
            <h3 className="section-title mb-0">Governance Loop History</h3>
            <span className="ml-auto text-xs text-slate-500 bg-surface-700/50 px-2 py-0.5 rounded-full">
              {(governance as any).governance_loop_history.length} iteration{(governance as any).governance_loop_history.length > 1 ? 's' : ''}
            </span>
          </div>

          {/* Trend chart */}
          {(governance as any).governance_loop_history.length > 1 && (
            <ResponsiveContainer width="100%" height={160} className="mb-6">
              <LineChart data={(governance as any).governance_loop_history} margin={{ left: 0, right: 10, top: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="loop_number" tickFormatter={(v: number) => `Loop ${v}`} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0', fontSize: 12 }} />
                <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
                <Line type="monotone" dataKey="auc_roc" name={metric1Label} stroke="#6366f1" strokeWidth={2.5} dot={{ r: 4, fill: '#6366f1', strokeWidth: 0 }} />
                <Line type="monotone" dataKey="disparate_impact" name="Disparate Impact" stroke="#ec4899" strokeWidth={2.5} dot={{ r: 4, fill: '#ec4899', strokeWidth: 0 }} />
              </LineChart>
            </ResponsiveContainer>
          )}

          {/* Per-loop accordions */}
          <div className="space-y-3">
            {(governance as any).governance_loop_history.map((loop: GovernanceLoopRecord) => (
              <GovernanceLoopCard key={loop.loop_number} loop={loop} isReg={isReg} />
            ))}
          </div>
        </div>
      )}

      {/* Feature Importance */}
      {featImportanceData.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-2 mb-5">
            <Activity size={18} className="text-purple-400" />
            <h3 className="section-title mb-0">Feature Importance (SHAP)</h3>
          </div>
          <ResponsiveContainer width="100%" height={Math.max(200, featImportanceData.length * 28)}>
            <BarChart data={featImportanceData} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis
                dataKey="feature"
                type="category"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                width={140}
              />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0' }}
              />
              <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} label={{ position: 'right', fill: '#94a3b8', fontSize: 10 }} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

function GovernanceLoopCard({ loop, isReg }: { loop: GovernanceLoopRecord, isReg: boolean }) {
  const [expanded, setExpanded] = useState(false)
  const isPassed = loop.overall_result === 'PASS'

  return (
    <div className={clsx(
      'rounded-xl border p-4 transition-all duration-200',
      isPassed ? 'border-emerald-800/40 bg-emerald-950/20' : 'border-red-800/40 bg-red-950/20'
    )}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 text-left"
      >
        {isPassed
          ? <CheckCircle size={16} className="text-emerald-400 flex-shrink-0" />
          : <XCircle size={16} className="text-red-400 flex-shrink-0" />}
        <span className="text-sm font-bold text-slate-200 flex-1">Loop {loop.loop_number} — {loop.overall_result}</span>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          {loop.auc_roc != null && <span>{isReg ? 'R2' : 'AUC'}: <span className="text-slate-300 font-mono">{loop.auc_roc.toFixed(4)}</span></span>}
          {loop.disparate_impact != null && <span>DI: <span className={clsx('font-mono', loop.disparate_impact >= 0.8 ? 'text-emerald-400' : 'text-red-400')}>{loop.disparate_impact.toFixed(3)}</span></span>}
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </div>
      </button>

      {expanded && (
        <div className="mt-4 space-y-3 border-t border-surface-600/30 pt-4">
          {/* Metrics row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {[
              { label: isReg ? 'R2 Score' : 'AUC-ROC', value: loop.auc_roc, fmt: (v: number) => v.toFixed(4), color: 'text-brand-400' },
              { label: isReg ? 'RMSE' : 'F1 Score', value: isReg ? loop.rmse : loop.f1_score, fmt: (v: number) => v.toFixed(4), color: 'text-purple-400' },
              { label: 'Disparate Impact', value: loop.disparate_impact, fmt: (v: number) => v.toFixed(3), color: loop.disparate_impact != null && loop.disparate_impact >= 0.8 ? 'text-emerald-400' : 'text-red-400' },
              { label: 'EOD', value: loop.equal_opportunity_difference, fmt: (v: number) => v.toFixed(3), color: loop.equal_opportunity_difference != null && loop.equal_opportunity_difference <= 0.1 ? 'text-emerald-400' : 'text-red-400' },
              { label: 'Perf. Degradation %', value: loop.auc_degradation_pct, fmt: (v: number) => v.toFixed(2) + '%', color: loop.auc_degradation_pct != null && loop.auc_degradation_pct <= 10 ? 'text-emerald-400' : 'text-red-400' },
              { label: 'Bootstrap Var', value: loop.bootstrap_variance, fmt: (v: number) => v.toFixed(4), color: loop.bootstrap_variance != null && loop.bootstrap_variance <= 0.03 ? 'text-emerald-400' : 'text-red-400' },
            ].filter(m => m.value != null).map(m => (
              <div key={m.label} className="bg-surface-800/60 rounded-lg p-2.5 border border-surface-600/20">
                <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{m.label}</div>
                <div className={clsx('text-base font-bold font-mono tabular-nums', m.color)}>{m.fmt(m.value as number)}</div>
              </div>
            ))}
          </div>

          {/* LLM Narrative */}
          {loop.llm_narrative && (
            <div className="bg-surface-800/50 rounded-xl p-4 border border-surface-600/30">
              <div className="text-[10px] font-bold uppercase text-slate-500 tracking-wider mb-2">AI Governance Analysis</div>
              <p className="text-sm text-slate-300 leading-relaxed">{loop.llm_narrative}</p>
            </div>
          )}

          {/* Failure Reasons */}
          {loop.failure_reasons?.length > 0 && (
            <div className="bg-red-950/30 border border-red-800/30 rounded-xl p-3">
              <div className="text-[10px] font-bold uppercase text-red-400 tracking-wider mb-2">Failure Reasons</div>
              <ul className="space-y-1">
                {loop.failure_reasons.map((r, i) => (
                  <li key={i} className="text-sm text-red-300 flex gap-2"><span className="text-red-500">•</span>{r}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Corrective Action */}
          {loop.corrective_action && loop.corrective_action !== 'None (PASS)' && (
            <div className="flex items-center gap-2 text-xs">
              <RefreshCw size={12} className="text-amber-400" />
              <span className="text-slate-500">Corrective action:</span>
              <span className="text-amber-300 font-medium capitalize">{loop.corrective_action.replace(/_/g, ' ')}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
