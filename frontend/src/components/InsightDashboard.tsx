/**
 * InsightDashboard.tsx
 * Recharts visualizations: model leaderboard, fairness cards, drift/robustness gauge.
 */

import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend,
  RadialBarChart, RadialBar,
} from 'recharts'
import { ShieldCheck, ShieldAlert, TrendingUp, Activity } from 'lucide-react'
import type { ModelLeaderboardEntry, GovernanceAudit } from '../api/client'
import clsx from 'clsx'

interface Props {
  leaderboard: ModelLeaderboardEntry[]
  governance: GovernanceAudit
  featureImportance: Record<string, number>
  costEstimates: Record<string, any>
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

export default function InsightDashboard({ leaderboard, governance, featureImportance, costEstimates }: Props) {
  // Model leaderboard chart data
  const leaderboardData = leaderboard.map(m => ({
    name: m.model_name.replace(' (Optuna)', ' ✨').replace('Logistic Regression', 'LR'),
    AUC: m.auc_roc ?? 0,
    F1: m.f1_score ?? 0,
    isSelected: m.is_selected,
  }))

  // Feature importance data
  const featImportanceData = Object.entries(featureImportance)
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
      {leaderboardData.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-2 mb-5">
            <TrendingUp size={18} className="text-brand-400" />
            <h3 className="section-title mb-0">Model Leaderboard</h3>
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={leaderboardData} margin={{ left: 0, right: 8, top: 4, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0' }}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Bar dataKey="AUC" fill="#6366f1" radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>

          {/* Leaderboard table */}
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-700">
                  <th className="text-left pb-2 text-slate-400 font-medium">Model</th>
                  <th className="text-right pb-2 text-slate-400 font-medium">AUC-ROC</th>
                  <th className="text-right pb-2 text-slate-400 font-medium">F1</th>
                  <th className="text-right pb-2 text-slate-400 font-medium">Est. Time</th>
                  <th className="text-right pb-2 text-slate-400 font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((m, i) => (
                  <tr key={i} className={clsx('border-b border-surface-700/50', m.is_selected && 'bg-brand-900/10')}>
                    <td className="py-2.5 font-medium text-slate-200">
                      {m.is_selected && <span className="text-brand-400 mr-1">★</span>}
                      {m.model_name}
                    </td>
                    <td className="py-2.5 text-right tabular-nums text-slate-300">{m.auc_roc?.toFixed(4) ?? '—'}</td>
                    <td className="py-2.5 text-right tabular-nums text-slate-300">{m.f1_score?.toFixed(4) ?? '—'}</td>
                    <td className="py-2.5 text-right tabular-nums text-slate-400 text-xs">{m.cost_estimate_note ?? '—'}</td>
                    <td className="py-2.5 text-right">{m.is_selected && <span className="badge-info badge">Selected</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
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
