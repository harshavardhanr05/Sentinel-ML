/**
 * InsightDashboard.tsx
 * Recharts visualizations: model leaderboard, fairness cards, drift/robustness gauge.
 */

import React, { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, CartesianGrid, Legend,
  RadialBarChart, RadialBar,
  PieChart, Pie, Cell
} from 'recharts'
import { ShieldCheck, ShieldAlert, TrendingUp, Activity, ChevronDown, ChevronRight, CheckCircle, XCircle, RefreshCw, Info } from 'lucide-react'
import type { ModelLeaderboardEntry, GovernanceAudit } from '../api/client'
import clsx from 'clsx'

const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#3b82f6', '#14b8a6', '#f43f5e']

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
  const [activeLightboxChart, setActiveLightboxChart] = useState<any>(null)
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

        {governance.compliance_reasoning && (
          <div className="bg-brand-950/20 border border-brand-800/40 rounded-xl p-4 mb-5 flex gap-3">
            <Info size={16} className="text-brand-400 mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-xs font-semibold text-brand-400 uppercase tracking-wide">AI Governance Plan Justification</p>
              <p className="text-sm text-slate-300 mt-1 leading-relaxed">{governance.compliance_reasoning}</p>
            </div>
          </div>
        )}

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
            status={di !== null && di !== undefined && diThreshold !== undefined ? (di >= diThreshold ? 'PASS' : 'FAIL') : governance.fairness.status}
            description={`Protected: ${governance.fairness.protected_attribute || 'N/A'}`}
          />
          <MetricCard
            label="Equal Opp. Difference"
            value={eod}
            threshold={eodThreshold}
            status={eod !== null && eod !== undefined && eodThreshold !== undefined ? (eod <= eodThreshold ? 'PASS' : 'FAIL') : governance.fairness.status}
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

        {/* Visual Auditing (AI Charts) */}
        {(governance as any).ai_charts && (governance as any).ai_charts.length > 0 && (
          <div className="mt-8 pt-6 border-t border-surface-700/50">
            <h4 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
              <ShieldCheck size={15} className="text-brand-400" />
              Visual Auditing Reports
            </h4>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {(governance as any).ai_charts.map((chart: any, idx: number) => (
                <div key={chart.id || idx} className="bg-surface-800/60 backdrop-blur-md rounded-2xl border border-surface-600/40 p-4 flex flex-col gap-3">
                  <div>
                    <h5 className="text-sm font-bold text-slate-200">{chart.title}</h5>
                    {chart.insight && <p className="text-xs text-slate-500 mt-0.5">{chart.insight}</p>}
                  </div>
                  <div className="flex-1 bg-white/5 rounded-lg overflow-hidden flex items-center justify-center p-2 min-h-[220px]">
                    {chart.imageBase64 ? (
                      <div className="cursor-zoom-in w-full h-full flex items-center justify-center" onClick={() => setActiveLightboxChart(chart)}>
                        <img src={`data:image/png;base64,${chart.imageBase64}`} alt={chart.title} className="max-w-full max-h-[200px] object-contain rounded hover:scale-[1.01] transition-transform duration-200" style={{ imageRendering: 'auto' }} />
                      </div>
                    ) : (
                      <ResponsiveContainer width="100%" height={200}>
                        {chart.type === 'pie' || chart.type === 'doughnut' ? (
                          <PieChart>
                            <defs>
                              {COLORS.map((color, i) => (
                                <linearGradient id={`insight-pie-${chart.id}-${i}`} x1="0" y1="0" x2="0.8" y2="1" key={i}>
                                  <stop offset="0%" stopColor={color} stopOpacity={1} />
                                  <stop offset="100%" stopColor={color} stopOpacity={0.6} />
                                </linearGradient>
                              ))}
                              <filter id={`insight-shadow-${chart.id}`} x="-20%" y="-20%" width="140%" height="140%">
                                <feDropShadow dx="2" dy="5" stdDeviation="5" floodOpacity="0.4" floodColor="#000000" />
                              </filter>
                            </defs>
                            <Pie 
                              data={chart.data} cx="50%" cy="50%" 
                              innerRadius={chart.type === 'doughnut' ? 35 : 0} outerRadius={65} 
                              paddingAngle={5} dataKey="count" 
                              stroke="rgba(255,255,255,0.08)" strokeWidth={1.5}
                              filter={`url(#insight-shadow-${chart.id})`}
                            >
                              {chart.data.map((_: any, i: number) => <Cell key={i} fill={`url(#insight-pie-${chart.id}-${i % COLORS.length})`} />)}
                            </Pie>
                            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0', fontSize: 11 }} />
                            <Legend layout="horizontal" verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: '11px', paddingTop: '15px' }} />
                          </PieChart>
                        ) : chart.type === 'line' ? (
                          <LineChart data={chart.data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                            <XAxis dataKey="name" stroke="#94a3b8" fontSize={9} tick={{ fill: '#94a3b8' }} tickLine={false} />
                            <YAxis stroke="#94a3b8" fontSize={9} tick={{ fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0', fontSize: 11 }} />
                            <Line type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
                          </LineChart>
                        ) : (
                          <BarChart data={chart.data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                            <XAxis dataKey="name" stroke="#94a3b8" fontSize={9} tick={{ fill: '#94a3b8' }} tickLine={false} />
                            <YAxis stroke="#94a3b8" fontSize={9} tick={{ fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                            <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0', fontSize: 11 }} />
                            <Bar dataKey="count" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                          </BarChart>
                        )}
                      </ResponsiveContainer>
                    )}
                  </div>
                </div>
              ))}
            </div>
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
      {/* Lightbox Modal */}
      {activeLightboxChart && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-slate-950/80 backdrop-blur-md transition-all duration-300" onClick={() => setActiveLightboxChart(null)}>
          <div className="relative max-w-6xl w-full bg-surface-900 border border-surface-700/60 rounded-3xl p-6 shadow-2xl flex flex-col gap-4" onClick={(e) => e.stopPropagation()}>
            <button className="absolute top-4 right-4 text-slate-400 hover:text-slate-200 text-lg font-bold bg-surface-800 hover:bg-surface-700 w-8 h-8 flex items-center justify-center rounded-lg transition-colors" onClick={() => setActiveLightboxChart(null)}>✕</button>
            <div>
              <h3 className="text-lg font-bold text-slate-100 pr-8">{activeLightboxChart.title}</h3>
              {activeLightboxChart.insight && <p className="text-sm text-slate-400 mt-1">{activeLightboxChart.insight}</p>}
            </div>
            <div className="flex-1 bg-white/5 rounded-2xl overflow-hidden flex items-center justify-center p-4 min-h-[400px] max-h-[75vh]">
              <img src={`data:image/png;base64,${activeLightboxChart.imageBase64}`} alt={activeLightboxChart.title} className="max-h-full max-w-full object-contain rounded-xl" style={{ imageRendering: 'auto' }} />
            </div>
          </div>
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
