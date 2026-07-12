/**
 * DataAnalysisDashboard.tsx
 * Interactive Tableau-style analytics dashboard.
 * Renders AI-chosen charts (Pie, Bar, Line, Scatter, Histogram) with
 * rich hover tooltips. Shows SMOTE class distribution update when applicable.
 */

import React, { useMemo, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, ScatterChart, Scatter, ZAxis,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
  AreaChart, Area, Legend
} from 'recharts'
import {
  Activity, BarChart2, PieChart as PieIcon, TrendingUp,
  Info, ArrowRight, Zap, Database, BarChart as BarChartIcon
} from 'lucide-react'

interface Props {
  metrics: Record<string, any>
  targetColumn?: string
}

const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#0ea5e9', '#f43f5e', '#84cc16', '#06b6d4']
const STAGE_COLORS: Record<string, string> = {
  data_profiling: '#6366f1',
  feature_engineering: '#f59e0b',
  model_selection: '#10b981',
  governance: '#ec4899',
}

// Tableau-like tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  const data = payload[0].payload
  const value = typeof payload[0].value === 'number' ? payload[0].value : null
  const total = data.total as number | undefined

  return (
    <div className="bg-surface-800 backdrop-blur-lg border border-surface-500/50 shadow-2xl p-4 rounded-xl min-w-[200px] z-50">
      <div className="text-sm font-bold text-slate-100 border-b border-surface-600 pb-2 mb-3">
        {label ?? data.name ?? data.category ?? '—'}
      </div>
      {payload.map((entry: any, i: number) => {
        const v = typeof entry.value === 'number' ? entry.value : 0
        const pct = total && total > 0 ? ((v / total) * 100).toFixed(1) : null
        return (
          <div key={i} className="space-y-1">
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full" style={{ background: entry.color }} />
                <span className="text-xs text-slate-400 capitalize">{entry.name || 'Count'}</span>
              </div>
              <span className="text-sm font-bold text-slate-100">{v.toLocaleString()}</span>
            </div>
            {pct && (
              <div className="flex items-center justify-between gap-4 pl-4">
                <span className="text-xs text-slate-500">% of Total</span>
                <span className="text-xs font-bold text-brand-400">{pct}%</span>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

const HistogramTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-surface-800 border border-surface-500/50 shadow-2xl p-3 rounded-xl text-sm z-50">
      <div className="text-slate-400 mb-1">Range</div>
      <div className="text-slate-100 font-bold">{d.rangeLabel}</div>
      <div className="mt-2 text-slate-400">Count</div>
      <div className="text-emerald-400 font-bold">{d.count}</div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-24 gap-6 text-center">
      <div className="w-20 h-20 rounded-2xl bg-surface-700/60 flex items-center justify-center shadow-xl">
        <Database size={36} className="text-slate-500" />
      </div>
      <div>
        <h3 className="text-lg font-semibold text-slate-400">No Data Yet</h3>
        <p className="text-sm text-slate-500 mt-1 max-w-xs">
          Analytics will appear here after the Data Profiling stage completes.
          Upload a dataset and start the pipeline to see insights.
        </p>
      </div>
    </div>
  )
}

export default function DataAnalysisDashboard({ metrics, targetColumn }: Props) {
  const [activeLightboxChart, setActiveLightboxChart] = useState<any>(null)
  const {
    ai_charts = [],
    categorical_distributions = {},
    numeric_correlations = {},
    numeric_histograms = {},
    target_distribution = {},
    post_smote_target_distribution,
  } = metrics

  const hasData = Object.keys(target_distribution).length > 0
    || Object.keys(categorical_distributions).length > 0
    || Object.keys(numeric_histograms).length > 0
    || ai_charts.length > 0

  // Build charts list (AI-selected or fallback)
  const chartsToRender = useMemo(() => {
    const combined: any[] = []
    
    // 1. Build individual category and numeric chart objects
    const catCharts: any[] = []
    Object.keys(categorical_distributions).slice(0, 15).forEach((col, idx) => {
      const dist = categorical_distributions[col] || {}
      const cardinality = Object.keys(dist).length
      let chartType = 'bar'
      if (cardinality <= 4) {
        chartType = idx % 2 === 0 ? 'pie' : 'doughnut'
      } else {
        chartType = idx % 2 === 0 ? 'bar' : 'radar'
      }
      catCharts.push({ id: `cat-${col}`, title: `${col} Distribution`, type: chartType, dataKeyX: col, insight: `Category breakdown for ${col}.` })
    })

    const numCharts: any[] = []
    Object.keys(numeric_histograms).slice(0, 10).forEach((col, idx) => {
      const chartType = idx % 2 === 0 ? 'histogram' : 'area'
      numCharts.push({ id: `hist-${col}`, title: `${col} Distribution`, type: chartType, dataKeyX: col, insight: `Value distribution for ${col}.` })
    })

    // 2. Interleave defaults for storytelling (never show same type consecutive, mix categorical & numeric)
    const defaults: any[] = []
    const targetDistKeys = Object.keys(target_distribution)
    if (targetDistKeys.length > 0) {
      if (targetDistKeys.length > 15 && numeric_histograms[targetColumn]) {
        defaults.push({ id: 'target-dist', title: `Target Distribution (${targetColumn})`, type: 'area', dataKeyX: targetColumn, insight: 'Distribution of continuous target values.' })
      } else {
        defaults.push({ id: 'target-dist', title: `Target Distribution (${targetColumn})`, type: 'pie', dataKeyX: targetColumn, insight: 'Class balance of the target variable.' })
      }
    }

    const maxLength = Math.max(catCharts.length, numCharts.length)
    for (let i = 0; i < maxLength; i++) {
      if (i < catCharts.length) {
        defaults.push(catCharts[i])
      }
      if (i < numCharts.length) {
        defaults.push(numCharts[i])
      }
    }

    combined.push(...defaults)

    // 3. Then add the complex AI/Python charts if they exist
    if (ai_charts && ai_charts.length > 0) {
      combined.push(...ai_charts)
    }
    
    return combined
  }, [ai_charts, target_distribution, categorical_distributions, numeric_histograms, targetColumn])

  const getChartData = (chart: any): any[] => {
    if (chart.data) return chart.data
    if (chart.type === 'histogram' || chart.type === 'area') {
      const hist = numeric_histograms[chart.dataKeyX]
      if (!hist) return []
      return hist.counts.map((count: number, i: number) => ({
        rangeLabel: `${hist.bins[i].toFixed(1)} – ${hist.bins[i + 1]?.toFixed(1) ?? ''}`,
        count,
        binStart: hist.bins[i],
      }))
    }
    if (chart.type === 'pie' || chart.type === 'doughnut' || chart.type === 'bar' || chart.type === 'radar') {
      const dist = chart.dataKeyX === targetColumn
        ? target_distribution
        : categorical_distributions[chart.dataKeyX] || {}
      const total = Object.values(dist).reduce((a: any, b: any) => a + b, 0) as number
      return Object.entries(dist)
        .map(([name, count]) => ({ name, count: count as number, total }))
        .sort((a, b) => b.count - a.count)
    }
    if (chart.type === 'line' || chart.type === 'scatter') {
      return Object.entries(numeric_correlations)
        .map(([name, val]) => ({ name, correlation: val as number }))
        .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
        .slice(0, 15)
    }
    return []
  }

  const renderChart = (chart: any) => {
    if (chart.imageBase64) {
      return (
        <div key={chart.id} className="bg-surface-800/60 backdrop-blur-md rounded-2xl border border-surface-600/40 shadow-xl hover:shadow-2xl hover:border-brand-500/30 transition-all duration-300 flex flex-col" style={{ height: 420 }}>
          {/* Header */}
          <div className="p-5 pb-0 flex-shrink-0">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-brand-600/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <BarChart2 size={16} className="text-brand-400" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-slate-100">{chart.title}</h4>
                {chart.insight && (
                  <p className="text-xs text-slate-500 mt-0.5 flex items-center gap-1.5 leading-relaxed">
                    <Info size={11} className="text-blue-400 flex-shrink-0" />{chart.insight}
                  </p>
                )}
              </div>
            </div>
          </div>
          {/* Chart Image */}
          <div className="flex-1 min-h-0 p-4 flex items-center justify-center cursor-zoom-in" onClick={() => setActiveLightboxChart(chart)}>
            <img src={`data:image/png;base64,${chart.imageBase64}`} alt={chart.title} className="max-h-full max-w-full object-contain rounded-lg bg-white/5 hover:scale-[1.01] transition-transform duration-200" style={{ imageRendering: 'auto' }} />
          </div>
        </div>
      )
    }

    const data = getChartData(chart)
    if (!data.length) return null

    const ChartIcon = chart.type === 'pie' || chart.type === 'doughnut' ? PieIcon
      : chart.type === 'histogram' || chart.type === 'area' ? BarChartIcon
      : chart.type === 'radar' ? Activity
      : chart.type === 'line' ? TrendingUp : BarChart2
      
    const safeId = chart.id ? chart.id.replace(/[^a-zA-Z0-9_-]/g, '') : 'default'

    return (
      <div key={chart.id} className="bg-surface-800/60 backdrop-blur-md rounded-2xl border border-surface-600/40 shadow-xl hover:shadow-2xl hover:border-brand-500/30 transition-all duration-300 flex flex-col" style={{ height: 380 }}>
        {/* Header */}
        <div className="p-5 pb-0 flex-shrink-0">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-lg bg-brand-600/20 flex items-center justify-center flex-shrink-0 mt-0.5">
              <ChartIcon size={16} className="text-brand-400" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-100">{chart.title}</h4>
              {chart.insight && (
                <p className="text-xs text-slate-500 mt-0.5 flex items-center gap-1.5 leading-relaxed">
                  <Info size={11} className="text-blue-400 flex-shrink-0" />{chart.insight}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="flex-1 min-h-0 px-4 pb-4 pt-3">
          <ResponsiveContainer width="100%" height="100%">
            {chart.type === 'pie' ? (
              <PieChart>
                <defs>
                  {COLORS.map((color, i) => (
                    <linearGradient id={`pie-grad-${safeId}-${i}`} x1="0" y1="0" x2="0.8" y2="1" key={i}>
                      <stop offset="0%" stopColor={color} stopOpacity={1} />
                      <stop offset="100%" stopColor={color} stopOpacity={0.6} />
                    </linearGradient>
                  ))}
                  <filter id={`pie-shadow-${safeId}`} x="-20%" y="-20%" width="140%" height="140%">
                    <feDropShadow dx="2" dy="5" stdDeviation="5" floodOpacity="0.4" floodColor="#000000" />
                  </filter>
                </defs>
                <Pie 
                  data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={95} 
                  paddingAngle={5} dataKey="count" 
                  stroke="rgba(255,255,255,0.08)" strokeWidth={1.5}
                  filter={`url(#pie-shadow-${safeId})`}
                >
                  {data.map((_: any, i: number) => <Cell key={i} fill={`url(#pie-grad-${safeId}-${i % COLORS.length})`} />)}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend layout="horizontal" verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: '11px', paddingTop: '15px' }} />
              </PieChart>
            ) : chart.type === 'doughnut' ? (
              <PieChart>
                <defs>
                  {COLORS.map((color, i) => (
                    <linearGradient id={`doughnut-grad-${safeId}-${i}`} x1="0" y1="0" x2="1" y2="1" key={i}>
                      <stop offset="0%" stopColor={color} stopOpacity={1} />
                      <stop offset="100%" stopColor={color} stopOpacity={0.5} />
                    </linearGradient>
                  ))}
                  <filter id={`doughnut-shadow-${safeId}`} x="-20%" y="-20%" width="140%" height="140%">
                    <feDropShadow dx="0" dy="8" stdDeviation="6" floodOpacity="0.4" floodColor="#000000" />
                  </filter>
                </defs>
                <Pie 
                  data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={90} 
                  paddingAngle={5} dataKey="count" 
                  stroke="rgba(255,255,255,0.1)" strokeWidth={1}
                  filter={`url(#doughnut-shadow-${safeId})`}
                >
                  {data.map((_: any, i: number) => <Cell key={i} fill={`url(#doughnut-grad-${safeId}-${(i + 2) % COLORS.length})`} />)}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend layout="horizontal" verticalAlign="bottom" align="center" wrapperStyle={{ fontSize: '11px', paddingTop: '15px' }} />
              </PieChart>
            ) : chart.type === 'radar' ? (
              <RadarChart cx="50%" cy="50%" outerRadius="75%" data={data.slice(0, 8)}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 9 }} />
                <PolarRadiusAxis stroke="#334155" angle={30} domain={[0, 'auto']} tick={{ fill: '#64748b', fontSize: 8 }} />
                <Radar name="Count" dataKey="count" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.35} />
                <Tooltip content={<CustomTooltip />} />
              </RadarChart>
            ) : chart.type === 'area' ? (
              <AreaChart data={data} margin={{ top: 0, right: 10, left: 0, bottom: 20 }}>
                <defs>
                  <linearGradient id={`grad-${safeId}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="binStart" tickFormatter={(v: number) => v.toFixed(1)} stroke="#94a3b8" fontSize={10} tick={{ fill: '#94a3b8' }} tickLine={false} angle={-30} textAnchor="end" />
                <YAxis stroke="#94a3b8" fontSize={10} tick={{ fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <Tooltip content={<HistogramTooltip />} />
                <Area type="monotone" dataKey="count" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill={`url(#grad-${safeId})`} />
              </AreaChart>
            ) : chart.type === 'histogram' ? (
              <BarChart data={data} margin={{ top: 0, right: 10, left: 0, bottom: 20 }}>
                <defs>
                  <linearGradient id={`grad-hist-${safeId}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#6366f1" stopOpacity={0.8}/>
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.4}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="binStart" tickFormatter={(v: number) => v.toFixed(1)} stroke="#94a3b8" fontSize={10} tick={{ fill: '#94a3b8' }} tickLine={false} angle={-30} textAnchor="end" />
                <YAxis stroke="#94a3b8" fontSize={10} tick={{ fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <Tooltip content={<HistogramTooltip />} cursor={{ fill: '#334155', opacity: 0.4 }} />
                <Bar dataKey="count" fill={`url(#grad-hist-${safeId})`} radius={[3, 3, 0, 0]} />
              </BarChart>
            ) : chart.type === 'line' ? (
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} tick={{ fill: '#94a3b8' }} tickLine={false} />
                <YAxis stroke="#94a3b8" fontSize={10} tick={{ fill: '#94a3b8' }} tickLine={false} axisLine={false} domain={[-1, 1]} />
                <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#64748b', strokeDasharray: '4 4' }} />
                <Line type="monotone" dataKey="correlation" stroke="#6366f1" strokeWidth={2.5} dot={{ r: 4, fill: '#6366f1', strokeWidth: 0 }} activeDot={{ r: 6 }} />
              </LineChart>
            ) : chart.type === 'scatter' ? (
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} tick={{ fill: '#94a3b8' }} />
                <YAxis dataKey="correlation" stroke="#94a3b8" fontSize={10} tick={{ fill: '#94a3b8' }} />
                <ZAxis range={[40, 200]} />
                <Tooltip content={<CustomTooltip />} />
                <Scatter data={data} fill="#10b981">
                  {data.map((_: any, i: number) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Scatter>
              </ScatterChart>
            ) : (
              /* Default: horizontal bar */
              <BarChart data={data.slice(0, 12)} layout="vertical" margin={{ top: 0, right: 30, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                <XAxis type="number" stroke="#94a3b8" fontSize={10} tick={{ fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                <YAxis dataKey="name" type="category" width={90} stroke="#94a3b8" fontSize={10} tick={{ fill: '#cbd5e1' }} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: '#334155', opacity: 0.4 }} />
                <Bar dataKey="count" radius={[0, 3, 3, 0]}>
                  {data.map((_: any, i: number) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Bar>
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
            <Activity className="text-brand-400" size={26} />
            Data Analytics
          </h2>
          {targetColumn && (
            <p className="text-sm text-slate-400 mt-1.5">
              Target Column:{' '}
              <span className="font-mono text-brand-300 bg-brand-900/30 px-2 py-0.5 rounded border border-brand-800/40">
                {targetColumn}
              </span>
            </p>
          )}
        </div>
        {hasData && (
          <div className="text-xs text-slate-500 flex items-center gap-1.5 bg-surface-700/40 px-3 py-1.5 rounded-lg border border-surface-600/30">
            <Zap size={12} className="text-emerald-400" />
            AI-selected insights
          </div>
        )}
      </div>

      {!hasData ? (
        <EmptyState />
      ) : (
        <>
          {/* Main charts grid */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {chartsToRender.map(renderChart)}
          </div>

          {/* SMOTE Section */}
          {post_smote_target_distribution && Object.keys(post_smote_target_distribution).length > 0 && (
            <div className="mt-6 bg-emerald-950/30 border border-emerald-800/40 rounded-2xl p-6 shadow-xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-9 h-9 rounded-xl bg-emerald-900/40 flex items-center justify-center">
                  <Zap size={18} className="text-emerald-400" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-emerald-300">SMOTE Applied — Class Rebalancing</h3>
                  <p className="text-xs text-slate-500 mt-0.5">Synthetic Minority Over-sampling was used to balance the class distribution for training.</p>
                </div>
              </div>
              <div className="flex flex-col md:flex-row gap-4">
                {[
                  { label: 'Before SMOTE', dist: target_distribution, color: '#f59e0b' },
                  { label: 'After SMOTE', dist: post_smote_target_distribution, color: '#10b981' },
                ].map(({ label, dist, color }) => {
                  const total = Object.values(dist).reduce((a: any, b: any) => a + b, 0) as number
                  const maxVal = Math.max(...(Object.values(dist) as number[]))
                  return (
                    <div key={label} className="flex-1 bg-surface-900/60 rounded-xl p-4 border border-surface-700/40">
                      <div className="flex items-center gap-2 mb-4">
                        <ArrowRight size={14} style={{ color }} />
                        <span className="text-sm font-semibold text-slate-300">{label}</span>
                      </div>
                      <div className="space-y-3">
                        {Object.entries(dist).map(([cls, count]) => {
                          const pct = total > 0 ? (((count as number) / total) * 100).toFixed(1) : '0'
                          const width = maxVal > 0 ? Math.max(((count as number) / maxVal) * 100, 4) : 4
                          return (
                            <div key={cls}>
                              <div className="flex items-center justify-between text-xs mb-1">
                                <span className="font-mono text-slate-300">{cls}</span>
                                <span className="text-slate-400">{(count as number).toLocaleString()} <span className="text-slate-600">({pct}%)</span></span>
                              </div>
                              <div className="h-2 bg-surface-700 rounded-full overflow-hidden">
                                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${width}%`, backgroundColor: color }} />
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Correlation table */}
          {Object.keys(numeric_correlations).length > 0 && (
            <div className="bg-surface-800/50 rounded-2xl border border-surface-600/40 p-6 shadow-lg">
              <h3 className="text-sm font-bold text-slate-300 mb-4 flex items-center gap-2">
                <TrendingUp size={15} className="text-brand-400" />
                Feature Correlations with Target
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-2">
                {Object.entries(numeric_correlations)
                  .sort(([, a], [, b]) => Math.abs(b as number) - Math.abs(a as number))
                  .slice(0, 12)
                  .map(([feat, corr]) => {
                    const v = corr as number
                    const abs = Math.abs(v)
                    const isPos = v >= 0
                    return (
                      <div key={feat} className="bg-surface-700/60 rounded-xl p-3 border border-surface-600/30 hover:border-brand-500/30 transition-colors">
                        <div className="text-xs text-slate-500 truncate mb-1" title={feat}>{feat}</div>
                        <div className={`text-lg font-bold tabular-nums ${isPos ? 'text-emerald-400' : 'text-red-400'}`}>
                          {v.toFixed(3)}
                        </div>
                        <div className="mt-1.5 h-1 bg-surface-600 rounded-full overflow-hidden">
                          <div className={`h-full rounded-full ${isPos ? 'bg-emerald-500' : 'bg-red-500'}`} style={{ width: `${abs * 100}%` }} />
                        </div>
                      </div>
                    )
                  })}
              </div>
            </div>
          )}
        </>
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
