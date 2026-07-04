import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Label
} from 'recharts'
import { Activity, BarChart2, PieChart as PieChartIcon, TrendingUp } from 'lucide-react'

interface DataAnalysisDashboardProps {
  metrics: {
    target_distribution?: Record<string, number>
    numeric_correlations?: Record<string, number>
    categorical_distributions?: Record<string, Record<string, number>>
    error?: string
  }
}

const COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', '#f97316', '#eab308']

export default function DataAnalysisDashboard({ metrics }: DataAnalysisDashboardProps) {
  if (metrics?.error) {
    return (
      <div className="card text-center py-12">
        <Activity size={32} className="mx-auto mb-4 text-red-500" />
        <h3 className="text-lg font-medium text-slate-200">Analysis Error</h3>
        <p className="text-slate-400 mt-2">{metrics.error}</p>
      </div>
    )
  }

  // Format Correlation Data
  const correlationData = Object.entries(metrics?.numeric_correlations || {})
    .map(([feature, corr]) => ({
      feature,
      correlation: corr,
      abs_corr: Math.abs(corr)
    }))
    .sort((a, b) => b.abs_corr - a.abs_corr)
    .slice(0, 10)

  // Format Target Distribution
  const targetData = Object.entries(metrics?.target_distribution || {}).map(([key, val]) => ({
    name: key,
    value: Number((val * 100).toFixed(1))
  }))

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center gap-2 mb-6">
        <div className="p-2 bg-brand-900/30 rounded-lg">
          <Activity size={20} className="text-brand-400" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Data Analysis Dashboard</h2>
          <p className="text-sm text-slate-400">Descriptive statistics and feature distributions</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Target Distribution (Donut) */}
        <div className="card col-span-1 border border-surface-700 bg-surface-800/50 shadow-lg">
          <h3 className="text-sm font-medium text-slate-300 mb-4 flex items-center gap-2">
            <PieChartIcon size={16} className="text-brand-400" />
            Target Distribution
          </h3>
          <div className="h-64">
            {targetData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={targetData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {targetData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                    <Label 
                      value="Target" 
                      position="center" 
                      fill="#94a3b8"
                      className="text-sm font-medium"
                    />
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                    itemStyle={{ color: '#e2e8f0' }}
                    formatter={(value: number) => [`${value}%`, 'Frequency']}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-500 text-sm">
                No target data available
              </div>
            )}
          </div>
          <div className="mt-4 flex flex-wrap gap-2 justify-center">
            {targetData.map((entry, index) => (
              <div key={entry.name} className="flex items-center gap-1.5 text-xs text-slate-400">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
                {entry.name} ({entry.value}%)
              </div>
            ))}
          </div>
        </div>

        {/* Feature Correlations (Bar) */}
        <div className="card col-span-1 lg:col-span-2 border border-surface-700 bg-surface-800/50 shadow-lg">
          <h3 className="text-sm font-medium text-slate-300 mb-4 flex items-center gap-2">
            <TrendingUp size={16} className="text-purple-400" />
            Top Numeric Feature Correlations
          </h3>
          <div className="h-64">
            {correlationData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={correlationData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis 
                    dataKey="feature" 
                    stroke="#64748b" 
                    fontSize={12}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis stroke="#64748b" fontSize={12} />
                  <Tooltip
                    cursor={{ fill: '#334155', opacity: 0.4 }}
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                    itemStyle={{ color: '#e2e8f0' }}
                  />
                  <Bar dataKey="correlation" radius={[4, 4, 0, 0]}>
                    {correlationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.correlation > 0 ? '#10b981' : '#f43f5e'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-slate-500 text-sm">
                No numeric correlations available
              </div>
            )}
          </div>
        </div>

        {/* Categorical Distributions */}
        {Object.entries(metrics?.categorical_distributions || {}).slice(0, 3).map(([feature, dist], i) => {
          const data = Object.entries(dist).map(([val, pct]) => ({
            name: val,
            value: Number((pct * 100).toFixed(1))
          }))
          return (
            <div key={feature} className="card col-span-1 border border-surface-700 bg-surface-800/50 shadow-lg">
              <h3 className="text-sm font-medium text-slate-300 mb-4 flex items-center gap-2">
                <BarChart2 size={16} className="text-blue-400" />
                {feature} (Categorical)
              </h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                    <XAxis type="number" stroke="#64748b" fontSize={10} hide />
                    <YAxis dataKey="name" type="category" stroke="#64748b" fontSize={11} width={80} />
                    <Tooltip
                      cursor={{ fill: '#334155', opacity: 0.4 }}
                      contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                      formatter={(value: number) => [`${value}%`, 'Frequency']}
                    />
                    <Bar dataKey="value" fill={COLORS[i % COLORS.length]} radius={[0, 4, 4, 0]} barSize={20} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )
        })}

      </div>
    </div>
  )
}
