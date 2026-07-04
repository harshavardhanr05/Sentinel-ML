/**
 * ExplainabilityPanel.tsx
 * SHAP global importance bar chart + local example breakdowns.
 */

import React, { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { Lightbulb, ChevronDown, ChevronUp } from 'lucide-react'

interface LocalExample {
  type: string
  sample_index: number
  actual_label: number
  predicted_label: number
  shap_breakdown: Record<string, number>
}

interface Props {
  globalShap: Record<string, number>
  topFeatures: string[]
  localExamples: LocalExample[]
}

export default function ExplainabilityPanel({ globalShap, topFeatures, localExamples }: Props) {
  const [expandedExample, setExpandedExample] = useState<number | null>(null)

  const chartData = Object.entries(globalShap)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 12)
    .map(([feature, val]) => ({ feature, importance: val }))

  const exampleTypeLabels: Record<string, string> = {
    correct_positive: '✅ Correct Positive',
    correct_negative: '✅ Correct Negative',
    misclassified: '⚠️ Misclassified',
  }

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-5">
        <Lightbulb size={18} className="text-yellow-400" />
        <h3 className="section-title mb-0">Explainability (SHAP)</h3>
      </div>

      {/* Top features summary */}
      {topFeatures.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-5">
          {topFeatures.map((f, i) => (
            <span key={f} className="text-xs bg-purple-900/40 border border-purple-800/50 text-purple-300 px-2.5 py-1 rounded-full font-medium">
              #{i + 1} {f}
            </span>
          ))}
        </div>
      )}

      {/* Global chart */}
      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={Math.max(160, chartData.length * 26)}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 32, top: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
            <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <YAxis dataKey="feature" type="category" tick={{ fill: '#94a3b8', fontSize: 10 }} width={130} />
            <Tooltip
              contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0', fontSize: 12 }}
              formatter={(val: number) => [val.toFixed(4), 'Mean |SHAP|']}
            />
            <Bar dataKey="importance" fill="#a78bfa" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <div className="text-center text-slate-500 text-sm py-8">
          SHAP values not yet computed.
        </div>
      )}

      {/* Local examples */}
      {localExamples.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-semibold text-slate-300 mb-3">Local Examples</h4>
          <div className="space-y-2">
            {localExamples.map((ex, i) => (
              <div key={i} className="border border-surface-700 rounded-xl overflow-hidden">
                <button
                  className="w-full flex items-center justify-between p-3 text-left hover:bg-surface-700/30 transition-colors"
                  onClick={() => setExpandedExample(expandedExample === i ? null : i)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-medium text-slate-200">
                      {exampleTypeLabels[ex.type] || ex.type}
                    </span>
                    <span className="text-xs text-slate-500">
                      Sample #{ex.sample_index} | Actual: {ex.actual_label} → Predicted: {ex.predicted_label}
                    </span>
                  </div>
                  {expandedExample === i ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                </button>

                {expandedExample === i && (
                  <div className="px-4 pb-4 border-t border-surface-700 pt-3">
                    <div className="space-y-2">
                      {Object.entries(ex.shap_breakdown)
                        .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                        .map(([feat, val]) => (
                          <div key={feat} className="flex items-center gap-3">
                            <span className="text-xs text-slate-400 w-32 truncate flex-shrink-0">{feat}</span>
                            <div className="flex-1 bg-surface-700 rounded h-2">
                              <div
                                className="h-2 rounded"
                                style={{
                                  width: `${Math.min(100, Math.abs(val) * 100)}%`,
                                  background: val > 0 ? '#6366f1' : '#ef4444',
                                  marginLeft: val < 0 ? 'auto' : 0,
                                }}
                              />
                            </div>
                            <span className={`text-xs font-mono w-16 text-right ${val > 0 ? 'text-brand-400' : 'text-red-400'}`}>
                              {val > 0 ? '+' : ''}{val.toFixed(3)}
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
