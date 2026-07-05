/**
 * App.tsx — Main application shell for Sentinel-ML
 *
 * Layout:
 *   - Left: Pipeline DAG (live status)
 *   - Center/Right: Active content based on tab (Decision Card, Chat, Dashboard, Audit Trail, Explainability)
 *   - Top: Run management (new run, list of past runs)
 */

import React, { useState, useCallback } from 'react'
import {
  Shield, Upload, Play, ChevronRight, Database,
  BarChart2, Clock, Lightbulb, MessageSquare, Zap, AlertCircle, TrendingUp
} from 'lucide-react'
import {
  createRun, getRunState, listRuns, useRunWebSocket,
  type PipelineState
} from './api/client'
import PipelineDAG from './components/PipelineDAG'
import DecisionCard from './components/DecisionCard'
import ChatPanel from './components/ChatPanel'
import InsightDashboard from './components/InsightDashboard'
import DataAnalysisDashboard from './components/DataAnalysisDashboard'
import AuditTrailViewer from './components/AuditTrailViewer'
import ExplainabilityPanel from './components/ExplainabilityPanel'

type Tab = 'dag' | 'decisions' | 'data-analysis' | 'dashboard' | 'explainability' | 'chat' | 'audit'

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'dag', label: 'Pipeline', icon: <Play size={14} /> },
  { id: 'decisions', label: 'Decisions', icon: <Zap size={14} /> },
  { id: 'data-analysis', label: 'Data Analysis', icon: <TrendingUp size={14} /> },
  { id: 'chat', label: 'Reasoning', icon: <MessageSquare size={14} /> },
  { id: 'dashboard', label: 'Leaderboard', icon: <BarChart2 size={14} /> },
  { id: 'explainability', label: 'Explain', icon: <Lightbulb size={14} /> },
  { id: 'audit', label: 'Audit Trail', icon: <Clock size={14} /> },
]

const DEFAULT_STATE: Partial<PipelineState> = {
  stage_statuses: {
    objective_intake: 'pending', compliance: 'pending', data_profiling: 'pending',
    feature_engineering: 'pending', model_selection: 'pending', governance: 'pending',
    explainability: 'pending', reporting: 'pending',
  },
  current_stage: 'objective_intake',
  is_paused: false,
  decisions_log: [],
  model_leaderboard: [],
  governance_audit: {
    fairness: { status: 'NOT_RUN', per_group_confusion_matrices: {} },
    robustness: { status: 'NOT_RUN' },
    stability: { status: 'NOT_RUN' },
    compliance_checklist: [],
    compliance_thresholds: {},
    overall_status: 'NOT_RUN',
    failure_reasons: [],
    iteration_count: 0,
  },
  explainability: { global_shap_values: {}, top_features_summary: [], local_examples: [] },
  feature_log: { accepted: [], rejected: [], final_feature_set: [] },
  objective: { raw_text: '', task_type: 'unknown', protected_attributes: [], domain_tag: 'generic', is_ambiguous: false, clarification_needed: [] },
}

export default function App() {
  const [activeRunId, setActiveRunId] = useState<string | null>(null)
  const [pipelineState, setPipelineState] = useState<PipelineState | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>('dag')
  const [objective, setObjective] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [isStarting, setIsStarting] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)
  const [showNewRun, setShowNewRun] = useState(true)

  // WebSocket: update state live on every change
  const handleWsMessage = useCallback((state: PipelineState) => {
    setPipelineState(state)
    // Auto-switch to decisions tab when paused
    if (state.is_paused && state.pending_approval) {
      setActiveTab('decisions')
    }
  }, [])

  useRunWebSocket(activeRunId, handleWsMessage)

  const handleStartRun = async () => {
    if (!objective.trim() || !file) return
    setIsStarting(true)
    setStartError(null)
    try {
      const { run_id } = await createRun(objective, file)
      setActiveRunId(run_id)
      setShowNewRun(false)
      setActiveTab('dag')
      // Initial state poll
      const state = await getRunState(run_id)
      setPipelineState(state)
    } catch (e: any) {
      setStartError(e?.response?.data?.detail || 'Failed to start run')
    } finally {
      setIsStarting(false)
    }
  }

  const state = pipelineState

  const hasPendingDecision = state?.is_paused && state?.pending_approval
  const decisionCount = (state?.decisions_log?.filter(d => d.user_action === 'pending').length) || 0

  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="border-b border-surface-700 bg-surface-800/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 bg-gradient-to-br from-brand-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg shadow-brand-900/40">
              <Shield size={16} className="text-white" />
            </div>
            <span className="text-lg font-bold text-slate-100">Sentinel-ML</span>
            <span className="text-xs text-slate-500 mt-0.5">ML Governance Platform</span>
          </div>

          <div className="flex-1" />

          {activeRunId && (
            <div className="hidden sm:flex items-center gap-2 text-xs text-slate-500">
              <Database size={12} />
              <code className="font-mono text-slate-400">{activeRunId.slice(0, 8)}</code>
              {state?.is_paused && (
                <span className="badge-pending badge animate-pulse">⏳ Awaiting Decision</span>
              )}
              {!state?.is_paused && state?.current_stage && (
                <span className="badge-info badge">
                  {state.current_stage.replace(/_/g, ' ')}
                </span>
              )}
            </div>
          )}

          <button
            onClick={() => {
              setObjective('')
              setFile(null)
              setStartError(null)
              setShowNewRun(!showNewRun)
            }}
            className="btn-primary text-sm"
            id="new-run-btn"
          >
            <Upload size={14} />
            New Run
          </button>
        </div>
      </header>

      {/* ── New Run Form ────────────────────────────────────────────── */}
      {showNewRun && (
        <div className="bg-surface-800/60 border-b border-surface-700 backdrop-blur-sm">
          <div className="max-w-screen-2xl mx-auto px-6 py-6">
            <div className="max-w-2xl mx-auto animate-slide-up">
              <h2 className="text-xl font-semibold text-slate-100 mb-1">Start a New Pipeline Run</h2>
              <p className="text-sm text-slate-400 mb-5">
                Upload a tabular dataset (CSV/Parquet) and describe your objective in plain language.
              </p>

              <div className="space-y-4">
                <div>
                  <label htmlFor="objective-input" className="block text-sm font-medium text-slate-300 mb-1.5">
                    Business Objective
                  </label>
                  <textarea
                    id="objective-input"
                    value={objective}
                    onChange={e => setObjective(e.target.value)}
                    placeholder="e.g. Predict loan default. Minimize false negatives. Must be fair across gender and age. Domain: finance."
                    rows={3}
                    className="input-field resize-none"
                  />
                </div>

                <div>
                  <label htmlFor="dataset-upload" className="block text-sm font-medium text-slate-300 mb-1.5">
                    Dataset
                  </label>
                  <div className="relative">
                    <input
                      id="dataset-upload"
                      type="file"
                      accept=".csv,.parquet,.pq,.tsv"
                      onChange={e => setFile(e.target.files?.[0] || null)}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    />
                    <div className={`input-field flex items-center gap-3 cursor-pointer hover:border-brand-500 ${file ? 'border-brand-600' : ''}`}>
                      <Upload size={16} className="text-slate-400 flex-shrink-0" />
                      <span className={file ? 'text-slate-200' : 'text-slate-500'}>
                        {file ? file.name : 'Click to upload CSV or Parquet file'}
                      </span>
                    </div>
                  </div>
                </div>

                {startError && (
                  <div className="flex items-center gap-2 bg-red-900/30 border border-red-800/50 rounded-xl p-3 text-sm text-red-300">
                    <AlertCircle size={14} className="flex-shrink-0" />
                    {startError}
                  </div>
                )}

                <div className="flex gap-3">
                  <button
                    id="start-pipeline-btn"
                    onClick={handleStartRun}
                    disabled={!objective.trim() || !file || isStarting}
                    className="btn-primary flex-1 justify-center"
                  >
                    {isStarting ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Starting pipeline...
                      </>
                    ) : (
                      <>
                        <Play size={15} />
                        Start Pipeline
                        <ChevronRight size={14} />
                      </>
                    )}
                  </button>
                  {activeRunId && (
                    <button onClick={() => setShowNewRun(false)} className="btn-secondary">
                      Cancel
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Main content ────────────────────────────────────────────── */}
      {activeRunId && state ? (
        <div className="flex-1 max-w-screen-2xl mx-auto px-6 py-6 w-full">

          {/* Tab bar */}
          <div className="flex gap-1 mb-6 bg-surface-800 border border-surface-700 rounded-xl p-1 overflow-x-auto">
            {TABS.map(tab => (
              <button
                key={tab.id}
                id={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap relative
                  ${activeTab === tab.id
                    ? 'bg-brand-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-surface-700'
                  }`}
              >
                {tab.icon}
                {tab.label}
                {tab.id === 'decisions' && hasPendingDecision && (
                  <span className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-amber-400 rounded-full animate-pulse" />
                )}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="animate-fade-in">
            {activeTab === 'dag' && (
              <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 items-start">
                <div className="xl:col-span-3">
                  <div className="card p-0 overflow-hidden min-h-[650px] flex flex-col">
                    <div className="flex-1 w-full h-full">
                      <PipelineDAG
                        stageStatuses={state.stage_statuses}
                        currentStage={state.current_stage}
                        isPaused={state.is_paused}
                        governanceIterations={state.governance_audit?.iteration_count || 0}
                      />
                    </div>
                  </div>
                </div>
                <div className="space-y-4 xl:col-span-1">
                  {/* Run info card */}
                  <div className="card">
                    <h3 className="section-title text-sm">Run Info</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Task type</span>
                        <span className="font-medium text-slate-200">{state.objective.task_type || 'unknown'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Target column</span>
                        <code className="text-xs bg-surface-700 px-2 py-0.5 rounded text-brand-300">
                          {state.objective.target_column || 'TBD'}
                        </code>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Domain</span>
                        <span className="badge-info badge">{state.objective.domain_tag}</span>
                      </div>
                      {state.objective.protected_attributes.length > 0 && (
                        <div>
                          <span className="text-slate-400 text-xs">Protected attrs:</span>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {state.objective.protected_attributes.map(a => {
                              const reasoning = (state.objective as any).protected_attribute_reasoning?.[a]
                              return (
                                <span
                                  key={a}
                                  title={reasoning || 'Sensitive demographic attribute flagged for fairness monitoring'}
                                  className="text-xs bg-purple-900/40 border border-purple-800/40 text-purple-300 px-2 py-0.5 rounded cursor-help hover:bg-purple-900/60 hover:border-purple-700/60 transition-colors"
                                >
                                  {a}
                                </span>
                              )
                            })}
                          </div>
                        </div>
                      )}
                      {state.governance_audit?.iteration_count > 0 && (
                        <div className="flex justify-between">
                          <span className="text-slate-400">Governance loops</span>
                          <span className="font-bold text-orange-400">{state.governance_audit.iteration_count}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Pending decision alert */}
                  {hasPendingDecision && (
                    <div className="border border-amber-700/50 bg-amber-900/20 rounded-xl p-4">
                      <p className="text-sm text-amber-300 font-medium mb-2">⏳ Decision Required</p>
                      <p className="text-xs text-amber-400/80">{state.pending_approval?.proposed_action}</p>
                      <button
                        onClick={() => setActiveTab('decisions')}
                        className="btn-primary mt-3 text-xs w-full justify-center"
                      >
                        Review Decision <ChevronRight size={12} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeTab === 'decisions' && (
              <div className="max-w-2xl mx-auto">
                {hasPendingDecision ? (
                  <DecisionCard
                    runId={activeRunId}
                    card={state.pending_approval!}
                    onDecisionMade={() => {
                      // State will update via WebSocket
                    }}
                  />
                ) : (
                  <div className="card text-center py-16">
                    <Zap size={40} className="mx-auto mb-4 text-slate-600" />
                    <p className="text-slate-400 font-medium">No pending decisions.</p>
                    <p className="text-slate-500 text-sm mt-1">Decisions will appear here when the pipeline pauses at a checkpoint.</p>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'chat' && (
              <div style={{ height: 700 }}>
                <ChatPanel
                  decisions={state.decisions_log || []}
                  currentStage={state.current_stage}
                  errorMessage={state.error_message}
                />
              </div>
            )}

            {activeTab === 'data-analysis' && (
              <DataAnalysisDashboard
                metrics={state.data_analysis_metrics || {}}
                targetColumn={state.objective?.target_column}
              />
            )}

            {activeTab === 'dashboard' && (
              <InsightDashboard
                leaderboard={state.model_leaderboard || []}
                governance={state.governance_audit}
                featureImportance={state.explainability?.global_shap_values || {}}
                costEstimates={(state as any).cost_estimates || {}}
                finalFeatures={state.feature_log?.final_feature_set || []}
                taskType={state.objective?.task_type || 'classification'}
              />
            )}

            {activeTab === 'explainability' && (
              <ExplainabilityPanel
                globalShap={state.explainability?.global_shap_values || {}}
                topFeatures={state.explainability?.top_features_summary || []}
                localExamples={(state.explainability?.local_examples as any[]) || []}
                finalFeatures={state.feature_log?.final_feature_set || []}
                llmNarrative={state.explainability?.llm_narrative}
              />
            )}

            {activeTab === 'audit' && (
              <AuditTrailViewer
                decisions={state.decisions_log || []}
                runId={activeRunId}
                featureLog={state.feature_log}
                agentStepLog={(state as any).agent_step_log || []}
              />
            )}
          </div>
        </div>
      ) : !showNewRun ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Shield size={64} className="mx-auto mb-4 text-slate-700" />
            <h2 className="text-xl font-semibold text-slate-400 mb-2">No active run</h2>
            <p className="text-slate-500 text-sm mb-4">Start a new pipeline run to begin.</p>
            <button onClick={() => setShowNewRun(true)} className="btn-primary">
              <Upload size={14} />
              Start New Run
            </button>
          </div>
        </div>
      ) : null}
    </div>
  )
}
