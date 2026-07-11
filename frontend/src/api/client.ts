/**
 * frontend/src/api/client.ts
 * Axios API client + WebSocket hook for Sentinel-ML backend.
 */

import axios from 'axios'
import { useEffect, useRef, useCallback } from 'react'

const API_BASE = (import.meta as any).env.VITE_API_URL || '/api'
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
const WS_BASE = (import.meta as any).env.VITE_WS_URL || `${wsProtocol}//${window.location.host}`

export const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
})

// ── Types ────────────────────────────────────────────────────────────────────

export interface DecisionCard {
  stage: string
  problem_context?: string
  action_taken?: string
  proposed_action: string
  reasoning: string
  alternatives_considered: string[]
  cost_estimate?: string
  metrics_summary: Record<string, unknown>
  requires_response: boolean
  ai_execution_logs?: any[]
}

export interface DecisionLogEntry {
  entry_id: string
  stage: string
  problem_context?: string
  action_taken?: string
  proposed_action: string
  reasoning: string
  alternatives_considered: string[]
  user_action: string
  user_note?: string
  agent_justification?: string
  timestamp: string
  decided_at?: string
  ai_execution_logs?: Array<{
    attempt: number
    code: string
    error?: string
    fixed_code?: string
    status?: string
  }>
}

export interface FairnessMetrics {
  disparate_impact?: number
  equal_opportunity_difference?: number
  per_group_confusion_matrices: Record<string, unknown>
  protected_attribute?: string
  threshold_used?: number
  status: string
}

export interface GovernanceAudit {
  fairness: FairnessMetrics
  robustness: { auc_degradation_pct?: number; status: string }
  stability: { metric_variance?: number; metric_std?: number; status: string }
  compliance_checklist: string[]
  compliance_thresholds: Record<string, number>
  compliance_reasoning?: string
  ai_charts?: Array<{ id: string; title: string; insight?: string; imageBase64: string }>
  overall_status: string
  failure_reasons: string[]
  loopback_target?: string
  iteration_count: number
}

export interface ModelLeaderboardEntry {
  model_name: string
  model_family: string
  hyperparameters: Record<string, any>
  auc_roc?: number
  f1_score?: number
  precision?: number
  recall?: number
  accuracy?: number
  rmse?: number
  mae?: number
  train_auc_roc?: number
  train_f1_score?: number
  train_rmse?: number
  train_mae?: number
  calibration_curve: Array<{ bin_mean_predicted: number; fraction_of_positives: number }>
  cost_estimate_seconds?: number
  cost_estimate_note?: string
  is_selected: boolean
  features_used?: string[]
  explainability_summary?: string
}

export interface StageStatuses {
  objective_intake: string
  compliance: string
  data_profiling: string
  feature_engineering: string
  model_selection: string
  governance: string
  explainability: string
  reporting: string
}

export interface PipelineState {
  run_id: string
  created_at: string
  updated_at: string
  current_stage: string
  stage_statuses: StageStatuses
  is_paused: boolean
  pending_approval?: DecisionCard
  error_message?: string
  objective: {
    raw_text: string
    task_type: string
    target_column?: string
    protected_attributes: string[]
    domain_tag: string
    is_ambiguous: boolean
    clarification_needed: string[]
    feature_selection_top_k?: number
  }
  data_analysis_metrics?: any
  data_health_report?: {
    row_count: number
    column_count: number
    missingness_flags: Record<string, number>
    leakage_flags: Array<{ column: string; reason: string; severity: string }>
    imbalance_ratio?: number
    imbalance_flag: boolean
    severity_summary: Record<string, string>
    profiling_notes: string[]
  }
  feature_log: {
    accepted: Array<{ feature: string; transformation?: string | null; status: string; reason: string; metric_delta?: number | null; governance_flagged?: boolean; imputation_strategy?: string | null }>
    rejected: Array<{ feature: string; transformation?: string | null; status: string; reason: string; governance_flagged?: boolean }>
    final_feature_set: string[]
  }
  model_leaderboard: ModelLeaderboardEntry[]
  selected_model_name?: string
  governance_audit: GovernanceAudit
  explainability: {
    global_shap_values: Record<string, number>
    top_features_summary: string[]
    local_examples: Array<unknown>
    shap_plot_path?: string
    llm_narrative?: string
  }
  decisions_log: DecisionLogEntry[]
  final_recommendation: string
  final_recommendation_reasoning?: string
  model_card_path?: string
  audit_trail_path?: string
}

// ── API functions ─────────────────────────────────────────────────────────────

export async function createRun(objective: string, file: File): Promise<{ run_id: string }> {
  const form = new FormData()
  form.append('objective', objective)
  form.append('file', file)
  const res = await api.post('/runs', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return res.data
}

export async function listRuns(): Promise<Array<{ run_id: string; current_stage: string; is_paused: boolean; created_at: string }>> {
  const res = await api.get('/runs')
  return res.data
}

export async function getRunState(runId: string): Promise<PipelineState> {
  const res = await api.get(`/runs/${runId}/state`)
  return res.data
}

export async function submitDecision(
  runId: string,
  action: 'approve' | 'reject' | 'counter_propose',
  note?: string
): Promise<{ agent_justification?: string; ai_execution_logs?: any[] }> {
  const res = await api.post(`/runs/${runId}/decision`, { action, note })
  return res.data
}

export async function updateObjective(runId: string, updates: {
  target_column?: string
  protected_attributes?: string[]
  domain_tag?: string
  task_type?: string
}): Promise<void> {
  await api.post(`/runs/${runId}/objective`, updates)
}

export async function deleteRun(runId: string): Promise<void> {
  await api.delete(`/runs/${runId}`)
}

// ── WebSocket hook ─────────────────────────────────────────────────────────────

export function useRunWebSocket(
  runId: string | null,
  onMessage: (state: PipelineState) => void
) {
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let isActive = true

    const connect = () => {
      if (!runId || !isActive) return
      const ws = new WebSocket(`${WS_BASE}/ws/${runId}`)

      ws.onmessage = (event) => {
        if (!isActive) return
        try {
          const data = JSON.parse(event.data) as PipelineState
          onMessage(data)
        } catch {}
      }

      ws.onclose = () => {
        if (isActive) {
          setTimeout(connect, 2000)
        }
      }

      wsRef.current = ws
    }

    connect()

    return () => {
      isActive = false
      if (wsRef.current) {
        wsRef.current.onclose = null // prevent reconnect loop
        wsRef.current.close()
      }
    }
  }, [runId, onMessage])
}
