import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

export interface Run {
  id: string
  name: string
  env_id: string
  status: string
  created_at: string
  updated_at: string
  completed_at?: string
  instance_id?: string
  gpu_type?: string
  gpu_count?: number
  config?: any
}

export interface Metric {
  step: number
  timestamp: string
  throughput?: number
  throughput_per_gpu?: number
  mfu?: number
  peak_memory?: number
  lr?: number
  grad_norm?: number
  loss_mean?: number
  entropy_mean?: number
  kl_mean?: number
  reward_mean?: number
  reward_std?: number
  reward_min?: number
  reward_max?: number
  extras?: Record<string, any>
}

export interface Distribution {
  step: number
  name: string
  values: number[]
  timestamp: string
}

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
})

export const api = {
  // Runs
  listRuns: async (limit = 100): Promise<Run[]> => {
    const response = await apiClient.get(`/runs`, { params: { limit } })
    return response.data
  },

  getRun: async (runId: string): Promise<Run> => {
    const response = await apiClient.get(`/runs/${runId}`)
    return response.data
  },

  // Metrics
  getMetrics: async (runId: string, limit = 1000, offset = 0): Promise<Metric[]> => {
    const response = await apiClient.get(`/runs/${runId}/metrics`, {
      params: { limit, offset },
    })
    return response.data
  },

  // Distributions
  getDistributions: async (runId: string, step?: number): Promise<Distribution[]> => {
    const response = await apiClient.get(`/runs/${runId}/distributions`, {
      params: { step },
    })
    return response.data
  },

  // Samples
  getSamples: async (runId: string, step?: number, limit = 100): Promise<any[]> => {
    const response = await apiClient.get(`/runs/${runId}/samples`, {
      params: { step, limit },
    })
    return response.data
  },
}

export default api
