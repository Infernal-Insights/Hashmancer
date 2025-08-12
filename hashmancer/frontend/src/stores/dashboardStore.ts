import { create } from 'zustand'

interface Metrics {
  worker_count: number
  queue_length: number
  found_results: number
  cpu_usage: number
  memory_utilization: number
  disk_space: number
  gpu_temps: number[]
  cpu_load: number
  memory_usage: number
  backlog_target: number
  pending_jobs: number
  queued_batches: number
}

interface Worker {
  name: string
  status: 'idle' | 'maintenance' | 'offline'
  last_seen: string
  gpu_info?: string
  hashrate?: number
}

interface FoundResult {
  hash: string
  plaintext: string
  timestamp: string
  algorithm: string
}

interface DashboardState {
  metrics: Metrics | null
  workers: Worker[]
  foundResults: FoundResult[]
  recentFinds: string[]
  isLoading: boolean
  error: string | null
  lastUpdated: Date | null
  
  // Actions
  updateMetrics: (metrics: Metrics) => void
  updateWorkers: (workers: Worker[]) => void
  updateFoundResults: (results: FoundResult[]) => void
  updateWorkerStatus: (workerName: string, status: string) => Promise<void>
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  clearData: () => void
}

export const useDashboardStore = create<DashboardState>()((set) => ({
  metrics: null,
  workers: [],
  foundResults: [],
  recentFinds: [],
  isLoading: false,
  error: null,
  lastUpdated: null,

  updateMetrics: (metrics) => set({ 
    metrics, 
    lastUpdated: new Date(),
    error: null 
  }),

  updateWorkers: (workers) => set({ 
    workers, 
    lastUpdated: new Date(),
    error: null 
  }),

  updateFoundResults: (foundResults) => set({ 
    foundResults, 
    recentFinds: foundResults.map(r => r.plaintext).slice(0, 20),
    lastUpdated: new Date(),
    error: null 
  }),

  updateWorkerStatus: async (workerName: string, status: string) => {
    try {
      const response = await fetch('/worker_status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: workerName, status })
      })
      
      if (response.ok) {
        set((state) => ({
          workers: state.workers.map(worker => 
            worker.name === workerName ? { ...worker, status: status as any } : worker
          )
        }))
      } else {
        throw new Error('Failed to update worker status')
      }
    } catch (error) {
      throw new Error(`Failed to update worker status: ${error}`)
    }
  },

  setLoading: (isLoading) => set({ isLoading }),

  setError: (error) => set({ error, isLoading: false }),

  clearData: () => set({
    metrics: null,
    workers: [],
    foundResults: [],
    recentFinds: [],
    isLoading: false,
    error: null,
    lastUpdated: null,
  }),
}))