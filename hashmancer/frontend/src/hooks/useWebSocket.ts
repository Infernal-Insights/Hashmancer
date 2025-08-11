import { useEffect, useRef, useCallback } from 'react'
import { io, Socket } from 'socket.io-client'
import { useDashboardStore } from '../stores/dashboardStore'
import { useAuthStore } from '../stores/authStore'
import toast from 'react-hot-toast'

interface WebSocketHook {
  socket: Socket | null
  isConnected: boolean
  connect: () => void
  disconnect: () => void
}

export const useWebSocket = (): WebSocketHook => {
  const socket = useRef<Socket | null>(null)
  const isConnectedRef = useRef(false)
  const { token } = useAuthStore()
  const { updateMetrics, updateWorkers, updateFoundResults, setError } = useDashboardStore()

  const connect = useCallback(() => {
    if (socket.current?.connected || !token) return

    socket.current = io('/ws/portal', {
      auth: {
        token,
      },
      transports: ['websocket', 'polling'],
    })

    socket.current.on('connect', () => {
      isConnectedRef.current = true
      console.log('WebSocket connected')
      toast.success('Connected to live updates')
    })

    socket.current.on('disconnect', () => {
      isConnectedRef.current = false
      console.log('WebSocket disconnected')
      toast.error('Lost connection to server')
    })

    socket.current.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error)
      setError(`Connection error: ${error.message}`)
      toast.error('Failed to connect to server')
    })

    // Listen for real-time updates
    socket.current.on('metrics_update', (data) => {
      updateMetrics(data)
    })

    socket.current.on('workers_update', (data) => {
      updateWorkers(data)
    })

    socket.current.on('found_results', (data) => {
      updateFoundResults(data)
      if (data.length > 0) {
        toast.success(`New crack found: ${data[0].plaintext}`)
      }
    })

    socket.current.on('worker_status_change', (data) => {
      toast.info(`Worker ${data.name} is now ${data.status}`)
    })

    socket.current.on('job_completed', (data) => {
      toast.success(`Job ${data.job_id} completed`)
    })

    socket.current.on('error', (error) => {
      console.error('WebSocket error:', error)
      setError(error.message)
    })
  }, [token, updateMetrics, updateWorkers, updateFoundResults, setError])

  const disconnect = useCallback(() => {
    if (socket.current) {
      socket.current.disconnect()
      socket.current = null
      isConnectedRef.current = false
    }
  }, [])

  useEffect(() => {
    if (token) {
      connect()
    } else {
      disconnect()
    }

    return () => {
      disconnect()
    }
  }, [token, connect, disconnect])

  return {
    socket: socket.current,
    isConnected: isConnectedRef.current,
    connect,
    disconnect,
  }
}