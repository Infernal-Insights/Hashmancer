import { useEffect, useRef, useCallback, useState } from 'react'
import { useDashboardStore } from '../stores/dashboardStore'
import { useAuthStore } from '../stores/authStore'
import toast from 'react-hot-toast'

interface WebSocketHook {
  socket: WebSocket | null
  isConnected: boolean
  connect: () => void
  disconnect: () => void
}

export const useWebSocket = (): WebSocketHook => {
  const socket = useRef<WebSocket | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const { token } = useAuthStore()
  const { updateMetrics, updateWorkers, updateFoundResults, setError } = useDashboardStore()
  const reconnectTimeoutRef = useRef<number | undefined>()

  const connect = useCallback(() => {
    if (socket.current?.readyState === WebSocket.OPEN || !token) return

    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.host
      const wsUrl = `${protocol}//${host}/ws/portal`
      
      socket.current = new WebSocket(wsUrl)

      socket.current.onopen = () => {
        setIsConnected(true)
        console.log('WebSocket connected')
        toast.success('Connected to live updates')
        
        // Clear any reconnection timeout
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current)
          reconnectTimeoutRef.current = undefined
        }
      }

      socket.current.onclose = () => {
        setIsConnected(false)
        console.log('WebSocket disconnected')
        toast.error('Lost connection to server')
        
        // Try to reconnect after 5 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect WebSocket...')
          connect()
        }, 5000)
      }

      socket.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        setError('WebSocket connection error')
        toast.error('Failed to connect to server')
        setIsConnected(false)
      }

      socket.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          if (data.metrics) {
            updateMetrics(data.metrics)
          }
          
          if (data.workers) {
            updateWorkers(data.workers)
          }
          
          if (data.founds && data.founds.length > 0) {
            updateFoundResults(data.founds)
            // Parse found results to show toast notifications
            data.founds.forEach((found: string) => {
              const [, plaintext] = found.split(':', 2)
              if (plaintext) {
                toast.success(`New crack found: ${plaintext}`)
              }
            })
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }
    } catch (error) {
      console.error('Error creating WebSocket connection:', error)
      setError('Failed to create WebSocket connection')
      toast.error('Failed to connect to server')
    }
  }, [token, updateMetrics, updateWorkers, updateFoundResults, setError])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = undefined
    }
    
    if (socket.current) {
      socket.current.close()
      socket.current = null
      setIsConnected(false)
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
    isConnected,
    connect,
    disconnect,
  }
}