import React from 'react'
import { 
  Users, 
  Clock, 
  CheckCircle, 
  Thermometer,
  HardDrive,
  Zap
} from 'lucide-react'
import { useDashboardStore } from '../stores/dashboardStore'
import { useWebSocket } from '../hooks/useWebSocket'
import { format } from 'date-fns'
import CollapsibleSection from './CollapsibleSection'

const Dashboard: React.FC = () => {
  const { metrics, recentFinds, lastUpdated } = useDashboardStore()
  const { isConnected } = useWebSocket()

  const metricCards = [
    {
      name: 'Active Workers',
      value: metrics?.worker_count || 0,
      icon: Users,
      color: 'text-blue-400',
      bgColor: 'bg-blue-400/10'
    },
    {
      name: 'Queue Length',
      value: metrics?.queue_length || 0,
      icon: Clock,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-400/10'
    },
    {
      name: 'Found Results',
      value: metrics?.found_results || 0,
      icon: CheckCircle,
      color: 'text-green-400',
      bgColor: 'bg-green-400/10'
    },
    {
      name: 'GPU Temperature',
      value: (metrics?.gpu_temps && metrics.gpu_temps.length > 0) ? `${Math.max(...metrics.gpu_temps)}°C` : 'N/A',
      icon: Thermometer,
      color: 'text-red-400',
      bgColor: 'bg-red-400/10'
    }
  ]

  return (
    <div className="space-y-4">

      {/* Metrics Overview */}
      <CollapsibleSection title="⚡ System Metrics" defaultExpanded={true}>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {metricCards.map((metric) => {
            const Icon = metric.icon
            return (
              <div
                key={metric.name}
                className="bg-dark-surface hacker-border rounded-lg p-4 hacker-glow-subtle hover:hacker-glow transition-all duration-300"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">{metric.name}</p>
                    <p className="text-xl font-bold text-hacker-green mt-1 font-mono">
                      {metric.value}
                    </p>
                  </div>
                  <Icon className={`h-5 w-5 ${metric.color}`} />
                </div>
              </div>
            )
          })}
        </div>
      </CollapsibleSection>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Recent Finds - Takes up 2 columns */}
        <div className="lg:col-span-2">
          <CollapsibleSection title="🔑 Recent Cracks" defaultExpanded={true}>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {recentFinds && recentFinds.length > 0 ? (
                recentFinds.slice(0, 8).map((find, index) => {
                const [hash, plaintext] = find.split(':', 2)
                return (
                  <div
                    key={index}
                    className="p-2 bg-dark-bg rounded border border-dark-border text-sm font-mono hover:bg-green-400/5 transition-colors"
                  >
                    <div className="text-green-400 break-all">
                      <span className="text-gray-500">{hash.substring(0, 16)}...</span>
                      <span className="text-hacker-green"> → </span>
                      <span className="text-green-300 font-bold">{plaintext}</span>
                    </div>
                  </div>
                )
              })
            ) : (
              <div className="text-gray-400 text-sm italic text-center py-6">
                🔍 No passwords cracked yet<br />
                <span className="text-xs">Upload hashes to begin...</span>
              </div>
            )}
            </div>
          </CollapsibleSection>
        </div>

        {/* System Health - Takes up 1 column */}
        <div>
          <CollapsibleSection title="📊 System Status" defaultExpanded={true}>
            <div className="space-y-2">
              {/* Connection Status */}
              <div className="flex items-center justify-between p-2 bg-dark-bg rounded border border-dark-border">
                <div className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-blue-400" />
                  <span className="text-xs">WebSocket</span>
                </div>
                <span className={`text-xs font-medium px-2 py-1 rounded ${isConnected ? 'text-green-400 bg-green-400/10' : 'text-red-400 bg-red-400/10'}`}>
                  {isConnected ? '🟢' : '🔴'}
                </span>
              </div>

              {/* Worker Status */}
              <div className="flex items-center justify-between p-2 bg-dark-bg rounded border border-dark-border">
                <div className="flex items-center space-x-2">
                  <Users className="h-4 w-4 text-purple-400" />
                  <span className="text-xs">Workers</span>
                </div>
                <span className="text-xs font-medium text-hacker-green bg-green-400/10 px-2 py-1 rounded">
                  {metrics?.worker_count || 0}
                </span>
              </div>

              {/* Queue Status */}
              <div className="flex items-center justify-between p-2 bg-dark-bg rounded border border-dark-border">
                <div className="flex items-center space-x-2">
                  <HardDrive className="h-4 w-4 text-yellow-400" />
                  <span className="text-xs">Queue</span>
                </div>
                <span className="text-xs font-medium text-hacker-green bg-yellow-400/10 px-2 py-1 rounded">
                  {metrics?.queue_length || 0}
                </span>
              </div>

              {/* GPU Temperature */}
              {metrics?.gpu_temps && metrics.gpu_temps.length > 0 && (
                <div className="flex items-center justify-between p-2 bg-dark-bg rounded border border-dark-border">
                  <div className="flex items-center space-x-2">
                    <Thermometer className="h-4 w-4 text-red-400" />
                    <span className="text-xs">GPU Temp</span>
                  </div>
                  <span className={`text-xs font-medium px-2 py-1 rounded ${
                    Math.max(...metrics.gpu_temps) > 80 ? 'text-red-400 bg-red-400/10' : 'text-green-400 bg-green-400/10'
                  }`}>
                    {Math.max(...metrics.gpu_temps)}°C
                  </span>
                </div>
              )}
            </div>
          </CollapsibleSection>
        </div>
      </div>

      {/* Last Updated */}
      {lastUpdated && (
        <div className="text-center text-xs text-gray-500 font-mono border-t border-dark-border pt-2 mt-4">
          ⏰ Last updated: {format(lastUpdated, 'PPpp')}
        </div>
      )}
    </div>
  )
}

export default Dashboard