import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  Monitor, 
  Users, 
  BarChart3, 
  Settings, 
  LogOut,
  Activity,
  Wifi,
  WifiOff
} from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { useWebSocket } from '../hooks/useWebSocket'
import { useDashboardStore } from '../stores/dashboardStore'
import { format } from 'date-fns'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation()
  const { user, logout } = useAuthStore()
  const { isConnected } = useWebSocket()
  const { lastUpdated } = useDashboardStore()

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Monitor },
    { name: 'Workers', href: '/workers', icon: Users },
    { name: 'Analytics', href: '/analytics', icon: BarChart3 },
    { name: 'Settings', href: '/settings', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Header */}
      <header className="bg-dark-surface border-b hacker-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-display text-glow-strong">
                ⛧ HASHMANCER ⛧
              </h1>
              <div className="hidden md:block text-xs text-gray-400">
                ~ The arcane art of cryptographic conjuration ~
              </div>
            </div>

            {/* Status indicators */}
            <div className="flex items-center space-x-4">
              {/* Connection status */}
              <div className="flex items-center space-x-2">
                {isConnected ? (
                  <>
                    <Wifi className="h-4 w-4 text-hacker-green" />
                    <span className="text-xs text-hacker-green">Live</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="h-4 w-4 text-error-red" />
                    <span className="text-xs text-error-red">Offline</span>
                  </>
                )}
              </div>

              {/* Last updated */}
              {lastUpdated && (
                <div className="hidden sm:block text-xs text-gray-400">
                  Last updated: {format(lastUpdated, 'HH:mm:ss')}
                </div>
              )}

              {/* User menu */}
              <div className="flex items-center space-x-2">
                <span className="text-sm text-hacker-green">
                  {user?.username}
                </span>
                <button
                  onClick={logout}
                  className="btn-danger flex items-center space-x-1 text-xs px-2 py-1"
                >
                  <LogOut className="h-3 w-3" />
                  <span>Logout</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <nav className="w-64 bg-dark-surface border-r hacker-border min-h-screen">
          <div className="p-4">
            <div className="space-y-2">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href
                const Icon = item.icon
                
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`
                      flex items-center space-x-3 px-3 py-2 rounded-md text-sm font-medium
                      transition-all duration-200
                      ${isActive 
                        ? 'bg-hacker-green text-black hacker-glow-strong' 
                        : 'text-hacker-green hover:bg-dark-border hover:hacker-glow'
                      }
                    `}
                  >
                    <Icon className="h-5 w-5" />
                    <span>{item.name}</span>
                  </Link>
                )
              })}
            </div>

            {/* Quick stats */}
            <div className="mt-8 pt-4 border-t border-dark-border">
              <div className="text-xs text-gray-400 mb-2">System Status</div>
              <div className="flex items-center space-x-2 text-xs">
                <Activity className="h-3 w-3 text-hacker-green animate-pulse" />
                <span>Monitoring Active</span>
              </div>
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1 p-6">
          <div className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}

export default Layout