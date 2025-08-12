import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  Monitor, 
  Users, 
  BarChart3, 
  Settings, 
  Wifi,
  WifiOff
} from 'lucide-react'
import { useWebSocket } from '../hooks/useWebSocket'
import { useUiStore } from '../stores/uiStore'
import AsciiLogo from './AsciiLogo'
import MatrixRain from './MatrixRain'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation()
  const { isConnected } = useWebSocket()
  const { matrixRain } = useUiStore()

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Monitor },
    { name: 'Workers', href: '/workers', icon: Users },
    { name: 'Analytics', href: '/analytics', icon: BarChart3 },
    { name: 'Settings', href: '/settings', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Matrix Rain Background */}
      {matrixRain && <MatrixRain />}
      
      {/* ASCII Header */}
      <header className="bg-dark-surface border-b hacker-border">
        <div className="max-w-7xl mx-auto">
          {/* ASCII Logo */}
          <AsciiLogo />
          
          {/* Status Bar */}
          <div className="flex justify-center items-center px-4 pb-4">
            {/* Connection status - centered */}
            <div className="flex items-center space-x-2">
              {isConnected ? (
                <Wifi className="h-4 w-4 text-hacker-green" />
              ) : (
                <WifiOff className="h-4 w-4 text-error-red" />
              )}
              <span className={`text-xs ${isConnected ? 'text-hacker-green' : 'text-error-red'}`}>
                {isConnected ? 'Connected' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Bar */}
      <nav className="nav-background border-b hacker-border">
        <div className="w-full">
          <div className="flex justify-center space-x-12">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href
              const Icon = item.icon
              
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`
                    flex items-center space-x-3 px-8 py-4 text-sm font-bold
                    border-b-3 transition-all duration-300 rounded-t-lg
                    ${isActive 
                      ? 'border-green-400 text-green-400 hacker-glow bg-green-400/15 shadow-lg' 
                      : 'border-transparent text-white hover:text-green-400 hover:border-green-400/70 hover:bg-green-400/8'
                    }
                  `}
                >
                  <Icon className="h-5 w-5" />
                  <span className="font-bold text-base">{item.name}</span>
                </Link>
              )
            })}
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
  )
}

export default Layout