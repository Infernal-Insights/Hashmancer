import React, { useState } from 'react'
import { 
  Settings as SettingsIcon, 
  Save, 
  Key, 
  Globe, 
  Brain,
  Shield,
  RefreshCw,
  LogOut,
  Eye,
  EyeOff
} from 'lucide-react'
import toast from 'react-hot-toast'
import { useAuthStore } from '../stores/authStore'
import { useUiStore } from '../stores/uiStore'

const Settings: React.FC = () => {
  const { user, logout } = useAuthStore()
  const { matrixRain, setMatrixRain } = useUiStore()
  const [hashesConfig, setHashesConfig] = useState({
    apiKey: '',
    algorithms: 'md5,sha1,sha256',
    pollInterval: 0
  })

  const [markovConfig, setMarkovConfig] = useState({
    language: 'english',
    probabilisticOrder: false,
    inverseOrder: false
  })

  const [uiConfig, setUiConfig] = useState({
    showApiKey: false
  })

  const [isLoading, setIsLoading] = useState(false)

  const handleSaveHashesConfig = async () => {
    setIsLoading(true)
    try {
      // API calls would go here
      await new Promise(resolve => setTimeout(resolve, 1000)) // Mock delay
      toast.success('Hashes.com configuration saved')
    } catch (error) {
      toast.error('Failed to save configuration')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSaveMarkovConfig = async () => {
    setIsLoading(true)
    try {
      // API calls would go here
      await new Promise(resolve => setTimeout(resolve, 1000)) // Mock delay
      toast.success('Markov configuration saved')
    } catch (error) {
      toast.error('Failed to save configuration')
    } finally {
      setIsLoading(false)
    }
  }

  const handleTrainMarkov = async () => {
    setIsLoading(true)
    try {
      // API call would go here
      await new Promise(resolve => setTimeout(resolve, 2000)) // Mock delay
      toast.success(`Markov training started for ${markovConfig.language}`)
    } catch (error) {
      toast.error('Failed to start Markov training')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-display text-glow-strong">
            Settings
          </h1>
          <p className="text-gray-400 mt-1">
            Configure system parameters and integrations
          </p>
        </div>
      </div>

      {/* Hashes.com Integration */}
      <div className="bg-dark-surface hacker-border rounded-lg p-6 hacker-glow-subtle">
        <h3 className="text-lg font-semibold text-hacker-green mb-4 flex items-center">
          <Globe className="h-5 w-5 mr-2" />
          Hashes.com Integration
        </h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              API Key
            </label>
            <div className="relative">
              <Key className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type={uiConfig.showApiKey ? "text" : "password"}
                value={hashesConfig.apiKey}
                onChange={(e) => setHashesConfig({...hashesConfig, apiKey: e.target.value})}
                className="input-hacker pl-10 pr-10 w-full"
                placeholder="Enter your Hashes.com API key"
              />
              <button
                type="button"
                onClick={() => setUiConfig({...uiConfig, showApiKey: !uiConfig.showApiKey})}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-hacker-green"
              >
                {uiConfig.showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Algorithms
            </label>
            <input
              type="text"
              value={hashesConfig.algorithms}
              onChange={(e) => setHashesConfig({...hashesConfig, algorithms: e.target.value})}
              className="input-hacker w-full"
              placeholder="md5,sha1,sha256"
            />
            <p className="text-xs text-gray-500 mt-1">
              Comma-separated list of hash algorithms
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Poll Interval (seconds)
            </label>
            <input
              type="number"
              value={hashesConfig.pollInterval}
              onChange={(e) => setHashesConfig({...hashesConfig, pollInterval: parseInt(e.target.value) || 0})}
              className="input-hacker w-full"
              min="0"
              placeholder="0"
            />
            <p className="text-xs text-gray-500 mt-1">
              Set to 0 to disable automatic polling
            </p>
          </div>

          <button
            onClick={handleSaveHashesConfig}
            disabled={isLoading}
            className="btn-primary flex items-center space-x-2"
          >
            <Save className="h-4 w-4" />
            <span>Save Hashes.com Config</span>
          </button>
        </div>
      </div>

      {/* Markov Configuration */}
      <div className="bg-dark-surface hacker-border rounded-lg p-6 hacker-glow-subtle">
        <h3 className="text-lg font-semibold text-hacker-green mb-4 flex items-center">
          <Brain className="h-5 w-5 mr-2" />
          Markov Training
        </h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Language
            </label>
            <select
              value={markovConfig.language}
              onChange={(e) => setMarkovConfig({...markovConfig, language: e.target.value})}
              className="input-hacker w-full"
            >
              <option value="english">English</option>
              <option value="german">German</option>
              <option value="french">French</option>
              <option value="spanish">Spanish</option>
            </select>
          </div>

          <div className="space-y-3">
            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={markovConfig.probabilisticOrder}
                onChange={(e) => setMarkovConfig({...markovConfig, probabilisticOrder: e.target.checked})}
                className="form-checkbox h-4 w-4 text-hacker-green bg-dark-bg border-gray-600 rounded focus:ring-hacker-green focus:ring-offset-0"
              />
              <span className="text-sm text-gray-300">Probabilistic Ordering</span>
            </label>

            <label className="flex items-center space-x-3">
              <input
                type="checkbox"
                checked={markovConfig.inverseOrder}
                onChange={(e) => setMarkovConfig({...markovConfig, inverseOrder: e.target.checked})}
                className="form-checkbox h-4 w-4 text-hacker-green bg-dark-bg border-gray-600 rounded focus:ring-hacker-green focus:ring-offset-0"
              />
              <span className="text-sm text-gray-300">Inverse Order</span>
            </label>
          </div>

          <div className="flex space-x-3">
            <button
              onClick={handleSaveMarkovConfig}
              disabled={isLoading}
              className="btn-primary flex items-center space-x-2"
            >
              <Save className="h-4 w-4" />
              <span>Save Config</span>
            </button>

            <button
              onClick={handleTrainMarkov}
              disabled={isLoading}
              className="btn-secondary flex items-center space-x-2"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Train Markov</span>
            </button>
          </div>
        </div>
      </div>

      {/* Security Settings */}
      <div className="bg-dark-surface hacker-border rounded-lg p-6 hacker-glow-subtle">
        <h3 className="text-lg font-semibold text-hacker-green mb-4 flex items-center">
          <Shield className="h-5 w-5 mr-2" />
          Security
        </h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-dark-bg rounded border border-dark-border">
            <div>
              <h4 className="text-sm font-medium text-gray-300">Session Timeout</h4>
              <p className="text-xs text-gray-500">Automatically log out after inactivity</p>
            </div>
            <select className="input-hacker text-sm">
              <option value="30">30 minutes</option>
              <option value="60">1 hour</option>
              <option value="120">2 hours</option>
              <option value="0">Never</option>
            </select>
          </div>

          <div className="flex items-center justify-between p-4 bg-dark-bg rounded border border-dark-border">
            <div>
              <h4 className="text-sm font-medium text-gray-300">WebSocket Connection</h4>
              <p className="text-xs text-gray-500">Real-time updates via WebSocket</p>
            </div>
            <label className="flex items-center">
              <input
                type="checkbox"
                defaultChecked
                className="form-checkbox h-4 w-4 text-hacker-green bg-dark-bg border-gray-600 rounded focus:ring-hacker-green focus:ring-offset-0"
              />
            </label>
          </div>

          <div className="flex items-center justify-between p-4 bg-dark-bg rounded border border-dark-border">
            <div>
              <h4 className="text-sm font-medium text-gray-300">Auto-refresh Data</h4>
              <p className="text-xs text-gray-500">Automatically refresh dashboard data</p>
            </div>
            <label className="flex items-center">
              <input
                type="checkbox"
                defaultChecked
                className="form-checkbox h-4 w-4 text-hacker-green bg-dark-bg border-gray-600 rounded focus:ring-hacker-green focus:ring-offset-0"
              />
            </label>
          </div>
        </div>
      </div>

      {/* UI Preferences */}
      <div className="bg-dark-surface hacker-border rounded-lg p-6 hacker-glow-subtle">
        <h3 className="text-lg font-semibold text-hacker-green mb-4 flex items-center">
          <Eye className="h-5 w-5 mr-2" />
          Visual Effects
        </h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-dark-bg rounded border border-dark-border">
            <div>
              <h4 className="text-sm font-medium text-gray-300">Matrix Rain Effect</h4>
              <p className="text-xs text-gray-500">Falling green text background animation</p>
            </div>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={matrixRain}
                onChange={(e) => {
                  setMatrixRain(e.target.checked)
                  toast.success(e.target.checked ? 'Matrix rain enabled' : 'Matrix rain disabled')
                }}
                className="form-checkbox h-4 w-4 text-hacker-green bg-dark-bg border-gray-600 rounded focus:ring-hacker-green focus:ring-offset-0"
              />
            </label>
          </div>
        </div>
      </div>

      {/* Account Management */}
      <div className="bg-dark-surface hacker-border rounded-lg p-6 hacker-glow-subtle">
        <h3 className="text-lg font-semibold text-hacker-green mb-4 flex items-center">
          <LogOut className="h-5 w-5 mr-2" />
          Account
        </h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-dark-bg rounded border border-dark-border">
            <div>
              <h4 className="text-sm font-medium text-gray-300">Current User</h4>
              <p className="text-xs text-gray-500">Logged in as: <span className="text-hacker-green">{user?.username}</span></p>
            </div>
          </div>

          <div className="flex items-center justify-between p-4 bg-dark-bg rounded border border-error-red">
            <div>
              <h4 className="text-sm font-medium text-error-red">Logout</h4>
              <p className="text-xs text-gray-500">End current session and return to login</p>
            </div>
            <button
              onClick={() => {
                logout()
                toast.success('Logged out successfully')
              }}
              className="btn-danger flex items-center space-x-2"
            >
              <LogOut className="h-4 w-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>

      {/* Current Status */}
      <div className="bg-dark-surface hacker-border rounded-lg p-6 hacker-glow-subtle">
        <h3 className="text-lg font-semibold text-hacker-green mb-4 flex items-center">
          <SettingsIcon className="h-5 w-5 mr-2" />
          Current Configuration
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium text-gray-300 mb-2">Markov Settings</h4>
            <div className="space-y-1 text-gray-400">
              <div>Language: <span className="text-hacker-green">English</span></div>
              <div>Probabilistic Order: <span className="text-red-400">OFF</span></div>
              <div>Inverse Order: <span className="text-red-400">OFF</span></div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-300 mb-2">Integration Status</h4>
            <div className="space-y-1 text-gray-400">
              <div>Hashes.com: <span className="text-yellow-400">Not Configured</span></div>
              <div>WebSocket: <span className="text-green-400">Connected</span></div>
              <div>Auto-refresh: <span className="text-green-400">Enabled</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Settings