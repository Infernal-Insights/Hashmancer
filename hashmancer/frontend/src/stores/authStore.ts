import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AuthState {
  isAuthenticated: boolean
  user: {
    username: string
    role: string
  } | null
  token: string | null
  login: (passkey: string) => Promise<boolean>
  logout: () => void
  checkAuth: () => Promise<boolean>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      isAuthenticated: false,
      user: null,
      token: null,

      login: async (passkey: string) => {
        try {
          const response = await fetch('/login', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ passkey }),
          })

          if (response.ok) {
            const data = await response.json()
            set({
              isAuthenticated: true,
              user: { username: 'admin', role: 'admin' },
              token: data.cookie,
            })
            return true
          }
          return false
        } catch (error) {
          console.error('Login failed:', error)
          throw new Error('Invalid passkey')
        }
      },

      logout: () => {
        set({
          isAuthenticated: false,
          user: null,
          token: null,
        })
        
        // Call logout endpoint
        fetch('/api/logout', { method: 'POST' }).catch(console.error)
      },

      checkAuth: async () => {
        const { token } = get()
        if (!token) return false

        try {
          const response = await fetch('/api/verify', {
            headers: {
              'Authorization': `Bearer ${token}`,
            },
          })
          
          if (response.ok) {
            return true
          } else {
            get().logout()
            return false
          }
        } catch (error) {
          get().logout()
          return false
        }
      },
    }),
    {
      name: 'hashmancer-auth',
      partialize: (state) => ({
        isAuthenticated: state.isAuthenticated,
        user: state.user,
        token: state.token,
      }),
    }
  )
)