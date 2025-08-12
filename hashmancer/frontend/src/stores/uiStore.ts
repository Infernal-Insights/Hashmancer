import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface UiState {
  matrixRain: boolean
  setMatrixRain: (enabled: boolean) => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      matrixRain: false,
      setMatrixRain: (enabled: boolean) => set({ matrixRain: enabled })
    }),
    {
      name: 'ui-settings'
    }
  )
)