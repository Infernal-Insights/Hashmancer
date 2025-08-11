# Hashmancer Frontend

Modern React-based frontend for the Hashmancer security platform.

## Features

- **Real-time Dashboard**: Live updates via WebSocket connections
- **Responsive Design**: Mobile-first design with desktop optimization
- **Modern UI**: Hacker-themed dark interface with terminal aesthetics
- **Progressive Web App**: Offline capability and mobile app-like experience
- **TypeScript**: Full type safety and IntelliSense support
- **State Management**: Zustand for reactive state management

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Development

The frontend is configured to proxy API requests to the backend server running on port 8000:

- Frontend dev server: `http://localhost:3000`
- Backend API: `http://localhost:8000` (proxied via `/api`)
- WebSocket: `ws://localhost:8000/ws` (proxied via `/ws`)

## Architecture

```
src/
├── components/     # React components
├── hooks/         # Custom React hooks
├── stores/        # Zustand state stores
├── types/         # TypeScript type definitions
├── utils/         # Utility functions
└── main.tsx       # Application entry point
```

## Key Components

- **Dashboard**: Real-time metrics and system overview
- **Workers**: Worker management and status monitoring
- **Analytics**: Advanced charts and performance analysis
- **Settings**: Configuration and preferences

## State Management

Uses Zustand for lightweight, reactive state management:

- `authStore`: Authentication and session management
- `dashboardStore`: Real-time metrics and data

## WebSocket Integration

Real-time updates are handled via Socket.IO client connecting to the portal WebSocket endpoint. Updates include:

- System metrics
- Worker status changes
- Found results (crack notifications)
- Job completion events

## Styling

- **Tailwind CSS**: Utility-first CSS framework
- **Custom Theme**: Hacker/terminal aesthetic with green-on-black color scheme
- **Responsive**: Mobile-first design with desktop enhancements
- **Animations**: Subtle glow effects and transitions

## Building

Production builds are automatically output to `../server/static/` for serving by the FastAPI backend.

```bash
npm run build
```

## Progressive Web App

Configured with Vite PWA plugin for:

- Offline functionality
- Mobile app installation
- Service worker for caching
- Web app manifest

## Development Workflow

1. Start the backend server: `uvicorn hashmancer.server.main:app --reload`
2. Start the frontend dev server: `npm run dev`
3. Open `http://localhost:3000` for live development
4. API calls are automatically proxied to the backend

## Type Safety

Full TypeScript integration with:

- Strict type checking
- Path aliases for clean imports
- Component props validation
- Store type definitions