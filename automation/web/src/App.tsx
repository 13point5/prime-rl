import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import RunsList from './pages/RunsList'
import RunDashboard from './pages/RunDashboard'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <h1>Prime-RL Metrics Dashboard</h1>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<RunsList />} />
            <Route path="/runs/:runId" element={<RunDashboard />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
