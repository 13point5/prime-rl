import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, Run } from '../api/client'
import { formatDistanceToNow } from 'date-fns'
import './RunsList.css'

function RunsList() {
  const [runs, setRuns] = useState<Run[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    loadRuns()
  }, [])

  const loadRuns = async () => {
    try {
      setLoading(true)
      const data = await api.listRuns()
      setRuns(data)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to load runs')
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return '#10b981'
      case 'completed':
        return '#3b82f6'
      case 'failed':
        return '#ef4444'
      case 'cancelled':
        return '#6b7280'
      default:
        return '#9ca3af'
    }
  }

  if (loading) {
    return (
      <div className="runs-list">
        <h2>Training Runs</h2>
        <div className="loading">Loading runs...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="runs-list">
        <h2>Training Runs</h2>
        <div className="error">Error: {error}</div>
        <button onClick={loadRuns} className="retry-button">
          Retry
        </button>
      </div>
    )
  }

  if (runs.length === 0) {
    return (
      <div className="runs-list">
        <h2>Training Runs</h2>
        <div className="empty">No training runs found</div>
      </div>
    )
  }

  return (
    <div className="runs-list">
      <div className="runs-header">
        <h2>Training Runs</h2>
        <button onClick={loadRuns} className="refresh-button">
          ↻ Refresh
        </button>
      </div>

      <div className="runs-grid">
        {runs.map((run) => (
          <div
            key={run.id}
            className="run-card"
            onClick={() => navigate(`/runs/${run.id}`)}
          >
            <div className="run-card-header">
              <h3>{run.name}</h3>
              <span
                className="status-badge"
                style={{ backgroundColor: getStatusColor(run.status) }}
              >
                {run.status}
              </span>
            </div>

            <div className="run-card-details">
              <div className="detail-row">
                <span className="label">Environment:</span>
                <span className="value">{run.env_id}</span>
              </div>

              {run.gpu_type && (
                <div className="detail-row">
                  <span className="label">GPU:</span>
                  <span className="value">
                    {run.gpu_count}x {run.gpu_type}
                  </span>
                </div>
              )}

              {run.instance_id && (
                <div className="detail-row">
                  <span className="label">Instance:</span>
                  <span className="value">{run.instance_id}</span>
                </div>
              )}

              <div className="detail-row">
                <span className="label">Started:</span>
                <span className="value">
                  {formatDistanceToNow(new Date(run.created_at), {
                    addSuffix: true,
                  })}
                </span>
              </div>

              {run.completed_at && (
                <div className="detail-row">
                  <span className="label">Completed:</span>
                  <span className="value">
                    {formatDistanceToNow(new Date(run.completed_at), {
                      addSuffix: true,
                    })}
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default RunsList
