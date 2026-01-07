import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { api, Run, Metric, Distribution } from '../api/client'
import MetricsChart from '../components/MetricsChart'
import DistributionChart from '../components/DistributionChart'
import StatsCard from '../components/StatsCard'
import './RunDashboard.css'

function RunDashboard() {
  const { runId } = useParams<{ runId: string }>()
  const navigate = useNavigate()
  const [run, setRun] = useState<Run | null>(null)
  const [metrics, setMetrics] = useState<Metric[]>([])
  const [distributions, setDistributions] = useState<Distribution[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!runId) return
    loadData()
  }, [runId])

  const loadData = async () => {
    if (!runId) return

    try {
      setLoading(true)
      const [runData, metricsData, distributionsData] = await Promise.all([
        api.getRun(runId),
        api.getMetrics(runId),
        api.getDistributions(runId),
      ])
      setRun(runData)
      setMetrics(metricsData)
      setDistributions(distributionsData)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to load run data')
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="run-dashboard">
        <div className="loading">Loading run data...</div>
      </div>
    )
  }

  if (error || !run) {
    return (
      <div className="run-dashboard">
        <div className="error">Error: {error || 'Run not found'}</div>
        <button onClick={() => navigate('/')} className="back-button">
          ← Back to Runs
        </button>
      </div>
    )
  }

  const latestMetric = metrics[metrics.length - 1]
  const progress = run.config?.max_steps
    ? (latestMetric?.step || 0) / run.config.max_steps
    : 0

  return (
    <div className="run-dashboard">
      <div className="dashboard-header">
        <button onClick={() => navigate('/')} className="back-button">
          ← Back
        </button>
        <div className="header-content">
          <h2>{run.name}</h2>
          <div className="header-meta">
            <span className="env-badge">{run.env_id}</span>
            <span
              className="status-badge"
              data-status={run.status}
            >
              {run.status}
            </span>
          </div>
        </div>
        <button onClick={loadData} className="refresh-button">
          ↻ Refresh
        </button>
      </div>

      {run.config?.max_steps && (
        <div className="progress-section">
          <div className="progress-header">
            <span className="progress-label">Training Progress</span>
            <span className="progress-value">
              {latestMetric?.step || 0} / {run.config.max_steps} steps
              {' '}
              ({(progress * 100).toFixed(1)}%)
            </span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      )}

      <div className="stats-grid">
        {latestMetric?.reward_mean !== undefined && (
          <StatsCard
            title="Reward"
            value={latestMetric.reward_mean.toFixed(4)}
            subtitle={`±${(latestMetric.reward_std || 0).toFixed(4)}`}
            trend={getTrend(metrics, 'reward_mean')}
          />
        )}
        {latestMetric?.throughput !== undefined && (
          <StatsCard
            title="Throughput"
            value={latestMetric.throughput.toFixed(1)}
            subtitle="tokens/s"
            trend={getTrend(metrics, 'throughput')}
          />
        )}
        {latestMetric?.loss_mean !== undefined && (
          <StatsCard
            title="Loss"
            value={latestMetric.loss_mean.toFixed(4)}
            trend={getTrend(metrics, 'loss_mean', true)}
          />
        )}
        {latestMetric?.lr !== undefined && (
          <StatsCard
            title="Learning Rate"
            value={latestMetric.lr.toExponential(2)}
          />
        )}
      </div>

      <div className="charts-section">
        <h3>Metrics</h3>
        <div className="charts-grid">
          {hasMetric(metrics, 'reward_mean') && (
            <MetricsChart
              title="Reward"
              data={metrics}
              dataKeys={['reward_mean']}
              colors={['#8b5cf6']}
            />
          )}
          {hasMetric(metrics, 'loss_mean') && (
            <MetricsChart
              title="Loss"
              data={metrics}
              dataKeys={['loss_mean']}
              colors={['#ef4444']}
            />
          )}
          {hasMetric(metrics, 'throughput') && (
            <MetricsChart
              title="Throughput"
              data={metrics}
              dataKeys={['throughput', 'throughput_per_gpu']}
              colors={['#10b981', '#3b82f6']}
              labels={['Total', 'Per GPU']}
            />
          )}
          {hasMetric(metrics, 'entropy_mean') && (
            <MetricsChart
              title="Entropy"
              data={metrics}
              dataKeys={['entropy_mean']}
              colors={['#f59e0b']}
            />
          )}
          {hasMetric(metrics, 'grad_norm') && (
            <MetricsChart
              title="Gradient Norm"
              data={metrics}
              dataKeys={['grad_norm']}
              colors={['#06b6d4']}
            />
          )}
          {hasMetric(metrics, 'lr') && (
            <MetricsChart
              title="Learning Rate"
              data={metrics}
              dataKeys={['lr']}
              colors={['#ec4899']}
            />
          )}
        </div>
      </div>

      {distributions.length > 0 && (
        <div className="distributions-section">
          <h3>Latest Reward Distribution</h3>
          <DistributionChart distributions={distributions} />
        </div>
      )}
    </div>
  )
}

function hasMetric(metrics: Metric[], key: keyof Metric): boolean {
  return metrics.some((m) => m[key] !== undefined && m[key] !== null)
}

function getTrend(
  metrics: Metric[],
  key: keyof Metric,
  inverse = false
): 'up' | 'down' | 'flat' {
  if (metrics.length < 2) return 'flat'
  const recent = metrics.slice(-10)
  const first = recent[0][key] as number
  const last = recent[recent.length - 1][key] as number

  if (first === last) return 'flat'

  const increasing = last > first
  if (inverse) {
    return increasing ? 'down' : 'up'
  }
  return increasing ? 'up' : 'down'
}

export default RunDashboard
