import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { Distribution } from '../api/client'
import './MetricsChart.css'

interface DistributionChartProps {
  distributions: Distribution[]
}

function DistributionChart({ distributions }: DistributionChartProps) {
  // Get the latest distribution
  const latest = distributions[distributions.length - 1]

  if (!latest) {
    return <div className="metrics-chart">No distribution data available</div>
  }

  // Create histogram bins
  const bins = 30
  const values = latest.values
  const min = Math.min(...values)
  const max = Math.max(...values)
  const binSize = (max - min) / bins

  const histogram = Array.from({ length: bins }, (_, i) => {
    const binMin = min + i * binSize
    const binMax = binMin + binSize
    const count = values.filter((v) => v >= binMin && v < binMax).length
    return {
      bin: `${binMin.toFixed(2)}-${binMax.toFixed(2)}`,
      binLabel: binMin.toFixed(2),
      count,
    }
  })

  return (
    <div className="metrics-chart">
      <h4>{latest.name} (Step {latest.step})</h4>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={histogram} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="binLabel"
            stroke="#9ca3af"
            style={{ fontSize: '0.75rem' }}
            interval="preserveStartEnd"
          />
          <YAxis stroke="#9ca3af" style={{ fontSize: '0.75rem' }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1a1a1a',
              border: '1px solid #333',
              borderRadius: '0.375rem',
              fontSize: '0.875rem',
            }}
            labelStyle={{ color: '#9ca3af' }}
          />
          <Bar dataKey="count" fill="#8b5cf6" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default DistributionChart
