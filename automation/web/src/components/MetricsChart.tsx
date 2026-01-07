import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { Metric } from '../api/client'
import './MetricsChart.css'

interface MetricsChartProps {
  title: string
  data: Metric[]
  dataKeys: (keyof Metric)[]
  colors: string[]
  labels?: string[]
}

function MetricsChart({ title, data, dataKeys, colors, labels }: MetricsChartProps) {
  return (
    <div className="metrics-chart">
      <h4>{title}</h4>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="step"
            stroke="#9ca3af"
            style={{ fontSize: '0.75rem' }}
          />
          <YAxis
            stroke="#9ca3af"
            style={{ fontSize: '0.75rem' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1a1a1a',
              border: '1px solid #333',
              borderRadius: '0.375rem',
              fontSize: '0.875rem',
            }}
            labelStyle={{ color: '#9ca3af' }}
          />
          <Legend
            wrapperStyle={{ fontSize: '0.875rem' }}
            iconType="line"
          />
          {dataKeys.map((key, index) => (
            <Line
              key={key as string}
              type="monotone"
              dataKey={key as string}
              stroke={colors[index] || '#8b5cf6'}
              strokeWidth={2}
              dot={false}
              name={labels?.[index] || (key as string)}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

export default MetricsChart
