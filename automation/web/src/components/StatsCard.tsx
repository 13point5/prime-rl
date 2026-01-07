import './StatsCard.css'

interface StatsCardProps {
  title: string
  value: string | number
  subtitle?: string
  trend?: 'up' | 'down' | 'flat'
}

function StatsCard({ title, value, subtitle, trend }: StatsCardProps) {
  return (
    <div className="stats-card">
      <div className="stats-card-header">
        <h4>{title}</h4>
        {trend && (
          <span className={`trend-indicator trend-${trend}`}>
            {trend === 'up' && '↑'}
            {trend === 'down' && '↓'}
            {trend === 'flat' && '→'}
          </span>
        )}
      </div>
      <div className="stats-card-value">{value}</div>
      {subtitle && <div className="stats-card-subtitle">{subtitle}</div>}
    </div>
  )
}

export default StatsCard
