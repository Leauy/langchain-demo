import { Popconfirm } from 'antd'
import { DeleteOutlined } from '@ant-design/icons'

function HistoryPanel({ conversations, currentId, onSelect, onDelete }) {
  const formatTime = (dateStr) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diff = now - date

    if (diff < 60000) return '刚刚'
    if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`
    if (diff < 86400000) return `${Math.floor(diff / 3600000)} 小时前`
    if (diff < 604800000) return `${Math.floor(diff / 86400000)} 天前`

    return date.toLocaleDateString()
  }

  return (
    <div className="sidebar-history">
      {conversations.map(conv => (
        <div
          key={conv.id}
          className={`history-item ${currentId === conv.id ? 'active' : ''}`}
          onClick={() => onSelect(conv)}
        >
          <div className="history-item-title">{conv.title}</div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="history-item-time">{formatTime(conv.updated_at)}</div>
            <Popconfirm
              title="确定删除此对话?"
              onConfirm={(e) => {
                e.stopPropagation()
                onDelete(conv.id)
              }}
              onCancel={(e) => e.stopPropagation()}
            >
              <DeleteOutlined
                onClick={(e) => e.stopPropagation()}
                style={{ color: 'rgba(255,255,255,0.45)' }}
              />
            </Popconfirm>
          </div>
        </div>
      ))}
      {conversations.length === 0 && (
        <div style={{ padding: 16, color: 'rgba(255,255,255,0.45)', textAlign: 'center' }}>
          暂无对话记录
        </div>
      )}
    </div>
  )
}

export default HistoryPanel
