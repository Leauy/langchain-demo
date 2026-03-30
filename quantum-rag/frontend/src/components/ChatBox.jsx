import { useState } from 'react'
import { Input, Button, Select } from 'antd'
import { SendOutlined } from '@ant-design/icons'

const { TextArea } = Input
const { Option } = Select

function ChatBox({ onSend, onDatasourceChange, selectedDatasource, datasources, loading }) {
  const [input, setInput] = useState('')

  const handleSend = () => {
    if (!input.trim() || loading) return
    onSend(input.trim())
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-input">
      <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <Select
          placeholder="选择数据源"
          value={selectedDatasource}
          onChange={onDatasourceChange}
          style={{ width: 200 }}
          allowClear
        >
          {datasources.map(ds => (
            <Option key={ds.id} value={ds.id}>
              {ds.name}
            </Option>
          ))}
        </Select>
      </div>
      <div style={{ display: 'flex', gap: 12 }}>
        <TextArea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入您的问题..."
          autoSize={{ minRows: 1, maxRows: 4 }}
          style={{ flex: 1 }}
        />
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={handleSend}
          loading={loading}
          disabled={!input.trim()}
        >
          发送
        </Button>
      </div>
    </div>
  )
}

export default ChatBox
