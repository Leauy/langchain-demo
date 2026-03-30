import { Avatar } from 'antd'
import { UserOutlined, RobotOutlined } from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'

function MessageList({ messages }) {
  return (
    <div className="chat-messages">
      {messages.map(msg => (
        <div key={msg.id} className={`message-item ${msg.role}`}>
          <div className="message-avatar">
            <Avatar
              icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
              style={{
                backgroundColor: msg.role === 'user' ? '#1890ff' : '#52c41a'
              }}
            />
          </div>
          <div className="message-content">
            {msg.role === 'assistant' ? (
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            ) : (
              msg.content
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

export default MessageList
