import { useState, useEffect, useRef } from 'react'
import { Spin, Empty } from 'antd'
import MessageList from '../components/MessageList'
import ChatBox from '../components/ChatBox'
import SourceRef from '../components/SourceRef'
import { chatApi, datasourceApi } from '../api'

function Chat({ conversation, selectedDatasource, onDatasourceChange, onConversationUpdate }) {
  const [messages, setMessages] = useState([])
  const [currentSources, setCurrentSources] = useState([])
  const [datasources, setDatasources] = useState([])
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  // Load datasources
  useEffect(() => {
    const loadDatasources = async () => {
      try {
        const data = await datasourceApi.getDatasources()
        setDatasources(data.filter(ds => ds.status === 'ready'))
      } catch (error) {
        console.error('Failed to load datasources:', error)
      }
    }
    loadDatasources()
  }, [])

  // Load conversation messages
  useEffect(() => {
    if (conversation) {
      setMessages(conversation.messages || [])
      // Get sources from last assistant message
      const lastAssistant = [...(conversation.messages || [])]
        .reverse()
        .find(m => m.role === 'assistant')
      if (lastAssistant?.sources) {
        try {
          setCurrentSources(JSON.parse(lastAssistant.sources))
        } catch {
          setCurrentSources([])
        }
      } else {
        setCurrentSources([])
      }
    } else {
      setMessages([])
      setCurrentSources([])
    }
  }, [conversation])

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async (question) => {
    setLoading(true)
    setCurrentSources([])

    // Add user message immediately
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: question,
      created_at: new Date().toISOString()
    }
    setMessages(prev => [...prev, userMessage])

    try {
      const response = await chatApi.sendMessage(
        question,
        conversation?.id,
        selectedDatasource
      )

      // Add assistant message
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.answer,
        sources: JSON.stringify(response.sources),
        created_at: new Date().toISOString()
      }
      setMessages(prev => [...prev, assistantMessage])
      setCurrentSources(response.sources)

      // Update conversation
      if (!conversation) {
        onConversationUpdate()
      }
    } catch (error) {
      console.error('Failed to send message:', error)
      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: '抱歉，发生了错误，请稍后重试。',
        created_at: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-page">
      <div className="chat-header">
        <h2 style={{ margin: 0 }}>
          {conversation?.title || '知识问答'}
        </h2>
      </div>

      {messages.length > 0 ? (
        <>
          <MessageList messages={messages} />
          <div ref={messagesEndRef} />
          {loading && (
            <div style={{ padding: 24, textAlign: 'center' }}>
              <Spin tip="正在思考中..." />
            </div>
          )}
          {!loading && currentSources.length > 0 && (
            <SourceRef sources={currentSources} />
          )}
        </>
      ) : (
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <Empty description="开始一个新的对话" />
        </div>
      )}

      <ChatBox
        onSend={handleSend}
        onDatasourceChange={onDatasourceChange}
        selectedDatasource={selectedDatasource}
        datasources={datasources}
        loading={loading}
      />
    </div>
  )
}

export default Chat
