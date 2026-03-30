import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useNavigate, useLocation } from 'react-router-dom'
import { Layout, Menu, Button, theme } from 'antd'
import {
  MessageOutlined,
  DatabaseOutlined,
  PlusOutlined
} from '@ant-design/icons'
import Chat from './pages/Chat'
import DataSource from './pages/DataSource'
import HistoryPanel from './components/HistoryPanel'
import { historyApi } from './api'

const { Sider, Content } = Layout

function AppContent() {
  const navigate = useNavigate()
  const location = useLocation()
  const [conversations, setConversations] = useState([])
  const [currentConversation, setCurrentConversation] = useState(null)
  const [selectedDatasource, setSelectedDatasource] = useState(null)

  const loadConversations = async () => {
    try {
      const data = await historyApi.getConversations()
      setConversations(data)
    } catch (error) {
      console.error('Failed to load conversations:', error)
    }
  }

  useEffect(() => {
    loadConversations()
  }, [])

  const handleNewChat = () => {
    setCurrentConversation(null)
    setSelectedDatasource(null)
    navigate('/')
  }

  const handleSelectConversation = async (conversation) => {
    try {
      const data = await historyApi.getConversation(conversation.id)
      setCurrentConversation(data)
      navigate('/')
    } catch (error) {
      console.error('Failed to load conversation:', error)
    }
  }

  const handleDeleteConversation = async (id) => {
    try {
      await historyApi.deleteConversation(id)
      setConversations(conversations.filter(c => c.id !== id))
      if (currentConversation?.id === id) {
        setCurrentConversation(null)
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error)
    }
  }

  const handleConversationUpdate = () => {
    loadConversations()
  }

  const menuItems = [
    {
      key: '/datasource',
      icon: <DatabaseOutlined />,
      label: '数据源管理'
    },
    {
      key: '/',
      icon: <MessageOutlined />,
      label: '知识问答'
    }
  ]

  return (
    <Layout className="app-container">
      <Sider width={280} className="sidebar">
        <div className="sidebar-header">
          量子网络设备知识库
        </div>

        <Menu
          theme="dark"
          mode="vertical"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          className="sidebar-menu"
        />

        {location.pathname === '/' && (
          <>
            <div className="sidebar-footer">
              <Button
                type="primary"
                icon={<PlusOutlined />}
                block
                onClick={handleNewChat}
              >
                新建对话
              </Button>
            </div>
            <HistoryPanel
              conversations={conversations}
              currentId={currentConversation?.id}
              onSelect={handleSelectConversation}
              onDelete={handleDeleteConversation}
            />
          </>
        )}
      </Sider>

      <Content className="main-content">
        <Routes>
          <Route
            path="/"
            element={
              <Chat
                conversation={currentConversation}
                selectedDatasource={selectedDatasource}
                onDatasourceChange={setSelectedDatasource}
                onConversationUpdate={handleConversationUpdate}
              />
            }
          />
          <Route path="/datasource" element={<DataSource />} />
        </Routes>
      </Content>
    </Layout>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  )
}

export default App
