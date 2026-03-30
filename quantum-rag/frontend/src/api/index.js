import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000
})

// Chat API
export const chatApi = {
  sendMessage: async (question, conversationId = null, datasourceId = null) => {
    const response = await api.post('/chat', {
      question,
      conversation_id: conversationId,
      datasource_id: datasourceId
    })
    return response.data
  }
}

// History API
export const historyApi = {
  getConversations: async (skip = 0, limit = 50) => {
    const response = await api.get('/history', { params: { skip, limit } })
    return response.data
  },

  getConversation: async (id) => {
    const response = await api.get(`/history/${id}`)
    return response.data
  },

  deleteConversation: async (id) => {
    const response = await api.delete(`/history/${id}`)
    return response.data
  }
}

// Datasource API
export const datasourceApi = {
  getDatasources: async () => {
    const response = await api.get('/datasource')
    return response.data
  },

  getDatasource: async (id) => {
    const response = await api.get(`/datasource/${id}`)
    return response.data
  },

  createDatasource: async (formData) => {
    const response = await api.post('/datasource', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
  },

  updateDatasource: async (id, data) => {
    const response = await api.put(`/datasource/${id}`, data)
    return response.data
  },

  deleteDatasource: async (id) => {
    const response = await api.delete(`/datasource/${id}`)
    return response.data
  },

  reindexDatasource: async (id) => {
    const response = await api.post(`/datasource/${id}/reindex`)
    return response.data
  }
}

export default api
