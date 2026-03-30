import { useState, useEffect } from 'react'
import { Button, Modal, Form, Input, Upload, message } from 'antd'
import { PlusOutlined, UploadOutlined } from '@ant-design/icons'
import DataSourceCard from '../components/DataSourceCard'
import { datasourceApi } from '../api'

function DataSource() {
  const [datasources, setDatasources] = useState([])
  const [loading, setLoading] = useState(false)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [editModalVisible, setEditModalVisible] = useState(false)
  const [editingDatasource, setEditingDatasource] = useState(null)
  const [createForm] = Form.useForm()
  const [editForm] = Form.useForm()
  const [fileList, setFileList] = useState([])

  const loadDatasources = async () => {
    setLoading(true)
    try {
      const data = await datasourceApi.getDatasources()
      setDatasources(data)
    } catch (error) {
      console.error('Failed to load datasources:', error)
      message.error('加载数据源失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDatasources()
  }, [])

  const handleCreate = async (values) => {
    if (fileList.length === 0) {
      message.error('请选择要上传的文件')
      return
    }

    const formData = new FormData()
    formData.append('name', values.name)
    formData.append('description', values.description || '')
    formData.append('file', fileList[0].originFileObj)

    try {
      await datasourceApi.createDatasource(formData)
      message.success('数据源创建成功')
      setCreateModalVisible(false)
      createForm.resetFields()
      setFileList([])
      loadDatasources()
    } catch (error) {
      console.error('Failed to create datasource:', error)
      message.error('创建数据源失败')
    }
  }

  const handleEdit = async (values) => {
    try {
      await datasourceApi.updateDatasource(editingDatasource.id, values)
      message.success('数据源更新成功')
      setEditModalVisible(false)
      editForm.resetFields()
      setEditingDatasource(null)
      loadDatasources()
    } catch (error) {
      console.error('Failed to update datasource:', error)
      message.error('更新数据源失败')
    }
  }

  const handleReindex = async (id) => {
    try {
      await datasourceApi.reindexDatasource(id)
      message.success('开始重新向量化')
      loadDatasources()
    } catch (error) {
      console.error('Failed to reindex datasource:', error)
      message.error('重新向量化失败')
    }
  }

  const handleDelete = async (id) => {
    try {
      await datasourceApi.deleteDatasource(id)
      message.success('数据源删除成功')
      loadDatasources()
    } catch (error) {
      console.error('Failed to delete datasource:', error)
      message.error('删除数据源失败')
    }
  }

  const openEditModal = (datasource) => {
    setEditingDatasource(datasource)
    editForm.setFieldsValue({
      name: datasource.name,
      description: datasource.description
    })
    setEditModalVisible(true)
  }

  const uploadProps = {
    fileList,
    beforeUpload: (file) => {
      const isValidType = [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'application/pdf',
        'text/plain',
        'text/markdown'
      ].includes(file.type) ||
        file.name.endsWith('.md') ||
        file.name.endsWith('.txt')

      if (!isValidType) {
        message.error('只支持 Excel, PDF, TXT, Markdown 格式的文件')
        return false
      }

      setFileList([file])
      return false
    },
    onRemove: () => {
      setFileList([])
    }
  }

  return (
    <div className="datasource-page">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>数据源管理</h2>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setCreateModalVisible(true)}
        >
          新增数据源
        </Button>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', padding: 48 }}>
          加载中...
        </div>
      ) : (
        <div className="datasource-grid">
          {datasources.map(ds => (
            <DataSourceCard
              key={ds.id}
              datasource={ds}
              onEdit={openEditModal}
              onReindex={handleReindex}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}

      {datasources.length === 0 && !loading && (
        <div style={{ textAlign: 'center', padding: 48, color: '#8c8c8c' }}>
          暂无数据源，点击"新增数据源"开始创建
        </div>
      )}

      {/* Create Modal */}
      <Modal
        title="新增数据源"
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false)
          createForm.resetFields()
          setFileList([])
        }}
        onOk={() => createForm.submit()}
      >
        <Form form={createForm} layout="vertical" onFinish={handleCreate}>
          <Form.Item
            name="name"
            label="名称"
            rules={[{ required: true, message: '请输入数据源名称' }]}
          >
            <Input placeholder="输入数据源名称" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea placeholder="输入描述（可选）" rows={3} />
          </Form.Item>
          <Form.Item label="文件" required>
            <Upload {...uploadProps} maxCount={1}>
              <Button icon={<UploadOutlined />}>选择文件</Button>
            </Upload>
            <div style={{ color: '#8c8c8c', fontSize: 12, marginTop: 8 }}>
              支持 Excel (.xlsx, .xls), PDF, TXT, Markdown 格式
            </div>
          </Form.Item>
        </Form>
      </Modal>

      {/* Edit Modal */}
      <Modal
        title="编辑数据源"
        open={editModalVisible}
        onCancel={() => {
          setEditModalVisible(false)
          editForm.resetFields()
          setEditingDatasource(null)
        }}
        onOk={() => editForm.submit()}
      >
        <Form form={editForm} layout="vertical" onFinish={handleEdit}>
          <Form.Item
            name="name"
            label="名称"
            rules={[{ required: true, message: '请输入数据源名称' }]}
          >
            <Input placeholder="输入数据源名称" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea placeholder="输入描述（可选）" rows={3} />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default DataSource
