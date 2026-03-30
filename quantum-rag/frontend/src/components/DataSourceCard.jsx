import { Card, Tag, Button, Popconfirm, Space } from 'antd'
import {
  EditOutlined,
  ReloadOutlined,
  DeleteOutlined,
  FileExcelOutlined,
  FilePdfOutlined,
  FileTextOutlined
} from '@ant-design/icons'

function DataSourceCard({ datasource, onEdit, onReindex, onDelete }) {
  const getFileIcon = (type) => {
    switch (type) {
      case 'excel':
        return <FileExcelOutlined style={{ color: '#52c41a' }} />
      case 'pdf':
        return <FilePdfOutlined style={{ color: '#f5222d' }} />
      default:
        return <FileTextOutlined />
    }
  }

  const getStatusTag = (status) => {
    switch (status) {
      case 'ready':
        return <Tag color="green">就绪</Tag>
      case 'processing':
        return <Tag color="blue">处理中</Tag>
      case 'error':
        return <Tag color="red">错误</Tag>
      default:
        return <Tag>{status}</Tag>
    }
  }

  return (
    <div className="datasource-card">
      <div className="datasource-card-header">
        <div className="datasource-card-title">
          {getFileIcon(datasource.file_type)}
          <span style={{ marginLeft: 8 }}>{datasource.name}</span>
        </div>
        {getStatusTag(datasource.status)}
      </div>

      {datasource.description && (
        <div style={{ color: '#8c8c8c', fontSize: 13, marginBottom: 8 }}>
          {datasource.description}
        </div>
      )}

      <div className="datasource-card-stats">
        <span>文档数: {datasource.document_count}</span>
        {datasource.vector_dimension && (
          <span>向量维度: {datasource.vector_dimension}</span>
        )}
      </div>

      <div className="datasource-card-actions">
        <Space>
          <Button
            size="small"
            icon={<EditOutlined />}
            onClick={() => onEdit(datasource)}
          >
            编辑
          </Button>
          <Popconfirm
            title="确定重新向量化?"
            onConfirm={() => onReindex(datasource.id)}
          >
            <Button
              size="small"
              icon={<ReloadOutlined />}
              disabled={datasource.status === 'processing'}
            >
              重建索引
            </Button>
          </Popconfirm>
          <Popconfirm
            title="确定删除此数据源?"
            onConfirm={() => onDelete(datasource.id)}
          >
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
            >
              删除
            </Button>
          </Popconfirm>
        </Space>
      </div>
    </div>
  )
}

export default DataSourceCard
