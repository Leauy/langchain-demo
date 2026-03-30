import { Tag, Collapse } from 'antd'
import { FileTextOutlined } from '@ant-design/icons'

const { Panel } = Collapse

function SourceRef({ sources }) {
  if (!sources || sources.length === 0) return null

  return (
    <div className="sources-panel">
      <div style={{ marginBottom: 12, fontWeight: 500 }}>
        <FileTextOutlined style={{ marginRight: 8 }} />
        引用来源 ({sources.length})
      </div>
      <Collapse size="small">
        {sources.map((source, index) => (
          <Panel
            key={index}
            header={
              <div className="source-header">
                <span>
                  {source.module}
                  {source.sub_module && ` > ${source.sub_module}`}
                </span>
                <Tag color="blue">{(source.score * 100).toFixed(0)}%</Tag>
              </div>
            }
          >
            <div style={{ fontSize: 13, color: '#595959', whiteSpace: 'pre-wrap' }}>
              {source.content}
            </div>
          </Panel>
        ))}
      </Collapse>
    </div>
  )
}

export default SourceRef
