#!/usr/bin/env python3
"""
Initialize vector database with sample data.

Usage:
    python init_vectordb.py <datasource_id>
    python init_vectordb.py --sample  # Create sample datasource
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import init_db, SessionLocal, Datasource, DatasourceStatus
from backend.services import vectorstore_service
from backend.services.document_loader import document_loader
from langchain_core.documents import Document


def create_sample_datasource():
    """Create a sample datasource with demo documents."""
    db = SessionLocal()

    try:
        # Check if sample already exists
        existing = db.query(Datasource).filter(
            Datasource.name == "示例数据源"
        ).first()

        if existing:
            print(f"Sample datasource already exists with ID: {existing.id}")
            return existing.id

        # Create sample datasource
        datasource = Datasource(
            name="示例数据源",
            description="量子网络设备示例知识库",
            file_type="text",
            status=DatasourceStatus.PROCESSING.value
        )
        db.add(datasource)
        db.commit()
        db.refresh(datasource)

        # Create sample documents
        documents = [
            Document(
                page_content="QKM-S600量子密钥管理设备是一款高性能的量子密钥分发系统，支持最大密钥生成速率100kbps，可管理最多256个节点。",
                metadata={
                    "module": "设备概述",
                    "sub_module": "QKM-S600",
                    "source": "overview"
                }
            ),
            Document(
                page_content="QKM-S800是增强型量子密钥管理设备，在S600基础上增加了网络接口冗余和双电源供电，适用于核心网络部署。",
                metadata={
                    "module": "设备概述",
                    "sub_module": "QKM-S800",
                    "source": "overview"
                }
            ),
            Document(
                page_content="SSH采集引擎支持通过SSH协议远程采集网络设备配置信息，兼容Cisco、Huawei、H3C等主流厂商设备。",
                metadata={
                    "module": "引擎管理",
                    "sub_module": "SSH采集引擎",
                    "source": "engine"
                }
            ),
            Document(
                page_content="SNMP采集引擎使用SNMP v2c/v3协议获取设备性能指标，支持CPU、内存、接口流量等监控项。",
                metadata={
                    "module": "引擎管理",
                    "sub_module": "SNMP采集引擎",
                    "source": "engine"
                }
            ),
            Document(
                page_content="量子密钥管理系统的部署要求：服务器CPU不低于16核，内存64GB以上，存储空间500GB以上，网络带宽1Gbps以上。",
                metadata={
                    "module": "部署要求",
                    "sub_module": "硬件要求",
                    "source": "deploy"
                }
            ),
            Document(
                page_content="系统支持27种量子网络设备型号的自动发现和配置管理，包括QKM-S600、QKM-S800、QRNG-100等系列。",
                metadata={
                    "module": "设备管理",
                    "sub_module": "支持型号",
                    "source": "devices"
                }
            ),
        ]

        # Create vector store
        success = vectorstore_service.create_store(datasource.id, documents)

        if success:
            datasource.status = DatasourceStatus.READY.value
            datasource.document_count = len(documents)
            datasource.vector_dimension = 1024
            db.commit()
            print(f"Sample datasource created successfully with ID: {datasource.id}")
            return datasource.id
        else:
            datasource.status = DatasourceStatus.ERROR.value
            db.commit()
            print("Failed to create vector store for sample datasource")
            return None

    finally:
        db.close()


def init_datasource(datasource_id: int):
    """Reinitialize vector store for an existing datasource."""
    db = SessionLocal()

    try:
        datasource = db.query(Datasource).filter(
            Datasource.id == datasource_id
        ).first()

        if not datasource:
            print(f"Datasource with ID {datasource_id} not found")
            return False

        if not datasource.file_path or not os.path.exists(datasource.file_path):
            print(f"Source file not found for datasource {datasource_id}")
            return False

        print(f"Processing datasource: {datasource.name}")

        # Load documents
        documents = document_loader.load_file(
            datasource.file_path,
            datasource.file_type
        )
        print(f"Loaded {len(documents)} documents")

        # Delete old index and create new one
        vectorstore_service.delete_store(datasource_id)
        success = vectorstore_service.create_store(datasource_id, documents)

        if success:
            datasource.status = DatasourceStatus.READY.value
            datasource.document_count = len(documents)
            db.commit()
            print(f"Successfully initialized vector store for datasource {datasource_id}")
            return True
        else:
            datasource.status = DatasourceStatus.ERROR.value
            db.commit()
            print(f"Failed to create vector store for datasource {datasource_id}")
            return False

    finally:
        db.close()


def main():
    """Main entry point."""
    # Initialize database
    print("Initializing database...")
    init_db()

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable options:")
        print("  --sample     Create sample datasource with demo documents")
        print("  <id>         Reinitialize vector store for datasource with given ID")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--sample":
        create_sample_datasource()
    else:
        try:
            datasource_id = int(arg)
            init_datasource(datasource_id)
        except ValueError:
            print(f"Invalid argument: {arg}")
            print(__doc__)
            sys.exit(1)


if __name__ == "__main__":
    main()
