from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from Database.DBConfiguration.database import Base

class Notebook(Base):
    __tablename__ = "notebooks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    team_id = Column(Integer, ForeignKey("teams.id"))
    created_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    team = relationship("Team")
    creator = relationship("User")
    cells = relationship("NotebookCell", back_populates="notebook", cascade="all, delete-orphan")
    versions = relationship("NotebookVersion", back_populates="notebook", cascade="all, delete-orphan")

class NotebookCell(Base):
    __tablename__ = "notebook_cells"

    id = Column(Integer, primary_key=True, index=True)
    notebook_id = Column(Integer, ForeignKey("notebooks.id"))
    cell_type = Column(String, default="code")  # 'code' or 'markdown'
    content = Column(Text, default="")
    order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    notebook = relationship("Notebook", back_populates="cells")

class NotebookVersion(Base):
    __tablename__ = "notebook_versions"

    id = Column(Integer, primary_key=True, index=True)
    notebook_id = Column(Integer, ForeignKey("notebooks.id"))
    version_number = Column(Integer, default=1)
    commit_message = Column(String, default="Initial commit")
    created_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    snapshot = Column(JSON)  # JSON snapshot of notebook state (cells, content, metadata)

    # Relationships
    notebook = relationship("Notebook", back_populates="versions")
    creator = relationship("User")
