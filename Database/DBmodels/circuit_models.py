from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from Database.DBConfiguration.database import Base

class Circuit(Base):
    __tablename__ = "circuits"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    created_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_template = Column(Boolean, default=False)

    # Relationships
    project = relationship("Project", back_populates="circuits")
    creator = relationship("User")
    versions = relationship("CircuitVersion", back_populates="circuit", cascade="all, delete-orphan")

class CircuitVersion(Base):
    __tablename__ = "circuit_versions"

    id = Column(Integer, primary_key=True, index=True)
    circuit_id = Column(Integer, ForeignKey("circuits.id"))
    version_number = Column(Integer)
    created_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)
    data = Column(Text)  # JSON or serialized circuit data

    # Relationships
    circuit = relationship("Circuit", back_populates="versions")
    creator = relationship("User")

class CircuitTemplate(Base):
    __tablename__ = "circuit_templates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    created_by_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    data = Column(Text)  # JSON or serialized template data
    is_public = Column(Boolean, default=False)

    # Relationships
    creator = relationship("User")
