import uuid
from typing import Union
from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from sqlalchemy import Column, String, JSON, Text, func, DateTime, ARRAY
from sqlalchemy.dialects.postgresql import UUID as SQLAlchemyUUID, TSTZRANGE
from sqlalchemy.orm import declarative_base


# Feedback Models
class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


class UpdateFeedbackBody(BaseModel):
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None


# Trace Model
class GetTraceBody(BaseModel):
    run_id: UUID


Base = declarative_base()


class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(SQLAlchemyUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    user_id = Column(SQLAlchemyUUID(as_uuid=True), nullable=True)
    file_location = Column(String, nullable=False)
    document_metadata = Column(Text, nullable=True)
    embedding_ids = Column(ARRAY(String), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
