import uuid
from typing import Union, Optional
from pydantic import BaseModel
from uuid import UUID
from sqlalchemy import Column, String, JSON, Text, func, DateTime, ARRAY
from sqlalchemy.dialects.postgresql import UUID as SQLAlchemyUUID
from sqlalchemy.orm import declarative_base

# Pydantic Models
class SendFeedbackBody(BaseModel):
    """
    Pydantic model to represent the body of a feedback submission request.

    Attributes:
        run_id (UUID): The ID of the run for which the feedback is being submitted.
        key (str): The feedback key, defaulting to "user_score".
        score (Union[float, int, bool, None]): The score of the feedback, which can be a float, integer, boolean, or None.
        feedback_id (Optional[UUID]): An optional UUID to identify the feedback if it needs to be updated later.
        comment (Optional[str]): An optional comment associated with the feedback.
    """
    run_id: UUID
    key: str = "user_score"
    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


class UpdateFeedbackBody(BaseModel):
    """
    Pydantic model to represent the body of a feedback update request.

    Attributes:
        feedback_id (UUID): The ID of the feedback to be updated.
        score (Union[float, int, bool, None]): The updated score for the feedback.
        comment (Optional[str]): An optional updated comment for the feedback.
    """
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None


class GetTraceBody(BaseModel):
    """
    Pydantic model to represent the body of a request to retrieve a trace URL.

    Attributes:
        run_id (UUID): The ID of the run for which the trace URL is being requested.
    """
    run_id: UUID


# SQLAlchemy Models
Base = declarative_base()

class DocumentMetadata(Base):
    """
    SQLAlchemy model representing the metadata of a document in the database.

    Attributes:
        id (UUID): The unique identifier of the document metadata record.
        filename (str): The name of the uploaded file.
        user_id (UUID): The optional ID of the user who uploaded the document.
        file_location (str): The file path location where the document is stored.
        document_metadata (str): Additional metadata associated with the document, stored as a text field.
        embedding_ids (List[str]): A list of embedding IDs associated with the document.
        created_at (datetime): The timestamp when the record was created, defaulting to the current time.
    """
    __tablename__ = "document_metadata"

    id = Column(SQLAlchemyUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    user_id = Column(SQLAlchemyUUID(as_uuid=True), nullable=True)
    file_location = Column(String, nullable=False)
    document_metadata = Column(Text, nullable=True)
    embedding_ids = Column(ARRAY(String), nullable=True)
    created_at = Column(DateTime, server_default=func.now())