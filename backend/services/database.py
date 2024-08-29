from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError
from models.models import DocumentMetadata
from constants import RECORD_MANAGER_DB_URL

# Base class for the SQLAlchemy ORM models
Base = declarative_base()

# Database engine and session configuration
engine = create_engine(RECORD_MANAGER_DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """
    Creates all the tables in the database that are defined by subclasses of Base.
    This function should be called at the start of the application to ensure the
    database schema is up-to-date.
    """
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Dependency function that provides a database session for use in FastAPI routes.
    It ensures that the session is properly opened and closed, even if an exception occurs.

    Yields:
        db (Session): A SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_document(db: Session, document: DocumentMetadata):
    """
    Adds a new document record to the database.

    Args:
        db (Session): The database session to use for the operation.
        document (DocumentMetadata): The document metadata to be added.

    Returns:
        DocumentMetadata: The document metadata that was added to the database.

    Raises:
        SQLAlchemyError: If there is an error during the operation, the transaction
                         is rolled back and the error is printed.
    """
    try:
        db.add(document)
        db.commit()
        db.refresh(document)
        return document
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error occurred during document creation: {e}")
        raise


def get_documents(db: Session):
    """
    Fetches all document records from the database.

    Args:
        db (Session): The database session to use for the operation.

    Returns:
        List[DocumentMetadata]: A list of all document metadata records in the database.

    Raises:
        SQLAlchemyError: If there is an error during the operation, the error is printed.
    """
    try:
        return db.query(DocumentMetadata).all()
    except SQLAlchemyError as e:
        print(f"Error occurred while fetching documents: {e}")
        raise


def delete_document_by_id(db: Session, document_id: str):
    """
    Deletes a document record from the database by its ID.

    Args:
        db (Session): The database session to use for the operation.
        document_id (str): The ID of the document to be deleted.

    Returns:
        DocumentMetadata: The document metadata that was deleted from the database, or None if not found.

    Raises:
        SQLAlchemyError: If there is an error during the operation, the transaction
                         is rolled back and the error is printed.
    """
    try:
        document = db.query(DocumentMetadata).filter_by(id=document_id).first()
        if document:
            db.delete(document)
            db.commit()
            return document
        return None
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error occurred while deleting document: {e}")
        raise
