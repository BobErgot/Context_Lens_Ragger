from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from models.models import DocumentMetadata
from constants import RECORD_MANAGER_DB_URL

Base = declarative_base()


engine = create_engine(RECORD_MANAGER_DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_document(db: Session, document: DocumentMetadata):
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
    try:
        return db.query(DocumentMetadata).all()
    except SQLAlchemyError as e:
        print(f"Error occurred while fetching documents: {e}")
        raise


def delete_document_by_id(db: Session, document_id: str):
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
