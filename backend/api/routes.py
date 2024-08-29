import aiofiles
import os
from models.models import SendFeedbackBody, UpdateFeedbackBody, GetTraceBody
from langsmith import Client
import asyncio
import langsmith
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.ingest_service import DocumentProcessorService
from constants import COLLECTION_NAME
from langchain_community.embeddings import OllamaEmbeddings
from pathlib import Path
from backend.constants import COLLECTION_NAME
import uuid
from fastapi import Depends
from sqlalchemy.orm import Session
from models.models import DocumentMetadata
from services.database import get_db, create_document, get_documents, delete_document_by_id

from backend.services.ingest_service import get_embeddings_model

client = Client()

router = APIRouter()

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


async def _arun(func, *args, **kwargs):
    """
    Run a synchronous function in a separate thread, returning the result asynchronously.
    """
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


@router.post("/upload-document/")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a document, process it to extract embeddings, and store it in the database.

    :param file: The uploaded file.
    :param db: The database session dependency.
    :return: The result of the document processing.
    :raises HTTPException: If the document processing fails.
    """
    try:
        file_location = DATA_DIR / file.filename
        async with aiofiles.open(file_location, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        document_processor_service = DocumentProcessorService(
            collection_name=COLLECTION_NAME,
            embedding_model=embedding_model
        )

        result = await _arun(document_processor_service.ingest_document, str(file_location))

        document_metadata = DocumentMetadata(
            id=uuid.uuid4(),
            filename=file.filename,
            file_location=str(file_location),
            metadata=result,
            embedding_ids=result.get("embedding_ids"),
        )
        create_document(db, document_metadata)

        os.remove(file_location)

        return result
    except Exception as e:
        print(f"Error occurred while uploading and processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload and process document: {str(e)}")


@router.get("/documents/")
def fetch_documents(db: Session = Depends(get_db)):
    """
    Fetch all documents from the database.

    :param db: The database session dependency.
    :return: A list of documents.
    :raises HTTPException: If there is an error fetching the documents.
    """
    try:
        documents = get_documents(db)
        return {"status": "success", "documents": documents}
    except Exception as e:
        print(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    """
    Delete a document and its associated embeddings from the database.

    :param document_id: The UUID of the document to delete.
    :param db: The database session dependency.
    :return: A success message upon deletion.
    :raises HTTPException: If the document is not found or if deletion fails.
    """
    try:
        document = delete_document_by_id(db, document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document.embedding_ids:
            document_processor_service = DocumentProcessorService(
                collection_name=COLLECTION_NAME,
                embedding_model=get_embeddings_model()
            )
            document_processor_service.document_store.remove_embeddings_by_ids(document.embedding_ids)

        return {"status": "success", "message": "Document and embeddings deleted successfully"}
    except Exception as e:
        print(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/upload-directory/")
async def upload_directory(directory_path: str):
    """
    Process and upload all documents from a specified directory.

    :param directory_path: The path to the directory containing documents to upload.
    :return: The result of the document processing.
    :raises HTTPException: If the directory processing fails.
    """
    try:
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        document_processor_service = DocumentProcessorService(
            collection_name=COLLECTION_NAME,
            embedding_model=embedding_model
        )

        result = document_processor_service.ingest_documents_from_directory(directory_path)

        return {"status": "success", "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process directory: {str(e)}")


@router.post("/feedback")
async def send_feedback(body: SendFeedbackBody):
    """
    Submit feedback for a specific run.

    :param body: The feedback details including run ID, key, score, and comment.
    :return: A success message upon feedback submission.
    """
    client.create_feedback(
        body.run_id,
        body.key,
        score=body.score,
        comment=body.comment,
        feedback_id=body.feedback_id,
    )
    return {"result": "posted feedback successfully", "code": 200}


@router.patch("/feedback")
async def update_feedback(body: UpdateFeedbackBody):
    """
    Update existing feedback with new details.

    :param body: The updated feedback details including feedback ID, score, and comment.
    :return: A success message upon feedback update.
    """
    feedback_id = body.feedback_id
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=body.score,
        comment=body.comment,
    )
    return {"result": "patched feedback successfully", "code": 200}


async def aget_trace_url(run_id: str) -> str:
    """
    Retrieve or share a trace URL for a specific run ID.

    :param run_id: The run ID to retrieve the trace URL for.
    :return: The shared trace URL for the run.
    """
    for i in range(5):
        try:
            await _arun(client.read_run, run_id)
            break
        except langsmith.utils.LangSmithError:
            await asyncio.sleep(1 ** i)

    if await _arun(client.run_is_shared, run_id):
        return await _arun(client.read_run_shared_link, run_id)
    return await _arun(client.share_run, run_id)


@router.post("/get_trace")
async def get_trace(body: GetTraceBody):
    """
    Retrieve the trace URL for a specific LangSmith run ID.

    :param body: The body containing the run ID.
    :return: The trace URL or an error message if the run ID is not provided.
    """
    run_id = body.run_id
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(str(run_id))