import io
import logging
import os
from typing import List, Dict, Any
import subprocess
import nltk
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from constants import COLLECTION_NAME
from document_store.chroma_store import ChromaDocumentStore
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredImageLoader
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document, AIMessage, HumanMessage
from backend.constants import OPENAI_API_KEY

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> OllamaEmbeddings:
    """
    Initializes and returns an instance of OllamaEmbeddings for generating embeddings.

    This function sets up the OllamaEmbeddings model with the "nomic-embed-text" configuration,
    which is used to generate vector representations of text data that can be stored and queried
    in a vector store.

    Returns:
        OllamaEmbeddings: An instance of OllamaEmbeddings configured with the "nomic-embed-text" model.
    """
    return OllamaEmbeddings(model="nomic-embed-text")


class DocumentProcessorService:
    """
    A service class for processing and managing documents.

    This class handles document ingestion, processing (including OCR for images and text extraction),
    and embedding generation. It also provides methods for deleting document embeddings from the
    document store.

    Attributes:
        document_store (ChromaDocumentStore): The document store instance for storing and managing embeddings.
        text_splitter (RecursiveCharacterTextSplitter): A text splitter for splitting documents into chunks.
        llm (ChatOpenAI): A language model instance for generating comprehensive descriptions.
        processor (BlipProcessor): A processor for handling image captioning.
        model (BlipForConditionalGeneration): A model for generating image captions.
    """

    def __init__(self, collection_name: str, embedding_model: OllamaEmbeddings, index_path: str = "chroma_index"):
        """
        Initializes the DocumentProcessorService with the provided collection name, embedding model, and index path.

        Args:
            collection_name (str): The name of the document collection.
            embedding_model (OllamaEmbeddings): The embedding model used for generating embeddings.
            index_path (str, optional): The path to the Chroma index. Defaults to "chroma_index".
        """
        self.document_store = ChromaDocumentStore(
            collection_name=collection_name,
            embedding_model=embedding_model,
            index_path=index_path,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
        )
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def process_document(self, file_path: str) -> List[Document]:
        """
        Processes a document based on its file extension and returns a list of Document objects.

        Args:
            file_path (str): The path to the document file.

        Returns:
            List[Document]: A list of processed Document objects.
        """
        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext in [".png", ".jpg", ".jpeg"]:
            loader = UnstructuredImageLoader(file_path)
        elif ext == ".svg":
            return self.process_svg(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        documents = loader.load()
        return documents

    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extracts text from an image using Tesseract OCR.

        Args:
            image (Image.Image): The image from which to extract text.

        Returns:
            str: The extracted text.
        """
        text = pytesseract.image_to_string(image)
        return text.strip()

    def generate_detailed_caption(self, image: Image.Image, extracted_text: str) -> str:
        """
        Generates a detailed caption for the image by combining extracted text and image captioning.

        Args:
            image (Image.Image): The image for which to generate a caption.
            extracted_text (str): The text extracted from the image.

        Returns:
            str: A comprehensive description generated by the language model.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        combined_description = f"Caption: {caption}\nExtracted Text: {extracted_text}"
        prompt = (
            f"Here is a caption and text extracted from an image:\n\n"
            f"Caption: {caption}\n"
            f"Extracted Text: {extracted_text}\n\n"
            f"Based on this information, generate a comprehensive and detailed description of the image."
        )
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        if isinstance(response, AIMessage):
            comprehensive_description = response.content
        else:
            logger.error("Unexpected response type from GPT model")
        return comprehensive_description

    def describe_image(self, image: Image.Image) -> str:
        """
        Extracts text from the image and generates a comprehensive description using a combination of OCR, vision-language model, and GPT.

        Args:
            image (Image.Image): The image to describe.

        Returns:
            str: A comprehensive description of the image.
        """
        extracted_text = self.extract_text_from_image(image)
        detailed_caption = self.generate_detailed_caption(image, extracted_text)
        logger.info(f"Final comprehensive description: {detailed_caption}")
        return detailed_caption

    def process_svg(self, file_path: str) -> List[Document]:
        """
        Converts an SVG file to PNG, extracts text using OCR, generates a detailed description,
        and stores the description as a Document.

        Args:
            file_path (str): The path to the SVG file.

        Returns:
            List[Document]: A list containing a single Document object with the image description.
        """
        try:
            png_data = subprocess.check_output(["rsvg-convert", "-f", "png", file_path])
            image = Image.open(io.BytesIO(png_data))
            description = self.describe_image(image)
            metadata = {"source": file_path, "description_type": "image_caption"}
            document = Document(page_content=description, metadata=metadata)
            return [document]
        except Exception as e:
            logger.error(f"Failed to process SVG: {str(e)}")
            return []

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingests a document by processing it and storing the processed data in the document store.

        Args:
            file_path (str): The path to the document file.

        Returns:
            Dict[str, Any]: A dictionary containing the status, message, and embedding IDs.
        """
        try:
            documents = self.process_document(file_path)
            logger.info(f"Loaded {len(documents)} documents from {file_path}")

            docs_transformed = self.text_splitter.split_documents(documents)
            docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

            embedding_ids = self.document_store.add_documents(
                [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs_transformed]
            )
            self.document_store.save()
            logger.info(f"Successfully indexed documents from {file_path}")

            return {
                "status": "success",
                "message": f"Indexed {len(docs_transformed)} documents.",
                "embedding_ids": embedding_ids,
            }
        except Exception as e:
            logger.error(f"Failed to process document: {str(e)}")
            return {"status": "error", "message": str(e)}

    def delete_document_embeddings(self, embedding_ids: List[str]):
        """
        Removes embeddings from the document store using their IDs.

        Args:
            embedding_ids (List[str]): List of embedding IDs to be removed.
        """
        try:
            self.document_store.remove_embeddings_by_ids(embedding_ids)
            logger.info(f"Successfully removed {len(embedding_ids)} embeddings.")
        except Exception as e:
            logger.error(f"Failed to remove embeddings: {str(e)}")


def load_pdf_from_file(file_path: str) -> List[Document]:
    """
    Loads and processes a PDF file, returning a list of Document objects.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        List[Document]: A list of processed Document objects.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_word_from_file(file_path: str) -> List[Document]:
    """
    Loads and processes a Word document file (.docx), returning a list of Document objects.

    Args:
        file_path (str): The path to the Word document file.

    Returns:
        List[Document]: A list of processed Document objects.
    """
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    return documents


def load_image_from_file(file_path: str) -> List[Document]:
    """
    Loads and processes an image file, returning a list of Document objects.

    Args:
        file_path (str): The path to the image file.

    Returns:
        List[Document]: A list of processed Document objects.
    """
    if file_path.endswith('.svg'):
        return DocumentProcessorService(COLLECTION_NAME, get_embeddings_model()).process_svg(file_path)

    loader = UnstructuredImageLoader(file_path)
    documents = loader.load()
    return documents


def sanitize_filename(filename: str) -> str:
    """
    Sanitize the filename by removing any leading directory paths and restricting to a safe set of characters.
    """
    return os.path.basename(filename).replace("/", "_").replace("\\", "_")
