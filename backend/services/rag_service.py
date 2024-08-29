from operator import itemgetter
from typing import Dict, List, Optional, Sequence
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain.indexes import SQLRecordManager
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)

from constants import (
    COLLECTION_NAME,
    RECORD_MANAGER_DB_URL
)

from document_store.chroma_store import ChromaDocumentStore

def get_embeddings_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")


record_manager = SQLRecordManager(
    f"chroma/{COLLECTION_NAME}", db_url=RECORD_MANAGER_DB_URL
)

RESPONSE_TEMPLATE = """\
You are an expert in document analysis and information retrieval, tasked with answering \
questions based on the contents of the uploaded documents.

Generate a concise and informative answer of 80 words or less for the \
given question based solely on the provided document content. \
You must only use information from the document content. \
Use an unbiased and journalistic tone. \
Combine relevant sections of the document together into a coherent answer. \
Do not repeat text. \
Cite the document sections using [Section Number] notation or other relevant metadata provided. \
Place these citations at the end of the sentence or paragraph that references them - do not put them all at the end. \
If different sections refer to different entities or concepts, write separate answers for each entity or concept. \

You should use bullet points in your answer for readability. Put citations where they apply 
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the document content, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from the uploaded documents, not part of the conversation with the \
user.\
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def get_retriever() -> BaseRetriever:
    embedding = get_embeddings_model()

    document_store = ChromaDocumentStore(
        collection_name=COLLECTION_NAME,
        embedding_model=embedding,
        index_path="chroma_index",
    )

    retriever = document_store.get_retriever(search_kwargs=dict(k=6))

    return retriever


def create_retriever_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
            CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
                RunnableLambda(itemgetter("question")).with_config(
                    run_name="Itemgetter:question"
                )
                | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    default_response_synthesizer = prompt | llm

    response_synthesizer = (
            default_response_synthesizer
            | StrOutputParser()
    ).with_config(run_name="GenerateResponse")

    return (
            RunnablePassthrough.assign(chat_history=serialize_history)
            | context
            | response_synthesizer
    )


llm = Ollama(model="mistral")

retriever = get_retriever()
answer_chain = create_chain(llm, retriever)