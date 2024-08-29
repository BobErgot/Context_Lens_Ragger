import argparse
import functools
from operator import itemgetter
from langsmith.evaluation.evaluator import RunEvaluator, EvaluationResult
from typing import Optional
from langchain_core.language_models import LanguageModelLike
from langchain_core.schema.runnable import Runnable
from langchain_core.smith import RunEvalConfig
from langsmith import Client
from langsmith.schemas import Example, Run
from langsmith.evaluation.evaluator import EvaluationResult

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain.schema.runnable import RunnableMap

from constants import COLLECTION_NAME, RECORD_MANAGER_DB_URL
from document_store.chroma_store import ChromaDocumentStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

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

def get_embeddings_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")


def get_retriever() -> BaseRetriever:
    embedding = get_embeddings_model()

    document_store = ChromaDocumentStore(
        collection_name=COLLECTION_NAME,
        embedding_model=embedding,
        index_path="chroma_index",
    )

    retriever = document_store.get_retriever(search_kwargs=dict(k=6))

    return retriever


def create_chain(
        retriever: BaseRetriever,
        llm: LanguageModelLike,
        chat_history: Optional[list] = None
) -> Runnable:
    rephrase_template = """\
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow-Up Input: {question}
    Standalone Question:"""

    CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(rephrase_template)

    if chat_history:
        _inputs = RunnableMap(
            {
                "standalone_question": {
                                           "question": lambda x: x["question"],
                                           "chat_history": lambda x: x["chat_history"],
                                       }
                                       | CONDENSE_QUESTION_PROMPT
                                       | llm
                                       | StrOutputParser(),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
        )
        _context = {
            "context": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESPONSE_TEMPLATE),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
    else:
        _inputs = RunnableMap(
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: [],
            }
        )
        _context = {
            "context": itemgetter("question") | retriever,
            "question": lambda x: x["question"],
            "chat_history": lambda x: [],
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RESPONSE_TEMPLATE),
                ("human", "{question}"),
            ]
        )

    chain = _inputs | _context | prompt | llm | StrOutputParser()

    return chain


class CustomHallucinationEvaluator(RunEvaluator):
    @staticmethod
    def _get_llm_runs(run: Run) -> list:
        llm_runs = []
        for child in run.child_runs or []:
            if child.run_type == "llm":
                llm_runs.append(child)
            else:
                llm_runs.extend(CustomHallucinationEvaluator._get_llm_runs(child))
        return llm_runs

    def evaluate_run(
            self, run: Run, example: Example | None = None
    ) -> EvaluationResult:
        llm_runs = self._get_llm_runs(run)
        if not llm_runs:
            return EvaluationResult(key="hallucination", comment="No LLM runs found")
        if len(llm_runs) > 1:
            return EvaluationResult(
                key="hallucination", comment="Too many LLM runs found"
            )
        llm_run = llm_runs[0]
        messages = llm_run.inputs["messages"]
        return EvaluationResult(key="hallucination", comment="Processed messages", value=str(messages))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Chat QnA Complex Questions")
    parser.add_argument("--model-provider", default="ollama")
    parser.add_argument("--model-name", default="mistral")
    args = parser.parse_args()

    client = Client()

    # Check dataset exists
    ds = client.read_dataset(dataset_name=args.dataset_name)
    retriever = get_retriever()

    llm = Ollama(model=args.model_name)

    constructor = functools.partial(
        create_chain,
        retriever=retriever,
        llm=llm,
    )
    chain = constructor()

    eval_config = RunEvalConfig(evaluators=["qa"], prediction_key="output")
    results = client.run_on_dataset(
        dataset_name=args.dataset_name,
        llm_or_chain_factory=constructor,
        evaluation=eval_config,
        tags=["evaluation_chain"],
        verbose=True,
    )
    print(results)
    proj = client.read_project(project_name=results["project_name"])
    print(proj.feedback_stats)