from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import VectorStore
from langchain.text_splitter import TokenTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.callbacks import get_openai_callback

from typing import Optional, Union, Any, List
from pathlib import Path
import logging

import fitz
import config

LOGGER = logging.getLogger(__name__)


class ChatPDF:
    """PDF Document AI Assistant"""

    def __init__(self, open_ai_key) -> None:
        self.open_ai_key = open_ai_key
        self.history = []
        self.llm = OpenAI(
            openai_api_key=open_ai_key,
            model_name=config.text_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=open_ai_key, query_model_name=config.embedding_model
        )
        self.docstore = None

    def process_document(self, file):
        """Process the document to allow ChatPDF to search any particular information in it.
        Only plain text PDF for the moment.
        Args:
            file (Union[Path, Any]): file to process
        """
        print(file)
        doc = fitz.open(file)
        text = [page.get_text() for page in doc]
        text = " ".join(text)
        text_splitter = TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=0,
        )
        texts = text_splitter.split_text(text)
        LOGGER.info(f"Number of chunks: {len(texts)}")
        with get_openai_callback() as cb:
            embeddings = OpenAIEmbeddings(openai_api_key=self.open_ai_key)
            LOGGER.info(f"Number of tokens used for embeddings: {cb.total_tokens}")
            self.docsearch = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
            )

    def answer(self, question: str) -> str:
        """From a question relative to a document, generate the answer.
        Args:
            question (str): Question asked by the user.
        Returns:
            str: Answer generated with the LLM
        """
        with get_openai_callback() as cb:
            chain = VectorDBQA.from_chain_type(
                llm=self.llm,
                chain_type=config.chain_type,
                vectorstore=self.docsearch,
                verbose=config.verbose,
            )
            self.sources = self.get_sources(
                query=question
            )  #  TODO: Already considered in VectorDBQAWithSourcesChain but impossible to retrieve
            answer = chain.run(question)
            LOGGER.info(f"Number of tokens used for answering: {cb.total_tokens}")
        return answer

    def get_sources(self, query: str, k: int = config.k) -> List[str]:
        """Retrieve chunks of text similar to the question
        Args:
            query (str): question asked by the user
            k (int, optional): Number of chunks returned. Defaults to config.k.
        Returns:
            List[str]: k chunks the most similar to the question.
        """
        docs = self.docsearch.similarity_search(query=query, k=k)
        return [doc.page_content for doc in docs]
