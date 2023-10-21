import os
from enum import Enum

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.chroma import Chroma
from loguru import logger

from dev_rewind.core.collector import CollectorLayer
from dev_rewind.core.context import RuntimeContext
from dev_rewind.exc import DevRewindException


class RetrieverType(str, Enum):
    FILE = "FILE"
    AUTHOR = "AUTHOR"
    COMMIT = "COMMIT"


class RetrieverLayer(CollectorLayer):
    def create_ensemble_retriever(
        self,
        embeddings: Embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        ),
        ctx: RuntimeContext = None,
        retriever_kwargs: dict = None,
    ) -> BaseRetriever:
        if not ctx:
            ctx = self.collect_metadata()
        if not retriever_kwargs:
            # default kwargs
            retriever_kwargs = {"k": 4, "include_metadata": True}

        # supress warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        file_retriever = self.create_single_retriever(
            RetrieverType.FILE, embeddings, ctx, retriever_kwargs
        )
        commit_retriever = self.create_single_retriever(
            RetrieverType.COMMIT, embeddings, ctx, retriever_kwargs
        )
        author_retriever = self.create_single_retriever(
            RetrieverType.AUTHOR, embeddings, ctx, retriever_kwargs
        )

        final_retriever = EnsembleRetriever(
            retrievers=[file_retriever, commit_retriever, author_retriever]
        )
        logger.debug("retriever created")
        return final_retriever

    def create_single_retriever(
        self,
        retriever_type: str,
        embeddings: Embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        ),
        ctx: RuntimeContext = None,
        retriever_kwargs: dict = None,
    ) -> BaseRetriever:
        if not ctx:
            ctx = self.collect_metadata()
        if not retriever_kwargs:
            # default kwargs
            retriever_kwargs = {"k": 4, "include_metadata": True}

        # supress warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if retriever_type == RetrieverType.FILE:
            db = Chroma.from_documents(
                ctx.file_documents, embeddings, collection_name="file_db"
            )
        elif retriever_type == RetrieverType.COMMIT:
            db = Chroma.from_documents(
                ctx.commit_documents, embeddings, collection_name="commit_db"
            )
        elif retriever_type == RetrieverType.AUTHOR:
            db = Chroma.from_documents(
                ctx.author_documents, embeddings, collection_name="author_db"
            )
        else:
            # should not happen
            raise DevRewindException(f"invalid retriever type: {retriever_type}")

        return db.as_retriever(**retriever_kwargs)
