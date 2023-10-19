import os
import re
import typing

import git
from git import Commit, Repo
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.base import LLM
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.chroma import Chroma
from loguru import logger
from pydantic import BaseModel

from dev_rewind.context import RuntimeContext, FileContext
from dev_rewind.creator import Creator
from dev_rewind.exc import DevRewindException


class DevRewindConfig(BaseModel):
    repo_root: str = "."
    max_depth_limit: int = -1
    include_regex: str = ""


class DevRewind(object):
    def __init__(self, config: DevRewindConfig = None):
        if not config:
            config = DevRewindConfig()
        self.config = config

    def collect_metadata(self) -> RuntimeContext:
        exc = self._check_env()
        if exc:
            raise DevRewindException() from exc

        logger.debug("git metadata collecting ...")
        ctx = RuntimeContext()
        self._collect_files(ctx)
        self._collect_histories(ctx)

        logger.debug("building documentations ...")
        self._create_docs(ctx)
        logger.debug("metadata ready")
        return ctx

    def create_retriever(
            self,
            embeddings: Embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
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

        file_db = Chroma.from_documents(
            ctx.file_documents, embeddings, collection_name="file_db")
        commit_db = Chroma.from_documents(
            ctx.commit_documents, embeddings, collection_name="commit_db")
        author_db = Chroma.from_documents(
            ctx.author_documents, embeddings, collection_name="author_db")
        final_retriever = EnsembleRetriever(
            retrievers=[
                file_db.as_retriever(**retriever_kwargs),
                commit_db.as_retriever(**retriever_kwargs),
                author_db.as_retriever(**retriever_kwargs)])
        logger.debug("retriever created")
        return final_retriever

    def create_stuff_chain(
            self,
            llm: LLM = None,
            retriever: BaseRetriever = None,
            **kwargs
    ) -> Chain:
        if not llm:
            llm = OpenAI()
        if not retriever:
            retriever = self.create_retriever()

        prompt_template = """
You are a codebase analyzer.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
        """
        global_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )
        doc_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}",
        )
        chain_type_kwargs = {
            "prompt": global_prompt,
            "document_prompt": doc_prompt,
            **kwargs,
        }
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
        )
        logger.debug("chain created")
        return chain

    def _check_env(self) -> typing.Optional[BaseException]:
        try:
            repo = git.Repo(self.config.repo_root, search_parent_directories=True)
            # if changed after search
            self.config.repo_root = repo.git_dir
        except BaseException as e:
            return e
        return None

    def _collect_files(self, ctx: RuntimeContext):
        """ collect all files which tracked by git """
        git_repo = git.Repo(self.config.repo_root)
        git_track_files = set([each[1].path for each in git_repo.index.iter_blobs()])

        include_regex = None
        if self.config.include_regex:
            include_regex = re.compile(self.config.include_regex)

        for each in git_track_files:
            if include_regex:
                if not include_regex.match(each):
                    continue

            ctx.files[each] = FileContext(each)

        logger.debug(f"file {len(ctx.files)} collected")

    def _collect_history(self, repo: Repo, file_path: str) -> typing.List[Commit]:
        kwargs = {
            "paths": file_path,
        }
        if self.config.max_depth_limit != -1:
            kwargs["max_count"] = self.config.max_depth_limit

        result = []
        for commit in repo.iter_commits(**kwargs):
            result.append(commit)
        return result

    def _collect_histories(self, ctx: RuntimeContext):
        git_repo = git.Repo(self.config.repo_root)

        for each_file, each_file_ctx in ctx.files.items():
            commits = self._collect_history(git_repo, each_file)
            each_file_ctx.commits = commits
            logger.debug(f"file {each_file} ready")

    def _create_docs(self, ctx: RuntimeContext):
        creator = Creator()
        creator.create_doc(ctx)
