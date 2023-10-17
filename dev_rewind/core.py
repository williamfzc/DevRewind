import typing

import git
from git import Commit, Repo
from langchain.schema import Document
from pydantic import BaseModel
from loguru import logger

from dev_rewind.context import RuntimeContext, FileContext
from dev_rewind.creator import Creator
from dev_rewind.exc import DevRewindException


class DevRewindConfig(BaseModel):
    repo_root: str = "."
    max_depth_limit: int = -1


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

    def collect_documents(self) -> typing.List[Document]:
        return self.collect_metadata().documents

    def _check_env(self) -> typing.Optional[BaseException]:
        try:
            _ = git.Repo(self.config.repo_root)
        except BaseException as e:
            return e
        return None

    def _collect_files(self, ctx: RuntimeContext):
        """ collect all files which tracked by git """
        git_repo = git.Repo(self.config.repo_root)
        git_track_files = set([each[1].path for each in git_repo.index.iter_blobs()])
        for each in git_track_files:
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

    def _create_docs(self, ctx: RuntimeContext):
        creator = Creator()
        for each_file_ctx in ctx.files.values():
            for each_commit in each_file_ctx.commits:
                doc = creator.create_doc(each_file_ctx, each_commit)
                ctx.documents.append(doc)
