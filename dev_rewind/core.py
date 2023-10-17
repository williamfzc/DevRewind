import typing

import git
from git import Commit, Repo
from pydantic import BaseModel
from loguru import logger

from dev_rewind.context import RuntimeContext
from dev_rewind.exc import DevRewindException


class DevRewindConfig(BaseModel):
    repo_root: str = "."
    max_depth_limit: int = -1


class DevRewind(object):
    def __init__(self, config: DevRewindConfig = None):
        if not config:
            config = DevRewindConfig()
        self.config = config

    def rewind(self):
        exc = self._check_env()
        if exc:
            raise DevRewindException() from exc

        logger.info("start rewinding ...")
        logger.debug("git metadata collecting ...")
        ctx = RuntimeContext()
        self._collect_files(ctx)
        self._collect_histories(ctx)
        logger.debug("building documentations ...")

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
            ctx.files[each] = []
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

        for each_file in ctx.files:
            commits = self._collect_history(git_repo, each_file)
            ctx.files[each_file] = commits
