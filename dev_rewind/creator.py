import typing

from git import Commit
from langchain.schema import Document
from pydantic import BaseModel

from dev_rewind.context import RuntimeContext


class _Author(BaseModel):
    name: str
    email: str


class _File(BaseModel):
    name: str
    blames: typing.List[str] = []


class Creator(object):
    """
    creator was designed for converting commits to documents
    """

    def create_doc(self, ctx: RuntimeContext):
        author_dict: typing.Dict[str, _Author] = dict()
        commit_dict: typing.Dict[str, Commit] = dict()
        file_dict: typing.Dict[str, _File] = dict()

        # fill data
        for each_file in ctx.files.values():
            file_dict[each_file.name] = _File(name=each_file.name)
            for each_commit in each_file.commits:
                commit_dict[each_commit.hexsha] = each_commit
                author_dict[each_commit.author.email] = _Author(name=each_commit.author.name,
                                                                email=each_commit.author.email)
                file_dict[each_file.name].blames.append(each_commit.hexsha)

        # author doc
        for each_author in author_dict.values():
            page_content = f"""
[Author Info]
author: {each_author.name}
email: {each_author.email}
"""
            metadata = {
                "source": each_author.email
            }
            ctx.author_documents.append(Document(page_content=page_content, metadata=metadata))

        # commit doc
        for each_commit in commit_dict.values():
            # which should be searched by vectorstore
            page_content = f"""
[Commit Info]
msg: {each_commit.message.strip()}
author: {each_commit.author.name} <{each_commit.author.email}>
time: {each_commit.authored_datetime}
"""
            # which only should be used in LLM part
            metadata = {
                "sha": each_commit.hexsha,
                "files": str(list(each_commit.stats.files.keys())),
                "source": each_commit.hexsha,
            }

            ctx.commit_documents.append(Document(page_content=page_content, metadata=metadata))

        # file doc
        for each_file in file_dict.values():
            page_content = f"""
[File Info]
path: {each_file.name}
related commits: {", ".join(each_file.blames)}
"""
            metadata = {
                "source": each_file.name
            }

            ctx.file_documents.append(Document(page_content=page_content, metadata=metadata))
