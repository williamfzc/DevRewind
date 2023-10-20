import typing

from git import Commit
from langchain.schema import Document


class RuntimeContext(object):
    def __init__(self):
        self.files: typing.Dict[str, FileContext] = dict()

        # documents consist of 3 parts:
        # - authors
        # - commits
        # - files
        self.author_documents: typing.List[Document] = []
        self.commit_documents: typing.List[Document] = []
        self.file_documents: typing.List[Document] = []


class FileContext(object):
    def __init__(self, name: str):
        self.name: str = name
        self.commits: typing.List[Commit] = []
