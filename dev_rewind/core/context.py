import typing

from git import Commit
from langchain.schema import Document


class RuntimeContext(object):
    def __init__(self):
        self.files: typing.Dict[str, FileContext] = dict()


class FileContext(object):
    def __init__(self, name: str):
        self.name: str = name
        self.commits: typing.List[Commit] = []
