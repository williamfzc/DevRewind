import typing

from git import Commit
from langchain.schema import Document
from tinydb import TinyDB

from dev_rewind.core.cache import create_cache_db


class RuntimeContext(object):
    def __init__(self, cache_path: str):
        self.files: typing.Dict[str, FileContext] = dict()
        self.cache: TinyDB = create_cache_db(cache_path)


class FileContext(object):
    def __init__(self, name: str):
        self.name: str = name
        self.commits: typing.List[Commit] = []
