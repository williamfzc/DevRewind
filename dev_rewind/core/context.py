import typing

from git import Commit

from dev_rewind.core.cache import create_cache_db, Cache


class RuntimeContext(object):
    def __init__(self, cache_path: str):
        self.files: typing.Dict[str, FileContext] = dict()
        self.cache: Cache = create_cache_db(cache_path)


class FileContext(object):
    def __init__(self, name: str):
        self.name: str = name
        self.commits: typing.List[Commit] = []
