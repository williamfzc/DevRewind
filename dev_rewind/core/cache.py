import typing

from pydantic import BaseModel
from tinydb import TinyDB, Query
import pathlib
import chromadb


class CacheFileUnit(BaseModel):
    name: str
    keywords: typing.List[str]


class Cache(object):
    TINY_DB_FILE = "tinydb.json"
    CHROMA_DB_FILE = "chromadb"

    def __init__(self, cache_dir: str):
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        self.tinydb = TinyDB(cache_dir / self.TINY_DB_FILE)
        self.chromadb = chromadb.PersistentClient(
            path=(cache_dir / self.CHROMA_DB_FILE).as_posix()
        )

    def create(self, file_name: str, words: typing.List[str]):
        self.tinydb.insert(CacheFileUnit(name=file_name, keywords=words).model_dump())

    def read(self, file_name: str) -> typing.List[str]:
        file_query = Query()
        query_resp = self.tinydb.search(file_query.file_name == file_name)
        if not query_resp:
            return []
        return query_resp[0]["keywords"]


def create_cache_db(cache_dir: str) -> Cache:
    return Cache(cache_dir)
