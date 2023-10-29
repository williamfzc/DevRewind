import pathlib
import typing

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pydantic import BaseModel
from tinydb import TinyDB, Query


class CacheFileUnit(BaseModel):
    name: str
    keywords: typing.List[str]


class Cache(object):
    TINY_DB_FILE = "tinydb.json"

    CHROMA_DB_FILE = "chromadb"
    CHROMA_DB_COLLECTION = "keyword"

    def __init__(self, cache_dir: str):
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        self.tinydb = TinyDB(cache_dir / self.TINY_DB_FILE)
        self.chromadb = chromadb.PersistentClient(
            path=(cache_dir / self.CHROMA_DB_FILE).as_posix()
        )
        self.chromadb_collection = self.chromadb.get_or_create_collection(
            self.CHROMA_DB_COLLECTION,
            embedding_function=SentenceTransformerEmbeddingFunction(),
        )

    def create(self, file_name: str, words: typing.List[str]):
        q = Query()
        self.tinydb.upsert(
            CacheFileUnit(name=file_name, keywords=words).model_dump(),
            q.name == file_name,
        )

        # for search
        self.chromadb_collection.add(
            documents=[str(words)], metadatas=[{"source": file_name}], ids=[file_name]
        )

    def read(self, file_name: str) -> typing.List[str]:
        file_query = Query()
        query_resp = self.tinydb.search(file_query.name == file_name)
        if not query_resp:
            return []
        return query_resp[0]["keywords"]

    def query_files(self, words: typing.List[str]) -> typing.List[str]:
        query_result = self.chromadb_collection.query(query_texts=[str(words)])
        return [str(each) for each in query_result["ids"]]


def create_cache_db(cache_dir: str) -> Cache:
    return Cache(cache_dir)
