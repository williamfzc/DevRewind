from tinydb import TinyDB


def create_cache_db(path: str) -> TinyDB:
    return TinyDB(path)
