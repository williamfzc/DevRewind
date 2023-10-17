import pytest
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

from dev_rewind import DevRewind, DevRewindConfig


@pytest.fixture
def doc_fixture():
    config = DevRewindConfig()
    config.repo_root = ".."
    api = DevRewind(config)
    return api.collect_documents()


def test_document(doc_fixture):
    assert doc_fixture


def test_embed(doc_fixture):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(doc_fixture, embeddings)
    print("There are", db._collection.count(), "in the collection")

    query = "git"
    matching_docs = db.similarity_search(query)

    for each in matching_docs:
        print(f"Matching doc: {each}")
