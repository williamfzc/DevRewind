import os

import pytest
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma

from dev_rewind import DevRewind, DevRewindConfig
from dev_rewind.core.workflow.agent import create_agent


@pytest.fixture
def meta_fixture():
    config = DevRewindConfig()
    config.repo_root = os.path.dirname(__file__)
    api = DevRewind(config)
    return api.collect_metadata()


def test_document(meta_fixture):
    assert meta_fixture.file_documents
    assert meta_fixture.commit_documents
    assert meta_fixture.author_documents


def test_embed(meta_fixture):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(meta_fixture.commit_documents, embeddings)
    print("There are", db._collection.count(), "in the collection")
    print(db._collection.metadata)

    query = "git"
    matching_docs = db.similarity_search(query)

    for each in matching_docs:
        print(f"Matching doc: {each}")


def test_agent():
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("no openapi key for test")

    agent = create_agent()
    # hacky way from https://github.com/langchain-ai/langchain/issues/1358#issuecomment-1486132587
    try:
        response = agent.run(input="chain.py 的功能是什么？")
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse LLM output:"):
            raise e
        response = response.removeprefix("Could not parse LLM output:").removesuffix(
            "`"
        )
    print(response)


def test_with_openai(meta_fixture):
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("no openapi key for test")

    api = DevRewind()
    retriever = api.create_ensemble_retriever(ctx=meta_fixture)
    chain = api.create_stuff_chain(retriever=retriever)

    # author
    output = chain.run(
        """
How many commits authored by williamfzc? 
"""
    )
    print(f"output: {output}")

    # file
    output = chain.run(
        """
How many files in this repo?
    """
    )
    print(f"output: {output}")
