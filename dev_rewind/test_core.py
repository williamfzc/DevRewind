import os

import pytest
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
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


def test_with_openai(doc_fixture):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    db = Chroma.from_documents(doc_fixture, embeddings)

    prompt_template = """
Use the following pieces of commits to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
    global_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    doc_prompt = PromptTemplate(
        input_variables=["page_content", "file", "author"],
        template="commit msg: {page_content}, file: {file}, author: {author}",
    )

    chain_type_kwargs = {
        "prompt": global_prompt,
        "document_prompt": doc_prompt,
    }

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )

    # author
    output = qa.run("""
How many commits authored by williamfzc? 
""")
    print(f"output: {output}")

    # file
    output = qa.run("""
How many files in this repo?
    """)
    print(f"output: {output}")

