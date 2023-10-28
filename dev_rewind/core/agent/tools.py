import typing

from git import Commit
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel
from rapidfuzz import fuzz
from tinydb import Query

from dev_rewind.config import DevRewindConfig
from dev_rewind.core.context import RuntimeContext


def summarize_commits_with_llm(
    commits: typing.Iterable[Commit], keyword_limit: int, llm: BaseLLM
):
    return summarize_docs_with_llm(
        (Document(page_content=each.message) for each in commits), keyword_limit, llm
    )


def summarize_docs_with_llm(
    docs: typing.Iterable[Document], keyword_limit: int, llm: BaseLLM
):
    prompt_template = f"""
You are a codebase analyzer.
The following commits comes from git blame of a file:

"{{text}}"

Generate <= {keyword_limit} keywords to represent/predict the features of this file.
You should only keep the words related to features/business, just ignore something like git operators or verb.
Response should be a json list.
Keywords:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text", verbose=True
    )

    return stuff_chain.run(docs)


def create_keyword_tool(
    config: DevRewindConfig, runtime_context: RuntimeContext, custom_llm: BaseLLM
) -> BaseTool:
    class KeywordResponse(BaseModel):
        ok: bool
        msg: str
        data: str = ""

    class KeywordTool(BaseTool):
        name = "keyword"
        description = """
Return keywords of a file for helping index/present this file.
Always use this if you need to summarize a file.

It will return:
- ok: bool, file is valid or not
- msg: extra context for you
- data: keywords
"""
        verbose = True

        def find_best_match(self, file_name, file_list) -> (str, float):
            best_match = None
            best_similarity = 0

            for file in file_list:
                similarity = fuzz.partial_ratio(file_name, file)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = file

            return best_match, best_similarity

        def _run(self, file_name: str) -> KeywordResponse:
            if file_name not in runtime_context.files:
                possible_file_name, score = self.find_best_match(
                    file_name, list(runtime_context.files.keys())
                )
                if score < 80.0:
                    msg = f"all the files did not match {file_name}"
                    logger.warning(msg)
                    return KeywordResponse(ok=False, msg=msg)
                logger.warning(
                    f"{file_name} seems not a valid git tracked file. Try the most possible one: {possible_file_name}"
                )
                file_name = possible_file_name

            # check the cache
            file_query = Query()
            cache_resp = runtime_context.cache.search(file_query.file_name == file_name)
            if cache_resp:
                logger.debug(f"found {file_name} 's cache")
                return KeywordResponse(
                    ok=True,
                    msg=f"the real file name should be: {file_name}",
                    data=cache_resp[0]["resp"],
                )

            # start a chain for summarizing
            llm_resp = summarize_commits_with_llm(
                runtime_context.files[file_name].commits,
                config.keyword_limit,
                custom_llm,
            )
            runtime_context.cache.insert(
                {
                    "file_name": file_name,
                    "resp": llm_resp,
                }
            )

            return KeywordResponse(
                ok=True,
                msg=f"the real file name should be: {file_name}",
                data=llm_resp,
            )

    return KeywordTool()
