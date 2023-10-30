import typing

from langchain.llms import BaseLLM
from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel
from rapidfuzz import fuzz

from dev_rewind.config import DevRewindConfig
from dev_rewind.core.agent.summary import get_summary_engine
from dev_rewind.core.context import RuntimeContext


class ToolResponse(BaseModel):
    ok: bool
    msg: str
    data: str = ""


def create_keyword_tool(
    config: DevRewindConfig, runtime_context: RuntimeContext, custom_llm: BaseLLM
) -> BaseTool:
    class KeywordTool(BaseTool):
        name = "get_keywords_by_file"
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

        def _run(self, file_name: str) -> ToolResponse:
            if file_name not in runtime_context.files:
                possible_file_name, score = self.find_best_match(
                    file_name, list(runtime_context.files.keys())
                )
                if score < 80.0:
                    msg = f"all the files did not match {file_name}"
                    logger.warning(msg)
                    return ToolResponse(ok=False, msg=msg)
                logger.warning(
                    f"{file_name} seems not a valid git tracked file. Try the most possible one: {possible_file_name}"
                )
                file_name = possible_file_name

            # check the cache
            cached_keywords = runtime_context.cache.read(file_name)
            if cached_keywords:
                logger.debug(f"found {file_name} 's cache")
                return ToolResponse(
                    ok=True,
                    msg=f"the real file name should be: {file_name}",
                    data=str(cached_keywords),
                )

            # start a chain for summarizing
            keyword_engine = get_summary_engine(self.config.keyword_algo)
            keywords = keyword_engine.summarize_commits(
                runtime_context.files[file_name].commits,
                config.keyword_limit,
                custom_llm,
            )

            runtime_context.cache.create(file_name, keywords)

            return ToolResponse(
                ok=True,
                msg=f"the real file name should be: {file_name}",
                data=str(keywords),
            )

    return KeywordTool()


def create_file_tool(runtime_context: RuntimeContext) -> BaseTool:
    class SearchTool(BaseTool):
        name = "get_files_by_keyword"
        description = """
Receive some keywords as input, and return a file list which related to it. 

It will return:
- ok: bool, False if no files found
- msg: extra context for you
- data: file list
"""
        verbose = True

        def _run(self, keywords: typing.List[str]) -> ToolResponse:
            files = runtime_context.cache.query_files(keywords)
            return ToolResponse(
                ok=True,
                msg="",
                data=str(files),
            )

    return SearchTool()
