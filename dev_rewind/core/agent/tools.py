from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel
from rapidfuzz import fuzz

from dev_rewind.config import DevRewindConfig
from dev_rewind.core.context import RuntimeContext


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
Return keywords of a file for understanding file content better.
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
                logger.warning(f"{file_name} not in {runtime_context.files}")
                possible_file_name, score = self.find_best_match(
                    file_name, list(runtime_context.files.keys())
                )
                if score < 80.0:
                    msg = f"all the files did not match {file_name}"
                    logger.warning(msg)
                    return KeywordResponse(ok=False, msg=msg)
                file_name = possible_file_name

            # start a chain for summarizing
            prompt_template = f"""
You are a codebase analyzer.
The following commits comes from git blame of a file:

"{{text}}"

Generate <= {config.keyword_limit} keywords to represent/predict the features of this file.
You should only keep the words related to features/business, just ignore something like git operators or verb.
Response should be a json list.
Keywords:"""
            prompt = PromptTemplate.from_template(prompt_template)

            # Define LLM chain
            llm_chain = LLMChain(llm=custom_llm, prompt=prompt, verbose=True)

            # Define StuffDocumentsChain
            stuff_chain = StuffDocumentsChain(
                llm_chain=llm_chain, document_variable_name="text", verbose=True
            )

            file_context = runtime_context.files[file_name]

            llm_resp = stuff_chain.run(
                [Document(page_content=each.message) for each in file_context.commits]
            )
            return KeywordResponse(
                ok=True,
                msg=f"the real file name should be replaced with: {file_name}",
                data=llm_resp,
            )

    return KeywordTool()
