import os
import typing
from abc import ABC

from git import Commit
from keybert import KeyBERT
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.llms import BaseLLM
from langchain.output_parsers.json import parse_partial_json
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from dev_rewind.config import KeywordAlgo
from dev_rewind.core.stopword import stopword_set


class SummaryEngine(ABC):
    def summarize_commits(self, commits: typing.Iterable[Commit], *args, **kwargs):
        return self.summarize_docs(
            (Document(page_content=each.message) for each in commits), *args, **kwargs
        )

    def summarize_docs(
        self, docs: typing.Iterable[Document], *args, **kwargs
    ) -> typing.List[str]:
        raise NotImplementedError


class LLMSummaryEngine(SummaryEngine):
    def summarize_docs(
        self,
        docs: typing.Iterable[Document],
        keyword_limit: int,
        llm: BaseLLM,
        *_,
        **__,
    ) -> typing.List[str]:
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
        raw_output = stuff_chain.run(docs)
        possible_json_list: typing.List[str] = parse_partial_json(raw_output)
        possible_json_list.sort()
        return possible_json_list


class BertSummaryEngine(SummaryEngine):
    def summarize_docs(
        self, docs: typing.Iterable[Document], keyword_limit: int, *_, **__
    ) -> typing.List[str]:
        # todo: maybe should in a global context for better results?

        # create model for extracting
        kw_model = KeyBERT()
        # supress warning
        os.putenv("TOKENIZERS_PARALLELISM", "False")

        # convert to list for keybert
        stopword_list = list(stopword_set)

        keywords_list = kw_model.extract_keywords(
            list((each.page_content for each in docs)),
            use_mmr=True,
            top_n=keyword_limit,
            stop_words=stopword_list,
        )

        tokens = set()
        for each_keywords in keywords_list:
            if isinstance(each_keywords, tuple):
                tokens.add(each_keywords[0])
            else:
                for each_keyword in each_keywords:
                    tokens.add(each_keyword[0])
        return list(tokens)


def get_summary_engine(engine_type: KeywordAlgo):
    return {
        KeywordAlgo.BERT: BertSummaryEngine(),
        KeywordAlgo.LLM: LLMSummaryEngine(),
    }[engine_type]
