import typing

from langchain.chains import RetrievalQA, ConversationChain, MultiPromptChain, SequentialChain, TransformChain
from langchain.chains.base import Chain
from langchain.chains.router.llm_router import RouterOutputParser, LLMRouterChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.llms.base import LLM
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from loguru import logger

from dev_rewind.core.context import RuntimeContext
from dev_rewind.core.workflow.retriever import RetrieverLayer, RetrieverType


class ChainLayer(RetrieverLayer):
    """
    for seamlessly integrating with langchain

    question -> router -> commit   kind: "What func did 123456F changed?"
                       -> author   kind: "What func did williamfzc changed?"
                       -> file     kind: "What func did a/b/c.py changed?"
                       -> others

    commit: search commit_db, then file_db with name
    author: search commit_db, then file_db with name
    file:   search file_db, then commit_db with blames
    others: search all
    """

    prompt_infos = [
        {
            "name": RetrieverType.COMMIT,
            "description": "Good for answering questions about commit",
        },
        {
            "name": RetrieverType.AUTHOR,
            "description": "Good for answering questions about author",
        },
        {
            "name": RetrieverType.FILE,
            "description": "Good for answering questions about file/dir",
        },
        {
            "name": "others",
            "description": "Good for answering pointless questions",
        },
    ]

    def create_router_chain(self, llm: LLM = None, **kwargs) -> Chain:
        ctx = self.collect_metadata()
        if not llm:
            llm = OpenAI()

        destination_chains = {}
        default_chain = ConversationChain(llm=llm, output_key="text")
        for p_info in self.prompt_infos:
            name = p_info["name"]
            # prompt_template = p_info["prompt_template"]
            # prompt = PromptTemplate(template=prompt_template, input_variables=["input"])

            if name == RetrieverType.FILE:
                chain = self.create_file_first_search_chain(llm, ctx=ctx)
            elif name == RetrieverType.AUTHOR:
                chain = self.create_stuff_chain(
                    llm, self.create_single_retriever(RetrieverType.AUTHOR, ctx=ctx)
                )
            elif name == RetrieverType.COMMIT:
                chain = self.create_stuff_chain(
                    llm, self.create_single_retriever(RetrieverType.COMMIT, ctx=ctx)
                )
            else:
                chain = default_chain

            destination_chains[name] = chain

        router_chain = self.create_single_router_chain(llm=llm)
        chain = MultiPromptChain(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=default_chain,
            verbose=True,
        )
        return chain

    def create_single_router_chain(self, llm: LLM):
        destinations = [f"{p['name']}: {p['description']}" for p in self.prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        return router_chain

    def create_conclusion_chain(self) -> Chain:
        def transform_func(inputs: dict) -> dict:
            return {"result": str(inputs)}

        return TransformChain(
            input_variables=["file_context", "commit_context"],
            output_variables=["result"],
            transform=transform_func,
        )

    def create_file_first_search_chain(
            self, llm: LLM = None, ctx: RuntimeContext = None
    ) -> Chain:
        overall_chain = SequentialChain(
            chains=[
                self.create_stuff_chain(
                    llm,
                    self.create_single_retriever(RetrieverType.FILE, ctx=ctx),
                    output_key="file_context",
                    verbose=True,
                ),
                self.create_stuff_chain(
                    llm,
                    self.create_single_retriever(RetrieverType.COMMIT, ctx=ctx),
                    output_key="commit_context",
                    # TODO: bug here, can not handle any keys
                    additional_input_keys=[],
                    verbose=True,
                ),
                self.create_conclusion_chain(),
            ],
            input_variables=["input"],
            verbose=True,
        )
        return overall_chain


    def create_stuff_chain(
        self, llm: LLM = None, retriever: BaseRetriever = None, output_key: str = None, additional_input_keys: typing.List[str] = None, **kwargs
    ) -> Chain:
        if not llm:
            llm = OpenAI()
        if not retriever:
            retriever = self.create_ensemble_retriever()
        user_output_key = output_key

        prompt_template = """
You are a codebase analyzer.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
        if additional_input_keys:
            additional_str = "\n".join((f"{{{each}}}" for each in additional_input_keys))
            prompt_template = prompt_template.replace("{context}", f"{{context}}\n\n{additional_str}")
        logger.debug(f"prompt template: {prompt_template}")

        global_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"] + (additional_input_keys or []),
        )
        doc_prompt = PromptTemplate(
            template="{page_content}",
            input_variables=["page_content"],
        )
        chain_type_kwargs = {
            "prompt": global_prompt,
            "document_prompt": doc_prompt,
            **kwargs,
        }

        # https://github.com/langchain-ai/langchain/discussions/8668
        class CustomRetrievalQA(RetrievalQA):
            input_key: str = "input"
            output_key: str = user_output_key  #: :meta private:

            @property
            def output_keys(self) -> typing.List[str]:
                _output_keys = [self.output_key]
                return _output_keys

            @property
            def input_keys(self) -> typing.List[str]:
                return [self.input_key, *(additional_input_keys or [])]

        chain = CustomRetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
        )
        logger.debug(f"chain created, io: {chain.input_keys} / {chain.output_keys}")
        return chain
