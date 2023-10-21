from langchain.chains import RetrievalQA, LLMChain, ConversationChain, MultiPromptChain
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

    commit_prompt_template = """
You are a codebase analyzer.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""
    author_prompt_template = commit_prompt_template
    file_prompt_template = commit_prompt_template
    other_prompt_template = commit_prompt_template
    prompt_infos = [
        {
            "name": RetrieverType.COMMIT,
            "description": "Good for answering questions about commit",
            "prompt_template": commit_prompt_template,
        },
        {
            "name": RetrieverType.AUTHOR,
            "description": "Good for answering questions about author",
            "prompt_template": author_prompt_template,
        },
        {
            "name": RetrieverType.FILE,
            "description": "Good for answering questions about file/dir",
            "prompt_template": file_prompt_template,
        },
        {
            "name": "others",
            "description": "Good for answering pointless questions",
            "prompt_template": other_prompt_template,
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
                chain = self.create_stuff_chain(
                    llm, self.create_single_retriever(RetrieverType.FILE, ctx=ctx)
                )
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

    def create_stuff_chain(
        self, llm: LLM = None, retriever: BaseRetriever = None, **kwargs
    ) -> Chain:
        if not llm:
            llm = OpenAI()
        if not retriever:
            retriever = self.create_ensemble_retriever()

        prompt_template = """
You are a codebase analyzer.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
        """
        global_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )
        doc_prompt = PromptTemplate(
            input_variables=["page_content"],
            template="{page_content}",
        )
        chain_type_kwargs = {
            "prompt": global_prompt,
            "document_prompt": doc_prompt,
            **kwargs,
        }
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            memory=ConversationBufferWindowMemory(k=3),
        )
        logger.debug("chain created")
        return chain

    def create_mapreduce_chain(
        self, llm: LLM = None, retriever: BaseRetriever = None, **kwargs
    ) -> Chain:
        if not llm:
            llm = OpenAI()
        if not retriever:
            retriever = self.create_ensemble_retriever()

        # thanks: https://github.com/langchain-ai/langchain/issues/5096
        combined_prompt_template = """
You are a codebase analyzer.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{summaries}

Question: {question}
                """
        combined_prompt = PromptTemplate(
            template=combined_prompt_template,
            input_variables=["summaries", "question"],
        )

        map_prompt_template = """
Make a summary for these documents for question: {question}
Keep the summary as accurate and concise as possible.

{context}
"""
        question_prompt = PromptTemplate(
            template=map_prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = {
            "question_prompt": question_prompt,
            "combine_prompt": combined_prompt,
            **kwargs,
        }
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="map_reduce",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            memory=ConversationBufferWindowMemory(k=3),
        )
        logger.debug("mapreduce chain created")
        return chain

    def create_refine_chain(
        self, llm: LLM = None, retriever: BaseRetriever = None, **kwargs
    ) -> Chain:
        if not llm:
            llm = OpenAI()
        if not retriever:
            retriever = self.create_ensemble_retriever()

        # thanks: https://github.com/langchain-ai/langchain/issues/5096
        question_prompt = """
You are a codebase analyzer.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context_str}

Question: {question}
                        """
        question_prompt = PromptTemplate(
            template=question_prompt,
            input_variables=["context_str", "question"],
        )

        chain_type_kwargs = {"question_prompt": question_prompt, **kwargs}
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="refine",
            retriever=retriever,
            chain_type_kwargs=chain_type_kwargs,
            memory=ConversationBufferWindowMemory(k=3),
        )
        logger.debug("refine chain created")
        return chain
