from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.llms.base import LLM
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from loguru import logger

from dev_rewind.core.workflow.retriever import RetrieverLayer


class ChainLayer(RetrieverLayer):
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
