import typing

from langchain.agents import AgentExecutor, ConversationalChatAgent, AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.json import parse_json_markdown
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish, Document
from langchain.tools import BaseTool
from loguru import logger
from rapidfuzz import fuzz

from dev_rewind.core.collector import CollectorLayer
from dev_rewind.core.context import RuntimeContext


class CustomOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> typing.Union[AgentAction, AgentFinish]:
        # contains some prefix ...
        # really weird.
        if "```" not in text:
            return AgentFinish({"output": text}, "")
        ptr = text.index("```")
        text = text[ptr:]
        if not text.endswith("```"):
            text += "`"

        response = parse_json_markdown(text)

        action_value = response["action"]
        action_input_value = response["action_input"]
        if action_value == "Final Answer":
            return AgentFinish({"output": action_input_value}, text)
        else:
            return AgentAction(action_value, action_input_value, text)


class AgentLayer(CollectorLayer):
    def create_agent(self, runtime_context: RuntimeContext = None) -> AgentExecutor:
        if not runtime_context:
            runtime_context = self.collect_metadata()

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        tools = self.create_tools(runtime_context)
        chat_agent = ConversationalChatAgent.from_llm_and_tools(
            system_message="""
You are a codebase analyzer.

- Try more tools for answering.
- If you don't know the answer, just say that you don't know, don't try to make up an answer.

When a user asks something about a file, you should ALWAYS take these steps:

1. Use custom keyword tool for extracting the keywords from the file
2. Deduce the functionality of the file based on keywords and its name
3. Finally try to answer the question
            """,
            llm=OpenAI(),
            tools=tools,
            memory=memory,
            verbose=True,
            output_parser=CustomOutputParser(),
        )

        class CustomAgentExecutor(AgentExecutor):
            def run(self, *args, **kwargs) -> typing.Any:
                # hacky way from https://github.com/langchain-ai/langchain/issues/1358#issuecomment-1486132587
                try:
                    response = super().run(*args, **kwargs)
                except ValueError as e:
                    response = str(e)
                    if not response.startswith("Could not parse LLM output:"):
                        raise e
                    response = response.removeprefix(
                        "Could not parse LLM output:"
                    ).removesuffix("`")
                return response

        agent = CustomAgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=tools,
            verbose=True,
            memory=memory,
            max_iterations=3,
        )
        return agent

    def create_tools(self, runtime_context: RuntimeContext) -> typing.List[BaseTool]:
        class KeywordTool(BaseTool):
            name = "keyword"
            description = """
    Return keywords of a valid file for understanding file content better.
    Always use this if you need to summarize a file.
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

            def _run(self, file_name: str) -> typing.List[str]:
                # todo: change this return to an object
                if file_name not in runtime_context.files:
                    logger.warning(f"{file_name} not in {runtime_context.files}")
                    possible_file_name, score = self.find_best_match(
                        file_name, list(runtime_context.files.keys())
                    )
                    # todo: ai did not know the real file name
                    if score < 80.0:
                        logger.warning(f"all the files did not match {file_name}")
                        return []
                    file_name = possible_file_name

                # start a chain for summarizing
                # todo: configable
                prompt_template = """
You are a codebase analyzer.
The following commits comes from git blame of a file:

"{text}"

Generate <10 keywords to represent/predict the features of this file.
You should ignore everything unrelated to features like: refactor/optimization/updates/improvement ...
Response should be a json list.
Keywords:
                """
                prompt = PromptTemplate.from_template(prompt_template)

                # Define LLM chain
                # todo: not only openai
                llm_chain = LLMChain(llm=OpenAI(), prompt=prompt, verbose=True)

                # Define StuffDocumentsChain
                stuff_chain = StuffDocumentsChain(
                    llm_chain=llm_chain, document_variable_name="text", verbose=True
                )

                file_context = runtime_context.files[file_name]

                llm_resp = stuff_chain.run(
                    [
                        Document(page_content=each.message)
                        for each in file_context.commits
                    ]
                )
                return llm_resp

        return [KeywordTool()]
