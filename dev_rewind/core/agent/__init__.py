import typing

from langchain.agents import AgentExecutor, ConversationalChatAgent, AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.llms import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.json import parse_json_markdown, parse_partial_json
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool

from dev_rewind.core.agent.tools import create_keyword_tool
from dev_rewind.core.collector import CollectorLayer
from dev_rewind.core.context import RuntimeContext


class CustomOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def _process_dict(
        self, response: dict, text: str
    ) -> typing.Union[AgentAction, AgentFinish]:
        # resp to action
        action_value = response["action"]
        action_input_value = response["action_input"]
        if action_value == "Final Answer":
            return AgentFinish({"output": action_input_value}, text)
        else:
            return AgentAction(action_value, action_input_value, text)

    def parse(self, text: str) -> typing.Union[AgentAction, AgentFinish]:
        # contains some prefix ...
        # really weird.

        # try raw json first
        response = parse_partial_json(text)
        if response:
            return self._process_dict(response, text)

        # try markdown json
        response = parse_json_markdown(text)
        if response:
            return self._process_dict(response, text)

        # still failed, do some hack ...
        if "```" not in text:
            return AgentFinish({"output": text}, "")
        ptr = text.index("```")
        text = text[ptr:]
        if not text.endswith("```"):
            text += "`"

        response = parse_json_markdown(text)
        if response:
            return self._process_dict(response, text)


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


class AgentLayer(CollectorLayer):
    def create_agent(
        self, runtime_context: RuntimeContext = None, custom_llm: BaseLLM = None
    ) -> AgentExecutor:
        if not runtime_context:
            runtime_context = self.collect_metadata()
        if not custom_llm:
            # maybe should be a chat model
            custom_llm = OpenAI(temperature=0.1)

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        custom_tools = self.create_tools(runtime_context, custom_llm)
        chat_agent = ConversationalChatAgent.from_llm_and_tools(
            system_message="""
You are a codebase analyzer.

- Try more tools for answering.
- If you don't know the answer, just say that you don't know, don't try to make up an answer.

When a user asks something about a file, you should ALWAYS take these steps:

1. Use custom keyword tool for extracting the keywords from the file.
2. Deduce the functionality of the file based on keywords and its name.
3. Finally try to answer the question.
""",
            llm=custom_llm,
            tools=custom_tools,
            memory=memory,
            verbose=True,
            output_parser=CustomOutputParser(),
        )

        agent_executor = CustomAgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=custom_tools,
            verbose=True,
            memory=memory,
            max_iterations=3,
        )
        return agent_executor

    def create_tools(
        self, runtime_context: RuntimeContext, custom_llm: BaseLLM
    ) -> typing.List[BaseTool]:
        return [
            create_keyword_tool(self.config, runtime_context, custom_llm),
        ]
