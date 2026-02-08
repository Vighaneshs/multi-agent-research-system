"""Base agent — all specialized agents inherit from this."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


class BaseAgent(ABC):

    def __init__(self, model=None, temperature=None, tools=None):
        self.model_name = model or settings.default_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.tools = tools or []

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=settings.google_api_key,
        )
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def create_prompt_template(self, human_template: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", human_template)
        ])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def invoke_llm(self, messages: List[Dict]) -> str:
        """Invoke with exponential-backoff retry to handle rate limits."""
        response = self.llm.invoke(messages)
        return response.content

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{self.name}(model={self.model_name}, temp={self.temperature})"
