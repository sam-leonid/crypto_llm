import abc
from langchain.prompts.chat import ChatPromptTemplate


class BasePrompter(abc.ABC):
    @abc.abstractmethod
    def get_prompt(self):
        pass


class SummaryPrompter(BasePrompter):
    def __init__(self):
        self.template = """
        Answer the question based only on the following context. 
        Summarize the most important information about the cryptocurrency described in the whitepaper, focusing on its core features, technology, use cases, and potential. 
        **Limit your response to a maximum of 1000 characters.** 
        Prioritize conciseness and clarity. 

        Context:

        {context}

        Question: What are the key features and highlights of this cryptocurrency? 
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def get_prompt(self):
        return self.prompt


class QuestionPrompter(BasePrompter):
    def __init__(self):
        self.template = """
        Answer the question based only on the following context. Keep the answer short and concise.
        **Provide only the most relevant and concise answer.** 
        Do not generate any additional text unless explicitly asked to do so.

        Context:

        {context}

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def get_prompt(self, mode="base"):
        return self.prompt
