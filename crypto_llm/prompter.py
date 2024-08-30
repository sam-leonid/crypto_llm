import abc
from langchain.prompts.chat import ChatPromptTemplate


class BasePrompter(abc.ABC):
    @abc.abstractmethod
    def get_prompt(self):
        pass


class SummaryPrompter(BasePrompter):
    def __init__(self):
        self.template = """
            Ответить на вопрос, основываясь только на следующем контексте. 
            Кратко изложите наиболее важную информацию о криптовалюте, описанной в whitepaper, сосредоточившись на ее основных функциях, технологии, вариантах использования и потенциале. 
            **Ограничьте свой ответ максимум 1000 символами.** 
            В приоритете краткость и ясность.
            **Отвечайте на русском языке.**
            **Используйте только латинские или кириллические символы.**
            **Допустимо также использовать цифры и знаки пунктуации.**

            **Форматируйте свой ответ, выделяя каждую ключевую особенность или важный момент с новой строки для удобства чтения.**

            Контекст:

            {context}

            Вопрос: Каковы ключевые особенности и преимущества этой криптовалюты? 

            Ответ:
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
