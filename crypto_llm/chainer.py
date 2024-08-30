import os
import logging
from typing import List, Any
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from crypto_llm.vectorizer import FAISSVectorizer
from crypto_llm.prompter import QuestionPrompter, SummaryPrompter
from crypto_llm.model import NvidiaModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LlmChainer:
    def __init__(
        self,
        llm_name: str = "meta/llama-3.1-405b-instruct",
    ):
        self.summary_path = os.getenv("DATA_PATH") + "summaries/"
        self.vectorizer = FAISSVectorizer()
        self.llm = NvidiaModel(llm_name).get_model()
        logger.info("LlmChainer initialized with retriever and LLM.")

    @staticmethod
    def format_docs(docs: List[Any]) -> str:
        formatted_docs = "\n\n".join([d.page_content for d in docs])
        logger.debug("Formatted documents: %s", formatted_docs)
        return formatted_docs

    def create_chain(self, retriever, prompt):
        logger.info("Creating chain.")
        chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("Chain created.")
        return chain

    def check_summary_exists(self, currency_name: str) -> bool:
        logger.info("Checking if summary exists for: %s", currency_name)
        return os.path.exists(self.summary_path + currency_name + ".txt")

    def get_summary(self, currency_name: str) -> str:
        logger.info("Getting summary for: %s", currency_name)
        with open(self.summary_path + currency_name + ".txt", "r") as f:
            return f.read()

    def save_summary(self, currency_name: str, summary: str) -> None:
        logger.info("Saving summary for: %s", currency_name)
        with open(self.summary_path + currency_name + ".txt", "w") as f:
            f.write(summary)

    def run_chain(
        self, currency_name: str, question: str = " ", is_summary: bool = False
    ) -> str:
        logger.info("Running chain with question: %s", question)
        retriever = self.vectorizer.get_retriever(
            name=currency_name, is_summary=is_summary
        )
        if not retriever:
            logger.warning("Retriever not found for: %s", currency_name)
            return None
        if is_summary:
            if self.check_summary_exists(currency_name):
                return self.get_summary(currency_name)
            prompt = SummaryPrompter().get_prompt()
        else:
            prompt = QuestionPrompter().get_prompt()
        logger.info("Running chain with prompt: %s", prompt)
        chain = self.create_chain(retriever, prompt)
        result = chain.invoke(question)
        logger.info("Chain run completed with result: %s", result)
        if is_summary:
            self.save_summary(currency_name, result)
        return result


# Пример использования класса LlmChainer
if __name__ == "__main__":
    chainer = LlmChainer()

    question_example = "Какой алгоритм консенсуса в SOLANA?"
    result = chainer.run_chain("USDC", question="Какой принцип работы у криптовалюты?")
    print(result)
