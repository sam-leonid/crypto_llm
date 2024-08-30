import abc
import logging
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def get_model(self):
        pass


class NvidiaModel(BaseModel):
    def __init__(self, model_name: str = "meta/llama-3.1-405b-instruct"):
        self.model_name = model_name
        self.llm = ChatNVIDIA(
            model=self.model_name, nvidia_api_key=os.getenv("NVIDIA_API_KEY")
        )
        logger.info("NvidiaModel initialized with model: %s", model_name)

    def get_model(self):
        return self.llm
