import abc
import os
import pickle
import logging
from tqdm import tqdm
from typing import List
from langchain.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from crypto_llm.storage import FileStorage

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseVectorizer(abc.ABC):
    @abc.abstractmethod
    def calc_and_save_embedding(self):
        pass

    @abc.abstractmethod
    def calc_and_save_embedding_batch(self, name: str):
        # check if whitepaper exists and cmc description exists
        pass

    @abc.abstractmethod
    def get_retriever(self, name: str):
        pass


class FAISSVectorizer(BaseVectorizer):
    def __init__(self, model_name: str = "cointegrated/LaBSE-en-ru"):
        self.storage = FileStorage()
        self.wp_path = os.getenv("DATA_PATH") + "sources/whitepapers/"
        self.embedding_path = os.getenv("DATA_PATH") + "embeddings/"
        self.embedder = NVIDIAEmbeddings(
            model="nvidia/nv-embed-v1", api_key=os.getenv("NVIDIA_API_KEY")
        )
        logger.info("FAISSVectorizer initialized with model: %s", model_name)

    def calc_and_save_embedding(self, currency_name: str) -> bool:
        """
        Calculate and save embedding for a given currency_name.

        Args:
            currency_name (str): The currency_name of the whitepaper.

        Returns:
            None.
        """
        logger.info("Calculating and saving embedding for: %s", currency_name)
        if os.path.exists(self.embedding_path + currency_name):
            logger.info("Embedding already exists for: %s", currency_name)
        else:
            if not os.path.exists(self.wp_path + currency_name):
                logger.warning("Whitepaper not found for: %s", currency_name)
                self.storage.get_symbol_by_name(currency_name)
                self.storage.save_cmc_info()
                _, pdf_link = self.storage.get_pdf_whitepaper_link(currency_name)
                if not self.storage.get_wp_info(
                    currency_name=currency_name, pdf_link=pdf_link
                ):
                    logger.warning("PDF link not found for: %s", currency_name)
                    return False
            with open(self.wp_path + currency_name + ".pkl", "rb") as fp:
                whitepaper_raw = pickle.load(fp)
            db = FAISS.from_documents(whitepaper_raw, self.embedder)
            db.save_local(self.embedding_path + currency_name)
            logger.info("Embedding saved for: %s", currency_name)
        return True

    def calc_and_save_embedding_batch(self, currency_names: List[str]) -> None:
        """
        Calculate and save embeddings for a list of currency_names.

        Args:
            currency_names (List[str]): A list of currency_names of whitepapers.

        Returns:
            None.
        """
        logger.info("Calculating and saving embeddings for batch: %s", currency_names)
        for name in tqdm(currency_names):
            self.calc_and_save_embedding(name)
        logger.info("Batch processing complete.")

    def get_retriever(
        self,
        name: str,
        search_type: str = "similarity",
        is_summary: bool = False,
        k: int = 8,
    ):
        """
        Get a retriever for a given name.

        Args:
            name (str): The name of the whitepaper.
            search_type (str, optional): The type of search to perform. Defaults to "similarity".
            k (int, optional): The number of results to return. Defaults to 8.

        Returns:
            langchain.vectorstores.faiss.FAISS: The FAISS retriever.
        """
        if self.calc_and_save_embedding(name):
            if is_summary:
                k = 9999
            logger.info(
                "Getting retriever for: %s with search type: %s and k: %d",
                name,
                search_type,
                k,
            )
            db = FAISS.load_local(
                self.embedding_path + name,
                self.embedder,
                allow_dangerous_deserialization=True,
            )
            logger.info("Retriever obtained for: %s", name)
            return db.as_retriever(search_type="similarity", search_kwargs={"k": 1000})
        else:
            logger.warning("Embedding not found for: %s", name)
            return None
