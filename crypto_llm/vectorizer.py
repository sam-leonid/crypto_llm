import abc
import os
import pickle
import logging
from tqdm import tqdm
from typing import List
from langchain.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

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
        self.wp_path = os.getenv("DATA_PATH") + "sources/whitepapers/"
        self.embedding_path = os.getenv("DATA_PATH") + "embeddings/"
        self.embedder = NVIDIAEmbeddings(
            model="nvidia/nv-embed-v1", api_key=os.getenv("NVIDIA_API_KEY")
        )
        logger.info("FAISSVectorizer initialized with model: %s", model_name)

    def calc_and_save_embedding(self, name: str) -> None:
        """
        Calculate and save embedding for a given name.

        Args:
            name (str): The name of the whitepaper.

        Returns:
            None.
        """
        logger.info("Calculating and saving embedding for: %s", name)
        if not os.path.exists(self.embedding_path + name):
            if os.path.exists(self.wp_path + name):
                with open(self.wp_path + name, "rb") as fp:
                    whitepaper_raw = pickle.load(fp)
                db = FAISS.from_documents(whitepaper_raw, self.embedder)
                db.save_local(self.embedding_path + name)
                logger.info("Embedding saved for: %s", name)
            else:
                logger.warning("Whitepaper not found for: %s", name)
        else:
            logger.info("Embedding already exists for: %s", name)

    def calc_and_save_embedding_batch(self, names: List[str]) -> None:
        """
        Calculate and save embeddings for a list of names.

        Args:
            names (List[str]): A list of names of whitepapers.

        Returns:
            None.
        """
        logger.info("Calculating and saving embeddings for batch: %s", names)
        for name in tqdm(names):
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
