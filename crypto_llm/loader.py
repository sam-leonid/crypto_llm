import abc
import coinmarketcapapi
import pandas as pd
import os
from tqdm import tqdm
from time import sleep
import logging
import pickle
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseLoader(abc.ABC):
    @abc.abstractmethod
    def get_info_batch(self):
        pass

    @abc.abstractmethod
    def get_info(self):
        pass

    @abc.abstractmethod
    def save_info(self):
        pass


class WhitePaperLoader(BaseLoader):
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        is_separator_regex: bool = False,
    ):
        self.path = os.getenv("DATA_PATH") + "sources/whitepapers/"
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=is_separator_regex,
        )

    def get_info(self, name: str, link: str):
        try:
            data = PyPDFLoader(link).load()
            data = self.split_text(data)
            self.save_info(name, data)
        except Exception as e:
            logger.warning(f"Error fetching info for {link}. " + str(e))

    def get_info_batch(self, list_of_links: List):
        for name, link in tqdm(list_of_links):
            self.get_info(name, link)

    def split_text(self, data: List[Document]) -> List[Document]:
        return self.splitter.split_documents(data)

    def save_info(self, name: str, data: List[Document]):
        with open(self.path + name, "wb") as fp:
            pickle.dump(data, fp)


class CMCLoader(BaseLoader):
    def __init__(self):
        self.cmc_client = coinmarketcapapi.CoinMarketCapAPI(os.getenv("CMC_API_KEY"))
        self.path = os.getenv("DATA_PATH") + "sources/cmc/"
        self.cmc_list_path = self.path + "cmc_list.csv"
        self.cmc_detailed_info_path = self.path + "cmc_info.csv"
        self.cmc_list = (
            pd.read_csv(self.cmc_list_path)
            if os.path.exists(self.cmc_list_path)
            else None
        )
        self.cmc_detailed_info = (
            pd.read_csv(self.cmc_detailed_info_path)
            if os.path.exists(self.cmc_detailed_info_path)
            else pd.DataFrame(columns=["symbol"])
        )
        logger.info("CMCLoader initialized")

    def get_cmc_list(self, limit: int = 5) -> pd.DataFrame:
        """
        Fetches CMC list with given limit and saves it to `self.cmc_list`

        Args:
            limit (int): limit of items to fetch from CMC

        Returns:
            pd.DataFrame: DataFrame with CMC list
        """
        logger.info(f"Fetching CMC list with limit {limit}")
        response = self.cmc_client.cryptocurrency_listings_latest(limit=limit)
        self.cmc_list = pd.DataFrame(response.data)
        logger.info(f"Fetched {len(self.cmc_list)} items from CMC")
        return self.cmc_list

    @staticmethod
    def preprocess_data(data):
        for i in data:
            if "urls" in i:
                i.update(i["urls"])
                del i["urls"]
            i["technical_doc"] = i["technical_doc"][0] if i["technical_doc"] else " "
        return data

        return data

    def get_info(self, sym: str, sleep_time: int = 35, iters_wait: int = 2):
        """
        Fetches detailed info for a given symbol from CMC.

        Args:
            sym (str): symbol to fetch info for
            sleep_time (int): time to sleep between attempts in case of errors
            iters_wait (int): number of attempts to fetch info before giving up

        Returns:
            pd.DataFrame: DataFrame with detailed info, or None if failed to fetch
        """
        if sym not in self.cmc_detailed_info["symbol"].values:
            exc_rate = 0
            while True:
                try:
                    data = self.cmc_client.cryptocurrency_info(symbol=sym).data[sym]
                    data = self.preprocess_data(data)
                    logger.debug(f"Successfully fetched info for {sym}")
                    self.cmc_detailed_info = pd.concat(
                        [self.cmc_detailed_info, pd.DataFrame(data)]
                    )
                    return
                except Exception as e:
                    exc_rate += 1
                    logger.warning(
                        f"Error fetching info for {sym}. "
                        + f"Attempt {str(exc_rate)}. Error: {str(e)}"
                    )
                    if exc_rate == iters_wait:
                        logger.error(
                            f"Failed to fetch info for {sym}"
                            + f"after {iters_wait} attempts"
                        )
                        return
                    sleep(sleep_time)
        else:
            logger.info(f"Already fetched info for {sym}")

    def get_info_batch(self, sleep_time: int = 35, iters_wait: int = 2):
        """
        Fetches detailed info for all symbols in `self.cmc_list`.

        Args:
            sleep_time (int): time to sleep between attempts in case of errors
            iters_wait (int): number of attempts to fetch info before giving up

        Returns:
            pd.DataFrame: Concatenated DataFrame with detailed info for all symbols
        """
        logger.info("Starting to fetch detailed info")

        for sym in tqdm(self.cmc_list["symbol"].str.upper().unique()):
            self.get_info(sym=sym, sleep_time=sleep_time, iters_wait=iters_wait)

        logger.info("Finished fetching detailed info")
        return self.cmc_detailed_info

    def save_info(self):
        """
        Saves CMC list and detailed CMC info to csv files.

        Args:
            None

        Returns:
            None
        """
        if self.cmc_list is not None:
            self.cmc_list.to_csv(self.cmc_list_path, index=False)
            logger.info(f"Saved CMC list to {self.cmc_list_path}")
        else:
            logger.warning("CMC list is None, not saving")

        if self.cmc_detailed_info is not None:
            self.cmc_detailed_info.to_csv(self.cmc_detailed_info_path, index=False)
            logger.info(f"Saved detailed CMC info to {self.cmc_detailed_info_path}")
        else:
            logger.warning("Detailed CMC info is None, not saving")

    def get_pdf_whitepaper(self, name: str) -> List:
        return self.cmc_detailed_info[
            (self.cmc_detailed_info["name"] == name)
            and (self.cmc_detailed_info["technical_doc"].str.endswith(".pdf"))
        ][["name", "technical_doc"]].values.tolist()

    def get_all_pdf_whitepapers(self) -> List:
        return self.cmc_detailed_info[
            self.cmc_detailed_info["technical_doc"].str.endswith(".pdf")
        ][["name", "technical_doc"]].values.tolist()

    def get_description(self, name: str) -> List:
        return self.cmc_detailed_info[(self.cmc_detailed_info["name"] == name)][
            ["name", "technical_doc"]
        ].values.tolist()

    def get_all_descriptions(self) -> List:
        return self.cmc_detailed_info[["name", "technical_doc"]].values.tolist()
