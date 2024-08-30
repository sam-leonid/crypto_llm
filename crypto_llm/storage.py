import abc
import logging
import os
import pandas as pd
from crypto_llm.loader import WhitePaperLoader, CMCLoader
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseStorage(abc.ABC):
    @abc.abstractmethod
    def get_wp_info(self):
        pass

    @abc.abstractmethod
    def get_symbol_by_name(self):
        pass

    @abc.abstractmethod
    def save_cmc_info(self):
        pass

    @abc.abstractmethod
    def get_pdf_whitepaper_link(self):
        pass


class FileStorage(BaseStorage):
    def __init__(self):
        self.wp_loader = WhitePaperLoader()
        self.cmc_loader = CMCLoader()
        self.path = os.getenv("DATA_PATH") + "sources/cmc/"
        self.cmc_list_path = os.path.join(self.path, "cmc_list.csv")
        self.cmc_detailed_info_path = os.path.join(self.path, "cmc_info.csv")
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
        logger.info("Storage initialized")

    def get_all_cmc_list(
        self,
        max_limit: int = 10_000,
        step=5_000,
        sleep_time: int = 35,
        iters_wait: int = 2,
    ) -> None:
        self.cmc_list = self.cmc_loader.get_all_cmc_list(
            max_limit, step, sleep_time, iters_wait
        )

    def get_wp_info(self, currency_name: str, pdf_link: str):
        return self.wp_loader.get_info(name=currency_name, link=pdf_link)

    def get_symbol_by_name(self, currency_name: str) -> None:
        self.cmc_detailed_info = self.cmc_loader.get_info_by_name(
            cmc_detailed_info=self.cmc_detailed_info,
            cmc_list=self.cmc_list,
            currency_name=currency_name,
        )

    def save_cmc_info(self):
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

    def show_all_currency_names(self) -> List:
        return self.cmc_list["name"].tolist()

    def get_pdf_whitepaper_link(self, name: str) -> List:
        result = self.cmc_detailed_info[
            (self.cmc_detailed_info["name"] == name)
            & (self.cmc_detailed_info["technical_doc"].str.endswith(".pdf"))
        ][["name", "technical_doc"]].values.tolist()
        return result[0] if result else [None, None]

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
